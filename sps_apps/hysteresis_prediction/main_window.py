import logging
from typing import Optional
from uuid import uuid4

import numpy as np
from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from qtpy import QtGui, QtWidgets

from .core.application_context import context
from .data import Acquisition, BufferData, CycleData
from .generated.main_window_ui import Ui_main_window
from .inference import Inference
from .io import IO
from .utils import load_cursor
from .widgets import ModelLoadDialog, PlotModel
from .widgets.plot_settings_widget import AppStatus
from .widgets.prediction_analysis_widget import (
    PredictionAnalysisModel,
    PredictionAnalysisWidget,
)
from .widgets.status_tracker import StatusManager
from .widgets.trim_widget import TrimModel, TrimWidgetView

log = logging.getLogger(__name__)

__all__ = ["MainWindow"]


BUFFER_SIZE = 150000


class MainWindow(Ui_main_window, ApplicationFrame):
    def __init__(
        self,
        buffer_size: int = BUFFER_SIZE,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        ApplicationFrame.__init__(self, parent)
        Ui_main_window.__init__(self)

        self.setupUi(self)

        self._analysis_widgets: dict[str, PredictionAnalysisWidget] = {}
        self._trim_wide_widgets: dict[str, TrimWidgetView] = {}

        log_console = LogConsole(self)
        self.log_console = log_console
        log_console.toggleExpandedMode()

        timing_model = TimingBarModel(
            domain=TimingBarDomain.SPS, japc=context.japc
        )
        timing_bar = TimingBar(self, model=timing_model)
        self.timing_bar = timing_bar

        self._buffer_size = buffer_size
        self._acquisition = Acquisition(min_buffer_size=buffer_size)
        self._inference = Inference(parent=self)

        self._status_manager = StatusManager(self)

        plot_model = PlotModel(self._acquisition, parent=self)
        self.widgetPlot.model = plot_model

        self._io = IO()

        self._connect_signals()
        self._connect_actions()

        # status messages
        self._connect_status()

        self._acquisition.run()

    def _connect_signals(self) -> None:
        self.widgetSettings.timespan_changed.connect(
            self.widgetPlot.set_time_span
        )

        self.widgetSettings.downsample_changed.connect(
            self.widgetPlot.model.set_downsample
        )
        self.widgetSettings.toggle_predictions.connect(
            self._inference.set_do_inference
        )
        self._inference.cycle_predicted.connect(
            self.widgetPlot.model.new_predicted_cycle
        )
        self._inference.cycle_predicted.connect(self.on_new_prediction)
        self._acquisition.new_buffer_data.connect(
            self._inference.predict_last_cycle
        )
        self._acquisition.cycle_started.connect(
            self.widgetSettings._on_new_cycle
        )
        self._acquisition.buffer.buffer_size_changed.connect(
            lambda x: self.widgetSettings._set_progressbar(
                x, self._buffer_size
            )
        )

        self._inference.model_loaded.connect(
            self.widgetSettings.on_model_loaded
        )
        self._inference.model_loaded.connect(
            lambda: QtWidgets.QMessageBox.information(
                self, "Model loaded", "Model successfully loaded."
            )
        )

    def _connect_actions(self) -> None:
        self.actionShow_Plot_Settings.triggered.connect(
            self.toggle_plot_settings
        )
        self.actionContinuous_Data_Export.toggled.connect(self._io.set_enabled)
        self.action_Clear_Buffer.triggered.connect(
            self._acquisition.buffer.reset_buffer
        )

        self.action_Load_Model.triggered.connect(self.on_load_model_triggered)

        self.actionPrediction_Analysis.triggered.connect(
            self.show_predicion_analysis
        )
        self.action_Trim_View.triggered.connect(self.show_trim_widget)

    def _connect_status(self) -> None:
        self.widgetSettings.toggle_predictions.connect(
            lambda enabled, *_: self._status_manager.statusChanged.emit(
                AppStatus.INFERENCE_ENABLED
                if enabled
                else AppStatus.INFERENCE_DISABLED
            )
        )
        self._inference.model_loaded.connect(
            lambda *_: self._status_manager.statusChanged.emit(
                AppStatus.MODEL_LOADED
            )
        )
        self._inference.started.connect(
            lambda *_: self._status_manager.statusChanged.emit(
                AppStatus.INFERENCE_RUNNING
            )
        )
        self._inference.completed.connect(
            lambda *_: self._status_manager.statusChanged.emit(
                AppStatus.INFERENCE_IDLE
            )
        )
        self._status_manager.setStatus.connect(
            self.widgetSettings.status_changed.emit
        )
        self._acquisition.buffer.buffer_size_changed.connect(
            lambda size, *_: self._status_manager.statusChanged.emit(
                AppStatus.BUFFER_WAITING
                if size < BUFFER_SIZE
                else AppStatus.BUFFER_FULL
            )
        )

        self._status_manager.statusChanged.emit(AppStatus.NO_MODEL)

    def on_load_model_triggered(self) -> None:
        dialog = ModelLoadDialog(parent=self)
        result = dialog.exec()

        if result == QtWidgets.QDialog.Rejected:
            log.debug("Model load dialog cancelled.")
            return
        elif result == QtWidgets.QDialog.Accepted:
            ckpt_path = dialog.ckpt_path
            device = dialog.device

            self._inference.load_model.emit(ckpt_path, device)

    def on_new_prediction(self, cycle_data: CycleData, prediction: np.ndarray):
        try:
            self._acquisition.buffer.dispatch_data(
                BufferData.REF_B,
                cycle_data.cycle,
                cycle_data.cycle_timestamp,
                prediction,
            )
        except:  # noqa: broad-except
            log.exception("An exception occurred while saving reference B.")
            return

    def toggle_plot_settings(self) -> None:
        if self.actionShow_Plot_Settings.isChecked():
            self.widgetSettings.show()
        else:
            self.widgetSettings.hide()

    def show_predicion_analysis(self) -> None:
        with load_cursor():
            model = PredictionAnalysisModel()
            widget = PredictionAnalysisWidget(model=model, parent=None)

            self._acquisition.new_measured_data.connect(model.newData.emit)

            uuid = str(uuid4())
            self._analysis_widgets[uuid] = widget

            def on_close() -> None:
                widget = self._analysis_widgets.pop(uuid)
                widget.deleteLater()

                self._acquisition.new_measured_data.disconnect(
                    model.newData.emit
                )

            widget.windowClosed.connect(on_close)
        widget.show()

    def show_trim_widget(self) -> None:
        with load_cursor():
            model = TrimModel()
            widget = TrimWidgetView(model=model, parent=None)

            self._inference.cycle_predicted.connect(
                model.newPredictedData.emit
            )

            uuid = str(uuid4())
            self._trim_wide_widgets[uuid] = widget

            def on_close() -> None:
                widget = self._trim_wide_widgets.pop(uuid)
                widget.deleteLater()

                self._inference.cycle_predicted.disconnect(
                    model.newPredictedData.emit
                )

            widget.windowClosed.connect(on_close)
        widget.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._acquisition.stop()
        super().closeEvent(event)
