from __future__ import annotations

import logging
from uuid import uuid4

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from op_app_context import context
from qtpy import QtGui, QtWidgets

from .data import Acquisition
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
        parent: QtWidgets.QWidget | None = None,
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
            domain=TimingBarDomain.SPS, japc=context.japc_client
        )
        timing_bar = TimingBar(self, model=timing_model)
        self.timing_bar = timing_bar

        self._acquisition = Acquisition(parent=self)
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
            self._acquisition.new_predicted_data
        )
        self._acquisition.new_prediction.connect(
            self.widgetPlot.model.onNewPredicted
        )
        self._acquisition.new_measured_data.connect(self._io.save_data)
        self._acquisition.new_buffer_data.connect(
            self._inference.predict_last_cycle
        )
        self._acquisition.cycle_started.connect(
            self.widgetSettings._on_new_cycle
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
        self.action_Clear_Reference.triggered.connect(
            self._acquisition.reset_reference
        )

        self.action_Load_Model.triggered.connect(self.on_load_model_triggered)
        self.actionProgrammed_current.triggered.connect(
            lambda x: self._inference.set_use_programmed_current(
                self.actionProgrammed_current.isChecked()
            )
        )
        self.actionAutoregressive.triggered.connect(
            lambda x: self._inference.set_autoregressive(
                self.actionAutoregressive.isChecked()
            )
        )
        self.actionReset_state.triggered.connect(
            self._inference._predictor.reset_state
        )

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

        self._status_manager.statusChanged.emit(AppStatus.NO_MODEL)

    def on_load_model_triggered(self) -> None:
        dialog = ModelLoadDialog(parent=self)
        dialog.load_checkpoint.connect(self._inference.load_model.emit)
        result = dialog.exec()

        if result == QtWidgets.QDialog.Rejected:
            log.debug("Model load dialog cancelled.")
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

            self._acquisition.new_prediction.connect(model.on_new_prediction)
            self.action_Clear_Reference.triggered.connect(
                model.reset_reference_fields
            )

            uuid = str(uuid4())
            self._trim_wide_widgets[uuid] = widget

            def on_close() -> None:
                widget = self._trim_wide_widgets.pop(uuid)
                widget.deleteLater()

                self._acquisition.new_prediction.disconnect(
                    model.on_new_prediction
                )
                self.action_Clear_Reference.triggered.disconnect(
                    model.reset_reference_fields
                )

            widget.windowClosed.connect(on_close)
        widget.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._acquisition.stop()
        super().closeEvent(event)
