from __future__ import annotations

import logging
import types
from uuid import uuid4

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from op_app_context import context
from qtpy import QtGui, QtWidgets

from ._data_flow import LocalDataFlow
from .generated.main_window_ui import Ui_main_window
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


class MainWindow(Ui_main_window, ApplicationFrame):
    def __init__(
        self,
        data_flow: LocalDataFlow,
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

        self._data = data_flow

        self._status_manager = StatusManager(self)

        plot_model = PlotModel(self._data, parent=self)
        self.widgetPlot.model = plot_model

        self._io = IO()

        self._connect_signals()
        self._connect_actions()

        # status messages
        self._connect_status()

        self.show_trim_widget()
        self.show_prediction_analysis()

    def _connect_signals(self) -> None:
        self.widgetSettings.timespan_changed.connect(self.widgetPlot.set_time_span)

        self.widgetSettings.downsample_changed.connect(
            self.widgetPlot.model.set_downsample
        )

        # for UI only
        self._data.onCycleStart.connect(self.widgetSettings.onNewCycle)
        self._data.onCycleCorrectionCalculated.connect(
            self.widgetPlot.model.onNewPredicted
        )

        self._data.onCycleMeasured.connect(self._io.save_data)

        self._data._predict.model_loaded.connect(
            lambda: QtWidgets.QMessageBox.information(
                self,
                "Model loaded",
                "Model successfully loaded.\nPredictions will now start.",
            )
        )
        self._data._predict.model_loaded.connect(
            lambda: self._data._predict.set_do_inference(True)
        )

    def _connect_actions(self) -> None:
        self.actionShow_Plot_Settings.triggered.connect(self.toggle_plot_settings)
        self.actionContinuous_Data_Export.toggled.connect(self._io.set_enabled)
        self.action_Clear_Reference.triggered.connect(self._data.resetReference)

        self.action_Load_Model.triggered.connect(self.on_load_model_triggered)
        self.actionProgrammed_current.triggered.connect(
            lambda x: self._data._predict.set_use_programmed_current(
                self.actionProgrammed_current.isChecked()
            )
        )
        self.actionAutoregressive.triggered.connect(
            lambda x: self._data._predict.set_autoregressive(
                self.actionAutoregressive.isChecked()
            )
        )
        self.actionReset_state.triggered.connect(self._data._predict.reset_state)

        self.actionPrediction_Analysis.triggered.connect(self.show_prediction_analysis)
        self.action_Trim_View.triggered.connect(self.show_trim_widget)

    def _connect_status(self) -> None:
        # self.widgetSettings.toggle_predictions.connect(
        #     lambda enabled, *_: self._status_manager.statusChanged.emit(
        #         AppStatus.INFERENCE_ENABLED if enabled else AppStatus.INFERENCE_DISABLED
        #     )
        # )
        self._data._predict.model_loaded.connect(
            lambda *_: self._status_manager.statusChanged.emit(AppStatus.MODEL_LOADED)
            and self._status_manager.statusChanged.emit(AppStatus.INFERENCE_ENABLED)
        )
        self._data._predict.predictionStarted.connect(
            lambda *_: self._status_manager.statusChanged.emit(
                AppStatus.INFERENCE_RUNNING
            )
        )
        self._data._predict.predictionFinished.connect(
            lambda *_: self._status_manager.statusChanged.emit(AppStatus.INFERENCE_IDLE)
        )
        self._status_manager.setStatus.connect(self.widgetSettings.status_changed.emit)

        self._status_manager.statusChanged.emit(AppStatus.NO_MODEL)

    def on_load_model_triggered(self) -> None:
        dialog = ModelLoadDialog(parent=self)
        dialog.load_checkpoint.connect(self._data._predict.loadModel)
        result = dialog.exec()

        if result == QtWidgets.QDialog.Rejected:
            log.debug("Model load dialog cancelled.")
            return

    def toggle_plot_settings(self) -> None:
        if self.actionShow_Plot_Settings.isChecked():
            self.widgetSettings.show()
        else:
            self.widgetSettings.hide()

    def show_prediction_analysis(self) -> None:
        if len(self._analysis_widgets) > 0:
            uuid = list(self._analysis_widgets.keys())[0]
            widget = self._analysis_widgets[uuid]
            widget.show()
            widget.raise_()

            return

        with load_cursor():
            model = PredictionAnalysisModel()
            widget = PredictionAnalysisWidget(model=model, parent=None)

            self._data.onCycleMeasured.connect(model.onNewMeasuredData)

            uuid = str(uuid4())
            self._analysis_widgets[uuid] = widget

            def closeEvent(self, event: QtGui.QCloseEvent) -> None:
                event.ignore()
                self.hide()

            widget.closeEvent = types.MethodType(closeEvent, widget)  # type: ignore

        widget.show()

    def show_trim_widget(self) -> None:
        if len(self._trim_wide_widgets) > 0:
            uuid = list(self._trim_wide_widgets.keys())[0]
            widget = self._trim_wide_widgets[uuid]
            widget.show()
            widget.raise_()
            return

        with load_cursor():
            model = TrimModel()
            widget = TrimWidgetView(model=model, parent=None)

            self._data.onCycleCorrectionCalculated.connect(model.onNewPrediction)

            uuid = str(uuid4())
            self._trim_wide_widgets[uuid] = widget

            def closeEvent(self, event: QtGui.QCloseEvent) -> None:
                event.ignore()
                self.hide()

            widget.closeEvent = types.MethodType(closeEvent, widget)

        widget.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._acquisition.stop()
        super().closeEvent(event)
