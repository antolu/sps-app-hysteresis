from __future__ import annotations

import logging

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.rbac import RbaButton
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from op_app_context import context
from qtpy import QtGui, QtWidgets

from .contexts import app_context
from .flow import LocalDataFlow
from .generated.main_window_ui import Ui_main_window
from .history import PredictionHistory
from .io import IO
from .widgets import ModelLoadDialog, PlotModel
from .widgets.history_widget import HistoryWidget
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

        log_console = LogConsole(self)
        self.log_console = log_console
        log_console.toggleExpandedMode()

        self.rba_widget = RbaButton(self)

        timing_model = TimingBarModel(
            domain=TimingBarDomain.SPS, japc=context.japc_client
        )
        timing_bar = TimingBar(self, model=timing_model)
        self.timing_bar = timing_bar

        self._data = data_flow

        plot_model = PlotModel(self._data, parent=self)
        self.widgetPlot.model = plot_model

        self._io = IO()
        self._history = PredictionHistory(self)
        self._history_widget = HistoryWidget(self._history, parent=None)

        self._trim_model = TrimModel(trim_settings=app_context().TRIM_SETTINGS)
        self._trim_widget = TrimWidgetView(model=self._trim_model, parent=None)

        self._connect_signals()
        self._connect_actions()

        self._trim_widget.show()
        self._history_widget.show()

        if not app_context().ONLINE:
            self.action_Load_Model.setEnabled(False)
            self.actionProgrammed_current.setEnabled(False)
            self.actionAutoregressive.setEnabled(False)

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

        # connect history
        self._data.onCycleCorrectionCalculated.connect(self._history.add_cycle)
        self._data.onCycleStart.connect(self._history.add_cycle)
        self._data.onCycleMeasured.connect(self._history.update_cycle)

        # trim
        self._data.onTrimApplied.connect(self._trim_widget.onTrimApplied)

        self._data.onModelLoaded.connect(
            lambda: QtWidgets.QMessageBox.information(
                self,
                "Model loaded",
                "Model successfully loaded.\nPredictions will now start.",
            )
        )
        self._data.onModelLoaded.connect(
            lambda: self._data._predict.set_do_inference(True)  # noqa: SLF001
        )

        assert self.rba_widget is not None
        self.rba_widget.model.login_succeeded.connect(context.set_rbac_token)

    def _connect_actions(self) -> None:
        self.actionShow_Plot_Settings.triggered.connect(self.toggle_plot_settings)
        self.actionContinuous_Data_Export.toggled.connect(self._io.set_enabled)
        self.action_Clear_Reference.triggered.connect(self._data.resetReference)

        if not app_context().ONLINE:
            self.action_Load_Model.triggered.connect(self.on_load_model_triggered)
            self.actionProgrammed_current.triggered.connect(
                lambda x: self._data._predict.set_use_programmed_current(  # noqa: SLF001
                    state=self.actionProgrammed_current.isChecked()
                )
            )
            self.actionAutoregressive.triggered.connect(
                lambda x: self._data._predict.set_autoregressive(  # noqa: SLF001
                    self.actionAutoregressive.isChecked()
                )
            )
            self.actionReset_state.triggered.connect(self._data.resetState.emit)

        self.actionPrediction_Analysis.triggered.connect(self._history_widget.show)
        self.action_Trim_View.triggered.connect(self._trim_widget.show)

    def on_load_model_triggered(self) -> None:
        dialog = ModelLoadDialog(parent=self)
        dialog.load_checkpoint.connect(self._data._predict.loadModel)  # noqa: SLF001
        result = dialog.exec()

        if result == QtWidgets.QDialog.Rejected:
            log.debug("Model load dialog cancelled.")
            return

    def toggle_plot_settings(self) -> None:
        if self.actionShow_Plot_Settings.isChecked():
            self.widgetSettings.show()
        else:
            self.widgetSettings.hide()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self._data.stop()
        super().closeEvent(event)
