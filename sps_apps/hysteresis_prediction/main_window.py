from __future__ import annotations

import logging
import types
from uuid import uuid4

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from accwidgets.rbac import RbaButton
from op_app_context import context
from qtpy import QtGui, QtWidgets

from .flow import LocalDataFlow
from .generated.main_window_ui import Ui_main_window
from .io import IO
from .utils import load_cursor
from .widgets import ModelLoadDialog, PlotModel
from .widgets.trim_widget import TrimModel, TrimWidgetView
from .history import PredictionHistory
from .widgets.history_widget import HistoryWidget

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

        self._trim_widgets: dict[str, TrimWidgetView] = {}

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
        self._history = PredictionHistory(self)

        plot_model = PlotModel(self._data, parent=self)
        self.widgetPlot.model = plot_model

        self._io = IO()
        self._history_widget = HistoryWidget(self._history, parent=None)

        self._connect_signals()
        self._connect_actions()

        self.show_trim_widget()
        self._history_widget.show()

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

        self._data.onModelLoaded.connect(
            lambda: QtWidgets.QMessageBox.information(
                self,
                "Model loaded",
                "Model successfully loaded.\nPredictions will now start.",
            )
        )
        self._data.onModelLoaded.connect(
            lambda: self._data._predict.set_do_inference(True)
        )

        assert self.rba_widget is not None
        self.rba_widget.model.login_succeeded.connect(context.set_rbac_token)

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
        self.actionReset_state.triggered.connect(self._data.resetState.emit)

        self.actionPrediction_Analysis.triggered.connect(self._history_widget.show)
        self.action_Trim_View.triggered.connect(self.show_trim_widget)

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

    def show_trim_widget(self) -> None:
        if len(self._trim_widgets) > 0:
            uuid = list(self._trim_widgets.keys())[0]
            widget = self._trim_widgets[uuid]
            widget.show()
            widget.raise_()
            return

        with load_cursor():
            model = TrimModel()
            widget = TrimWidgetView(model=model, parent=None)

            self._data.onCycleCorrectionCalculated.connect(model.onNewPrediction)
            model.GainChanged.connect(self._data.setGain)

            uuid = str(uuid4())
            self._trim_widgets[uuid] = widget

            def closeEvent(self: TrimWidgetView, event: QtGui.QCloseEvent) -> None:
                event.ignore()
                self.hide()

                if self.toggle_button.state2Activated:
                    self.toggle_button.click()

            widget.closeEvent = types.MethodType(closeEvent, widget)

        widget.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for widget in self._analysis_widgets.values():
            widget.close()
        for widget in self._trim_widgets.values():
            widget.close()
        self._data.stop()
        super().closeEvent(event)
