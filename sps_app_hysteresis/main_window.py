from __future__ import annotations

import logging

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.rbac import RbaButton
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from op_app_context import context
from qtpy import QtGui, QtWidgets

from .contexts import app_context
from .generated.main_window_ui import Ui_main_window
from .history import PredictionHistory
from .io import IO
from .pipeline import StandalonePipeline
from .standalone._inference import PredictionMode
from .widgets import ModelLoadDialog, PlotModel
from .widgets.history_widget import HistoryWidget
from .widgets.trim_widget import TrimModel, TrimWidgetView

log = logging.getLogger(__name__)

__all__ = ["MainWindow"]


class MainWindow(Ui_main_window, ApplicationFrame):
    def __init__(
        self,
        pipeline: StandalonePipeline,
        parent: QtWidgets.QWidget | None = None,
    ):
        ApplicationFrame.__init__(self, parent)
        Ui_main_window.__init__(self)

        self.setupUi(self)

        log_console = LogConsole(self)
        self.log_console = log_console
        log_console.toggleExpandedMode()
        log_console.freeze()

        self.rba_widget = RbaButton(self)
        if context.rbac_token is not None:
            self.rba_widget.model.update_token(context.rbac_token)

        timing_model = TimingBarModel(domain=TimingBarDomain.SPS)
        timing_bar = TimingBar(self, model=timing_model)
        self.timing_bar = timing_bar

        self._data = pipeline

        plot_model = PlotModel(self._data, parent=self)
        self.widgetPlot.model = plot_model

        self._io = IO()
        self._history = PredictionHistory(self)
        self._history_widget = HistoryWidget(
            self._history, parent=None, measured_available=app_context().B_MEAS_AVAIL
        )

        self._trim_model = TrimModel(trim_settings=app_context().TRIM_SETTINGS)
        self._trim_widget = TrimWidgetView(model=self._trim_model, parent=None)

        self._connect_signals()
        self._connect_actions()

        self._trim_widget.show()
        self._history_widget.show()

        self.actionAutoregressive.setChecked(False)
        if app_context().ONLINE:
            self.action_Load_Model.setEnabled(False)
            self.actionProgrammed_current.setEnabled(False)
            self.actionAutoregressive.setEnabled(False)

        log_console.clear()
        log_console.unfreeze()

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
        self._data.onCycleStart.connect(self._history.update_cycle)
        self._data.onCycleMeasured.connect(self._history.update_cycle)
        self._data.onNewReference.connect(self._history.onReferenceChanged)

        self._trim_widget.referenceReset.connect(self._data.onResetReference)

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
            # Set up prediction mode radio buttons
            self._setup_prediction_mode_actions()
            self.actionReset_state.triggered.connect(self._data.resetState.emit)

        self.actionPrediction_Analysis.triggered.connect(self._history_widget.show)
        self.action_Trim_View.triggered.connect(self._trim_widget.show)

    def _setup_prediction_mode_actions(self) -> None:
        """Set up the prediction mode radio button group."""
        # Create action group to make radio buttons mutually exclusive
        self._prediction_mode_group = QtWidgets.QActionGroup(self)
        self._prediction_mode_group.addAction(self.actionMode_Combined)
        self._prediction_mode_group.addAction(self.actionMode_Hysteresis_Only)
        self._prediction_mode_group.addAction(self.actionMode_Eddy_Current_Only)

        # Connect actions to handlers
        self.actionMode_Combined.triggered.connect(
            lambda: self._on_prediction_mode_changed(PredictionMode.COMBINED)
        )
        self.actionMode_Hysteresis_Only.triggered.connect(
            lambda: self._on_prediction_mode_changed(PredictionMode.HYSTERESIS_ONLY)
        )
        self.actionMode_Eddy_Current_Only.triggered.connect(
            lambda: self._on_prediction_mode_changed(PredictionMode.EDDY_CURRENT_ONLY)
        )

    def _on_prediction_mode_changed(self, mode: PredictionMode) -> None:
        """Handle prediction mode changes."""
        log.info(f"Prediction mode changed to: {mode.value}")

        # Set the prediction mode
        self._data._predict.set_prediction_mode(mode)  # noqa: SLF001

        # Reset reference when mode changes
        self._data.onResetReference("all")
        log.info("Reference reset due to prediction mode change")

    def on_load_model_triggered(self) -> None:
        dialog = ModelLoadDialog(parent=self)
        dialog.loadLocalCheckpoint.connect(self._data._predict.loadLocalModel)  # noqa: SLF001
        dialog.loadMlpCheckpoint.connect(self._data._predict.loadMlpModel)  # noqa: SLF001
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
        self._trim_widget.close()
        self._history_widget.close()
        super().closeEvent(event)
