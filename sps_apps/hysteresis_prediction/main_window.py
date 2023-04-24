import logging
from typing import Optional

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from PyQt5.QtWidgets import QWidget
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import QDialog, QMessageBox

from .data import Acquisition
from .generated.main_window_ui import Ui_main_window
from .inference import Inference
from .settings import context
from .widgets import ModelLoadDialog, PlotModel

log = logging.getLogger(__name__)

__all__ = ["MainWindow"]


class MainWindow(Ui_main_window, ApplicationFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        ApplicationFrame.__init__(self, parent)
        Ui_main_window.__init__(self)

        self.setupUi(self)

        log_console = LogConsole(self)
        self.log_console = log_console
        log_console.toggleExpandedMode()

        timing_model = TimingBarModel(
            domain=TimingBarDomain.SPS, japc=context.japc
        )
        timing_bar = TimingBar(self, model=timing_model)
        self.timing_bar = timing_bar

        self._acquisition = Acquisition(min_buffer_size=300000)
        self._inference = Inference(parent=self)

        plot_model = PlotModel(self._acquisition, parent=self)
        self.widgetPlot.model = plot_model

        self.actionShow_Plot_Settings.triggered.connect(
            self.toggle_plot_settings
        )

        self.widgetSettings.timespan_changed.connect(
            self.widgetPlot.set_time_span
        )

        self.widgetSettings.downsample_changed.connect(
            plot_model.set_downsample
        )
        self.widgetSettings.toggle_predictions.connect(
            self._inference.set_do_inference
        )
        self._inference.cycle_predicted.connect(plot_model.new_predicted_cycle)
        self._acquisition.new_buffer_data.connect(
            self._inference.predict_last_cycle
        )
        self._acquisition.cycle_started.connect(
            self.widgetSettings._on_new_cycle
        )
        self._acquisition.buffer.buffer_size_changed.connect(
            lambda x: self.widgetSettings._set_progressbar(x, 300000)
        )

        self.action_Load_Model.triggered.connect(self.on_load_model_triggered)
        self._inference.model_loaded.connect(
            lambda: QMessageBox.information(
                self, "Model loaded", "Model successfully loaded."
            )
        )

        self._acquisition.run()

    def on_load_model_triggered(self) -> None:
        dialog = ModelLoadDialog(parent=self)
        result = dialog.exec()

        if result == QDialog.Rejected:
            log.debug("Model load dialog cancelled.")
            return
        elif result == QDialog.Accepted:
            ckpt_path = dialog.ckpt_path
            device = dialog.device

            self._inference.load_model.emit(ckpt_path, device)

    def toggle_plot_settings(self) -> None:
        if self.actionShow_Plot_Settings.isChecked():
            self.widgetSettings.show()
        else:
            self.widgetSettings.hide()

    def closeEvent(self, event: QCloseEvent) -> None:
        self._acquisition.stop()
        super().closeEvent(event)
