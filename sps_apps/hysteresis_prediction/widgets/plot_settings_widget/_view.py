from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional
import math

from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import QWidget

from ...generated.plot_settings_widget_ui import Ui_PlotSettingsWidget
from ...utils import run_in_main_thread
from ._status import LOG_MESSAGES, AppStatus

log = logging.getLogger(__name__)


FMT = "%Y-%m-%d %H:%M:%S.%f"


class PlotSettingsWidget(Ui_PlotSettingsWidget, QWidget):
    timespan_changed = Signal(int, int)  # min, max
    downsample_changed = Signal(int)
    toggle_predictions = Signal(bool)
    status_changed = Signal(AppStatus)

    def __init__(self, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)

        self._prediction_enabled = False

        self.spinBoxTimespan.valueChanged.connect(self._timespan_changed)
        self.spinBoxDownsample.valueChanged.connect(self.downsample_changed)
        self.buttonResetAxis.clicked.connect(self._timespan_changed)
        self.buttonPredict.clicked.connect(self._on_prediction_toggle)

        self.status_changed.connect(self._on_new_status)

        self.buttonPredict.setEnabled(False)

    @run_in_main_thread
    def on_model_loaded(self) -> None:
        """Called when the model is loaded. Enable prediction button."""
        self.buttonPredict.setEnabled(True)

    def _timespan_changed(self, *_: Any) -> None:
        self.timespan_changed.emit(self.spinBoxTimespan.value(), 0)

    @run_in_main_thread
    def _on_new_status(self, status: AppStatus) -> None:
        message = LOG_MESSAGES[status]
        self.labelStatus.setText(message)

    @Slot(str, str, float)
    @run_in_main_thread
    def onNewCycle(self, pls: str, lsa: str, timestamp: float) -> None:
        self.labelUser.setText(pls.split(".")[-1])
        self.labelCycle.setText(lsa)
        self.labelCycleTime.setText(
            datetime.fromtimestamp(timestamp / 1e9).strftime(FMT)[:-3]
        )

        self.blink_led()

    @run_in_main_thread
    def blink_led(self) -> None:
        self.Led.setStatus(self.Led.Status.ON)
        QTimer.singleShot(500, lambda: self.Led.setStatus(self.Led.Status.OFF))

    @run_in_main_thread
    def _on_prediction_toggle(self, *_: Any) -> None:
        if self._prediction_enabled:
            self._prediction_enabled = False
            self.buttonPredict.setText("Start Predictions")
        else:
            self._prediction_enabled = True
            self.buttonPredict.setText("Stop Predictions")

        self.toggle_predictions.emit(self._prediction_enabled)

    @run_in_main_thread
    def _set_progressbar(self, value: int, total: int) -> None:
        try:
            new_value = value / total * 100 if value < total else 100
            log.debug(f"Setting progress bar value to {new_value}.")
            self.progressBar.setValue(math.ceil(new_value))

            if new_value >= 100:
                self.progressBar.hide()
            else:
                self.progressBar.show()
        except:  # noqa E722
            log.exception(
                "An exception occurred while setting progress bar value."
            )
