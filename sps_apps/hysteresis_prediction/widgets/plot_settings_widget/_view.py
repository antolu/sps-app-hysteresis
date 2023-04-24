from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from qtpy.QtCore import QTimer, Signal
from qtpy.QtWidgets import QWidget

from ...generated.plot_settings_widget_ui import Ui_PlotSettingsWidget
from ...utils import run_in_main_thread

log = logging.getLogger(__name__)


FMT = "%Y-%m-%d %H:%M:%S.%f"


class PlotSettingsWidget(Ui_PlotSettingsWidget, QWidget):
    timespan_changed = Signal(int, int)  # min, max
    downsample_changed = Signal(int)
    new_cycle = Signal(str, str, float)  # PLS, LSA, timestamp
    toggle_predictions = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)

        self._prediction_enabled = False

        self.spinBoxTimespan.valueChanged.connect(self._timespan_changed)
        self.spinBoxDownsample.valueChanged.connect(self.downsample_changed)
        self.buttonResetAxis.clicked.connect(self._timespan_changed)
        self.buttonPredict.clicked.connect(self._on_prediction_toggle)

        self.new_cycle.connect(self._on_new_cycle)

    def _timespan_changed(self, *_: Any) -> None:
        self.timespan_changed.emit(self.spinBoxTimespan.value(), 0)

    @run_in_main_thread
    def _on_new_cycle(self, pls: str, lsa: str, timestamp: float) -> None:
        self.labelUser.setText(pls.split(".")[-1])
        self.labelCycle.setText(lsa)
        self.labelCycleTime.setText(
            datetime.fromtimestamp(timestamp / 1e9).strftime(FMT)
        )

        self.blink_led()

    @run_in_main_thread
    def blink_led(self) -> None:
        self.Led.setStatus(self.Led.Status.ON)
        QTimer.singleShot(500, lambda: self.Led.setStatus(self.Led.Status.OFF))

    @run_in_main_thread
    def _on_prediction_toggle(self) -> None:
        if self._prediction_enabled:
            self._prediction_enabled = False
            self.buttonPredict.setText("Start Predictions")
        else:
            self._prediction_enabled = True
            self.buttonPredict.setText("Stop Predictions")

        self.toggle_predictions.emit(self._prediction_enabled)

    @run_in_main_thread
    def _set_progressbar(self, value: int, total: int) -> None:
        print(f"setting new value to {value}")
        new_value = value / total * 100 if value < total else 100
        self.progressBar.setValue(new_value)
