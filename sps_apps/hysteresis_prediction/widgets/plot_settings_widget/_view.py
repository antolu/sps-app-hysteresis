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

    def __init__(self, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)

        self.spinBoxTimespan.valueChanged.connect(self._timespan_changed)
        self.spinBoxDownsample.valueChanged.connect(self.downsample_changed)
        self.buttonResetAxis.clicked.connect(self._timespan_changed)

    def _timespan_changed(self, *_: Any) -> None:
        self.timespan_changed.emit(self.spinBoxTimespan.value(), 0)

    def _on_new_cycle(self, pls: str, lsa: str, timestamp: float) -> None:
        self.labelUser.setText(pls.split(".")[-1])
        self.labelCycle.setText(lsa)
        self.labelTimestamp.setText(
            datetime.fromtimestamp(timestamp).strftime(FMT)
        )

        self.blink_led()

    @run_in_main_thread
    def blink_led(self) -> None:
        self.Led.setStatus(self.Led.Status.ON)
        QTimer.singleShot(500, lambda: self.Led.setStatus(self.Led.Status.OFF))
