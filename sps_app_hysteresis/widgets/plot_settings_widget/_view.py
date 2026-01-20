from __future__ import annotations

import logging
from typing import Any

import hystcomp_utils.cycle_data
from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import QWidget

from ...generated.plot_settings_widget_ui import Ui_PlotSettingsWidget
from ...utils import run_in_main_thread

log = logging.getLogger(__package__)


FMT = "%Y-%m-%d %H:%M:%S.%f"


class PlotSettingsWidget(Ui_PlotSettingsWidget, QWidget):
    timespan_changed = Signal(int, int)  # min, max
    downsample_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)

        self.spinBoxTimespan.valueChanged.connect(self._timespan_changed)
        self.spinBoxDownsample.valueChanged.connect(self.downsample_changed)
        self.buttonResetAxis.clicked.connect(self._timespan_changed)

    def _timespan_changed(self, *_: Any) -> None:
        self.timespan_changed.emit(self.spinBoxTimespan.value(), 0)

    @Slot(hystcomp_utils.cycle_data.CycleData)
    @run_in_main_thread
    def onNewCycle(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        log.debug(f"[{cycle_data}] Cycle is starting. Blinking LED")
        self.labelUser.setText(cycle_data.user.split(".")[-1])
        self.labelCycle.setText(cycle_data.cycle)
        self.labelCycleTime.setText(cycle_data.cycle_time.strftime(FMT)[:-3])

        self.blink_led()

    @run_in_main_thread
    def blink_led(self) -> None:
        self.Led.setStatus(self.Led.Status.ON)
        QTimer.singleShot(500, lambda: self.Led.setStatus(self.Led.Status.OFF))
