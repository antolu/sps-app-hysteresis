from __future__ import annotations

import logging

from qtpy.QtCore import QObject, Signal

log = logging.getLogger(__package__)


class PlotSettingsModel(QObject):
    signal = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent=parent)
