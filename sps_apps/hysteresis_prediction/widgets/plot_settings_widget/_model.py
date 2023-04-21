from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtCore import QObject, Signal

log = logging.getLogger(__name__)


class PlotSettingsModel(QObject):
    signal = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent=parent)
