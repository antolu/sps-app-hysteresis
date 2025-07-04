"""
The TrimModel is responsible for the logic of the TrimWidget.
"""

from __future__ import annotations

import logging

from qtpy import QtCore

from ...local.trim import TrimSettings

log = logging.getLogger(__package__)


class TrimModel(QtCore.QObject):
    """Model for the TrimWidget."""

    contextChanged = QtCore.Signal(str)

    def __init__(
        self, trim_settings: TrimSettings, parent: QtCore.QObject | None = None
    ):
        super().__init__(parent)

        self._trim_settings = trim_settings
        self._cycle: str | None = None

    @QtCore.Slot(str)
    def setCycle(self, cycle: str) -> None:
        self._cycle = cycle

        self.contextChanged.emit(cycle)

    @property
    def cycle(self) -> str:
        if self._cycle is None:
            msg = "No cycle selected."
            log.error(msg)
            raise ValueError(msg)

        return self._cycle

    @property
    def settings(self) -> TrimSettings:
        return self._trim_settings
