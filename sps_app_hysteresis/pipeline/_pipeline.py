from __future__ import annotations

import logging

from qtpy import QtCore

log = logging.getLogger(__name__)


class Pipeline(QtCore.QObject):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @QtCore.Slot(str)
    def onResetReference(self, cycle: str) -> None:
        raise NotImplementedError

    @property
    def onModelLoaded(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleStart(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def resetState(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onTrimApplied(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onNewReference(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onMetricsAvailable(self) -> QtCore.Signal:
        raise NotImplementedError

    @QtCore.Slot(str, float)
    def setGain(self, cycle: str, gain: float) -> None:
        raise NotImplementedError
