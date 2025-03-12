from __future__ import annotations

import logging

from qtpy import QtCore

log = logging.getLogger(__name__)


class DataFlow:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @QtCore.Slot(str)
    def resetReference(self, cycle: str) -> None:
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

    @QtCore.Slot(str, float)
    def setGain(self, cycle: str, gain: float) -> QtCore.Signal:
        raise NotImplementedError


class FlowWorker(QtCore.QObject):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

        self._data_flow: DataFlow | None = None
        self._mutex = QtCore.QMutex()
        self.cv = QtCore.QWaitCondition()

    def init_data_flow(self) -> None:
        if self._data_flow is not None:
            msg = "Data flow already initialized."
            raise RuntimeError(msg)

        with QtCore.QMutexLocker(self._mutex):
            self._init_data_flow_impl()

            self.cv.wakeAll()

    def _init_data_flow_impl(self) -> None:
        raise NotImplementedError

    def wait(self) -> None:
        self.cv.wait(self._mutex)

    def start(self) -> None:
        if self._data_flow is None:
            self.init_data_flow()

        assert self._data_flow is not None
        self._data_flow.start()

    def stop(self) -> None:
        if self._data_flow is None:
            return

        self._data_flow.stop()

    @property
    def data_flow(self) -> DataFlow:
        if self._data_flow is None:
            self.init_data_flow()

        assert self._data_flow is not None
        return self._data_flow
