from __future__ import annotations

import logging

from qtpy import QtCore

log = logging.getLogger(__name__)


class Pipeline:
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
    def setGain(self, cycle: str, gain: float) -> QtCore.Signal:
        raise NotImplementedError


class PipelineWorker(QtCore.QObject):
    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

        self._pipeline: Pipeline | None = None
        self._mutex = QtCore.QMutex()
        self.cv = QtCore.QWaitCondition()

    def init_pipeline(self) -> None:
        if self._pipeline is not None:
            msg = "Pipeline already initialized."
            raise RuntimeError(msg)

        with QtCore.QMutexLocker(self._mutex):
            self._init_pipeline_impl()

            self.cv.wakeAll()

    def _init_pipeline_impl(self) -> None:
        raise NotImplementedError

    def wait(self) -> None:
        self.cv.wait(self._mutex)

    def start(self) -> None:
        if self._pipeline is None:
            self.init_pipeline()

        assert self._pipeline is not None
        self._pipeline.start()

    def stop(self) -> None:
        if self._pipeline is None:
            return

        self._pipeline.stop()

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self.init_pipeline()

        assert self._pipeline is not None
        return self._pipeline
