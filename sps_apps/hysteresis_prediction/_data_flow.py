from __future__ import annotations

import pyda_japc
from qtpy import QtCore

from .data import (
    AddMeasurementReferencesEventBuilder,
    AddMeasurementsEventBuilder,
    AddProgrammedEventBuilder,
    BufferEventbuilder,
    CreateCycleEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    StartCycleEventBuilder,
    TrackDynEcoEventBuilder,
    TrackFullEcoEventBuilder,
)
from .inference import CalculateCorrection, Inference


class DataFlow:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def resetReference(self) -> None:
        raise NotImplementedError

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleStart(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        raise NotImplementedError

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        raise NotImplementedError


class LocalFlowWorker(QtCore.QObject):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider,
        buffer_size: int = 60000,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._data_flow: LocalDataFlow | None = None

        self._provider = provider
        self._buffer_size = buffer_size

        self._mutex = QtCore.QMutex()
        self.cv = QtCore.QWaitCondition()

    def init_data_flow(self) -> None:
        if self._data_flow is not None:
            msg = "Data flow already initialized."
            raise ValueError(msg)

        with QtCore.QMutexLocker(self._mutex):
            self._data_flow = LocalDataFlow(
                provider=self._provider,
                buffer_size=self._buffer_size,
                parent=self.parent(),
            )

            self.cv.wakeAll()

    def wait(self) -> None:
        self.cv.wait(self._mutex)

    def start(self) -> None:
        if self._data_flow is None:
            self.init_data_flow()

        assert self._data_flow is not None
        self._data_flow.start()

    def stop(self) -> None:
        if self._data_flow is not None:
            self._data_flow.stop()

    @property
    def data_flow(self) -> LocalDataFlow:
        if self._data_flow is None:
            msg = "Data flow not initialized. Call init_data_flow() first."
            raise ValueError(msg)
        return self._data_flow


class LocalDataFlow(DataFlow):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider,
        buffer_size: int = 60000,
        parent: QtCore.QObject | None = None,
    ) -> None:
        self._create_cycle = CreateCycleEventBuilder(provider=provider, parent=parent)
        self._add_measurements_pre = AddMeasurementsEventBuilder(
            provider=provider, parent=parent
        )
        self._buffer = BufferEventbuilder(buffer_size=buffer_size, parent=parent)
        self._predict = Inference(parent=parent)
        self._correction = CalculateCorrection(parent=parent)
        self._start_cycle = StartCycleEventBuilder(provider=provider, parent=parent)
        self._add_programmed = AddProgrammedEventBuilder(
            provider=provider, parent=parent
        )
        self._add_measurement_post = CycleStampedAddMeasurementsEventBuilder(
            provider=provider, parent=parent
        )
        self._add_measurement_ref = AddMeasurementReferencesEventBuilder(parent=parent)

        self._track_dyneco = TrackDynEcoEventBuilder(provider=provider, parent=parent)
        self._track_fulleco = TrackFullEcoEventBuilder(provider=provider, parent=parent)

        self._connect_signals()

    def start(self) -> None:
        self._create_cycle.start()
        self._add_measurements_pre.start()
        self._buffer.start()
        self._predict.start()
        self._correction.start()
        self._start_cycle.start()
        self._add_programmed.start()
        self._add_measurement_post.start()
        self._add_measurement_ref.start()
        self._track_dyneco.start()
        self._track_fulleco.start()

    def stop(self) -> None:
        self._create_cycle.stop()
        self._add_measurements_pre.stop()
        self._buffer.stop()
        self._predict.stop()
        self._correction.stop()
        self._start_cycle.stop()
        self._add_programmed.stop()
        self._add_measurement_post.stop()
        self._add_measurement_ref.stop()
        self._track_dyneco.stop()
        self._track_fulleco.stop()

    def resetReference(self) -> None:
        self._add_measurement_ref.resetReference()
        self._correction.resetReference()

    def _connect_signals(self) -> None:
        self._create_cycle.cycleDataAvailable.connect(
            self._add_measurements_pre.onNewCycleData
        )
        self._add_measurements_pre.cycleDataAvailable.connect(
            self._buffer.onNewCycleData
        )
        self._add_measurements_pre.cycleDataAvailable.connect(
            self._start_cycle.onNewCycleData
        )
        self._buffer.newBufferAvailable.connect(self._predict.onNewCycleDataBuffer)
        self._buffer.newEcoBufferAvailable.connect(self._predict.onNewCycleDataBuffer)
        self._predict.cycleDataAvailable.connect(self._correction.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._add_programmed.onNewCycleData)

        self._add_programmed.cycleDataAvailable.connect(self._buffer.onNewProgCycleData)
        self._add_programmed.cycleDataAvailable.connect(
            self._add_measurement_post.onNewCycleData
        )
        self._add_measurement_post.cycleDataAvailable.connect(
            self._add_measurement_ref.onNewCycleData
        )

        self._add_measurement_post.cycleDataAvailable.connect(
            self._buffer.onNewMeasCycleData
        )

        self._correction.cycleDataAvailable.connect(self._track_dyneco.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._track_fulleco.onNewCycleData)
        self._track_dyneco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)
        self._track_fulleco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        return self._create_cycle.cycleDataAvailable

    @property
    def onCycleStart(self) -> QtCore.Signal:
        return self._start_cycle.cycleDataAvailable

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        return self._correction.cycleDataAvailable

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        return self._add_measurement_ref.cycleDataAvailable


class UcapDataFlow(DataFlow): ...
