from __future__ import annotations

import typing

import pyda_japc

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

if typing.TYPE_CHECKING:
    from qtpy import QtCore


class DataFlow:
    def start(self) -> None:
        raise NotImplementedError

    @QtCore.Slot
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
        self._buffer.newBufferAvailable.connect(self._predict.onNewCycleData)
        self._buffer.newEcoBufferAvailable.connect(self._predict.onNewCycleData)
        self._predict.cycleDataAvailable.connect(self._correction.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._add_programmed.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._start_cycle.onNewCycleData)

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
