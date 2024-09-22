from __future__ import annotations

import pyda_japc

from .data import (
    AddMeasurementReferencesEventBuilder,
    AddMeasurementsEventBuilder,
    AddProgrammedEventBuilder,
    BufferEventbuilder,
    CreateCycleEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    TrackDynEcoEventBuilder,
    TrackFullEcoEventBuilder,
)
from .inference import CalculateCorrection, Inference


class DataFlow:
    def __init__(
        self, provider: pyda_japc.JapcProvider, buffer_size: int = 60000
    ) -> None:
        self._create_cycle = CreateCycleEventBuilder(provider=provider)
        self._add_measurements_pre = AddMeasurementsEventBuilder(provider=provider)
        self._buffer = BufferEventbuilder(buffer_size)
        self._predict = Inference()
        self._correction = CalculateCorrection()
        self._add_programmed = AddProgrammedEventBuilder(provider=provider)
        self._add_measurement_post = CycleStampedAddMeasurementsEventBuilder(
            provider=provider
        )
        self._add_measurement_ref = AddMeasurementReferencesEventBuilder()

        self._track_dyneco = TrackDynEcoEventBuilder(provider=provider)
        self._track_fulleco = TrackFullEcoEventBuilder(provider=provider)

        self._connect_signals()

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
