from __future__ import annotations


from .data import (
    CreateCycleEventBuilder,
    AddMeasurementsEventBuilder,
    BufferEventbuilder,
)
from .inference import Inference


class DataFlow:
    def __init__(self, buffer_size: int = 60000) -> None:
        self._create_cycle = CreateCycleEventBuilder()
        self._add_measurements = AddMeasurementsEventBuilder()
        self._buffer = BufferEventbuilder(buffer_size)
        self._predict = Inference()

        self._connect_signals()

    def _connect_signals(self) -> None:
        self._create_cycle.cycleDataAvailable.connect(
            self._add_measurements.onNewCycleData
        )
        self._add_measurements.cycleDataAvailable.connect(
            self._buffer.onNewCycleData
        )
        self._buffer.newBufferAvailable.connect(self._predict.onNewCycleData)
        self._buffer.newEcoBufferAvailable.connect(
            self._predict.onNewCycleData
        )
