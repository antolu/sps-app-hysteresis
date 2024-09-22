from __future__ import annotations


from .data import (
    CreateCycleEventBuilder,
    AddMeasurementsEventBuilder,
    BufferEventbuilder,
)


class DataFlow:
    def __init__(self, buffer_size: int = 60000) -> None:
        self._create_cycle = CreateCycleEventBuilder()
        self._add_measurements = AddMeasurementsEventBuilder()
        self._buffer = BufferEventbuilder(buffer_size)

    def _connect_signals(self) -> None:
        self._create_cycle.cycleDataAvailable.connect(
            self._add_measurements.onNewCycleData
        )
