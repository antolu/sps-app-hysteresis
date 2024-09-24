from __future__ import annotations

import logging
import typing

import hystcomp_utils.cycle_data  # noqa: F401
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import (
    BufferedSubscription,
    CycleStampGroupedTriggeredEventBuilder,
)

log = logging.getLogger(__name__)


PARAM_I_MEAS = "MBI/LOG.I.MEAS"
PARAM_B_MEAS = "SR.BMEAS-SP-B-SD/CycleSamples#samples"


class CycleStampedAddMeasurementsEventBuilder(CycleStampGroupedTriggeredEventBuilder):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider | None = None,
        track_cycle_data: bool = True,
        buffer_size: int = 10,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            buffered_subscriptions=[
                BufferedSubscription("I_MEAS", PARAM_I_MEAS, ignore_first_updates=True),
                BufferedSubscription("B_MEAS", PARAM_B_MEAS, ignore_first_updates=True),
            ],
            track_cycle_data=track_cycle_data,
            buffer_size=buffer_size,
            provider=provider,
            parent=parent,
        )

    def onCycleStampGroupTriggered(self, cycle_stamp: float, selector: str) -> None:
        cycle_data = typing.cast(
            hystcomp_utils.cycle_data.CycleData,
            self._cycle_data_buffer[selector][cycle_stamp],
        )

        i_meas_fspv = self._cycle_stamp_buffers[PARAM_I_MEAS][selector][cycle_stamp]
        b_meas_fspv = self._cycle_stamp_buffers[PARAM_B_MEAS][selector][cycle_stamp]

        i_meas = i_meas_fspv.value.get("value")
        b_meas = b_meas_fspv.value.get("value") / 1e4

        cycle_data.current_meas = i_meas
        cycle_data.field_meas = b_meas

        msg = f"[{cycle_data}] Added measurements to cycle data"
        log.debug(msg)

        self.cycleDataAvailable.emit(cycle_data)
