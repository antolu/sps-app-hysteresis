from __future__ import annotations

import datetime
import logging

import hystcomp_utils.cycle_data
import numpy as np
import pyda
import pyda.data
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import (
    EventBuilderAbc,
    Subscription,
)

log = logging.getLogger(__name__)


PARAM_DYNECO_IREF = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.MBI.DYNECO/IREF"


class TrackDynEcoEventBuilder(EventBuilderAbc):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions=[
                Subscription("DYNECO", PARAM_DYNECO_IREF, ignore_first_updates=True),
            ],
            provider=provider,
            parent=parent,
            no_metadata_source=True,
        )

        self._cycle_data_buffer: dict[str, hystcomp_utils.cycle_data.CycleData] = {}

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        parameter = str(fspv.query.endpoint)
        context = str(fspv.query.context)
        header = fspv.value.header
        cycle_time_s = str(header.cycle_time())[:-7]
        selector = str(header.selector)
        id_ = f"[{selector}@{cycle_time_s}]"

        if parameter != PARAM_DYNECO_IREF:
            msg = f"Received unknown acquisition for {parameter}@{context}."
            raise ValueError(msg)

        if selector not in self._cycle_data_buffer:
            log.error(f"Received trigger for cycle without data: {selector}.")
            return

        cycle_data = self._cycle_data_buffer[selector]

        i_prog_df = fspv.value.get("value")
        iref = np.vstack((i_prog_df.xs, i_prog_df.ys))

        if not np.allclose(cycle_data.cycle_timestamp, header.cycle_timestamp):
            msg = (
                f"{cycle_data} Cycle timestamp mismatch: {cycle_data.cycle_timestamp} "
                f"!= {header.cycle_timestamp}. This probably means that the cycle data "
                f"comes from a different cycle instance than the IREF data."
            )
            log.error(msg)
            raise ValueError(msg)

        now = datetime.datetime.now()
        msg = (
            f"{id_} Triggered by new DYNECO signal from IREF at {now}."
            f"Changing cycle data to DYNECO cycle and updating programmed current."
        )
        log.debug(msg)

        if cycle_data.cycle.endswith("ECO"):
            msg = f"{id_} Cycle data is already ECO. Skipping."
            log.warning(msg)
            return None

        orig_cycle = cycle_data.cycle
        cycle_data.cycle += "_DYNECO"
        cycle_data.current_prog = iref

        msg = f"{id_} Updated cycle data {orig_cycle} -> {cycle_data.cycle}."
        log.debug(msg)

        self.cycleDataAvailable.emit(cycle_data)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"[{cycle_data}]: Received data, saving to buffer."
        log.debug(msg)

        self._cycle_data_buffer[cycle_data.user] = cycle_data
