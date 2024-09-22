from __future__ import annotations

import logging

import hystcomp_utils.cycle_data
import numpy as np
import pyda
import pyda.data
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import (
    BufferedSubscription,
    BufferedSubscriptionEventBuilder,
    Subscription,
)

log = logging.getLogger(__name__)


PARAM_FULLECO_IREF = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.MBI.FULLECO/IREF"
TRIGGER = "XTIM.SX.FCY-MMODE-CT/Acquisition"


class TrackFullEcoEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions=[
                Subscription("MMODE", TRIGGER),
            ],
            buffered_subscriptions=[
                BufferedSubscription("FULLECO_IREF", PARAM_FULLECO_IREF),
            ],
            provider=provider,
            parent=parent,
        )

        self._cycle_data_buffer: dict[str, hystcomp_utils.cycle_data.CycleData] = {}

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        parameter = str(fspv.query.endpoint)
        selector = str(fspv.query.context)

        if parameter == TRIGGER:
            if selector not in self._cycle_data_buffer:
                log.error(f"Received trigger for cycle without data: {selector}.")
                return

            cycle_data = self._cycle_data_buffer[selector]

            mmode = fspv.value.get("MACHInE_MODE")

            if mmode != "FULLECO":
                log.info(f"{cycle_data} is not in FULLECO mode: {mmode}")
                return None

            log.info(f"{cycle_data} is in FULLECO mode")

            if cycle_data.cycle.endswith("FULLECO"):
                log.info(f"{cycle_data} is already in FULLECO mode")
                return None
            if cycle_data.cycle.endswith("DYNECO"):
                log.info(f"{cycle_data} is in DYNECO mode")
                return None

            cycle_data.cycle = f"{cycle_data.cycle}_FULLECO"

            msg = f"[{cycle_data}]: Adding FULLECO programmed current."
            log.debug(msg)

            i_prog_df = self._get_buffered_data(PARAM_FULLECO_IREF, selector).value.get(
                "value"
            )
            i_prog = np.vstack((i_prog_df.xs, i_prog_df.ys))

            if i_prog[1, 0] != i_prog[1, -1]:
                msg = (
                    "The programmed current does not start and end at the same value: "
                )
                msg += f"start={i_prog[1, 0]}, end={i_prog[1, -1]}"
                log.warning(msg)

            cycle_data.current_prog = i_prog

            msg = f"[{cycle_data}]: Added FULLECO programmed current."
            log.debug(msg)

            if cycle_data.cycle.endswith("ECO"):
                msg = f"[{cycle_data}]: ECO cycle has already had programs updated."
                log.debug(msg)
                self.cycleDataAvailable.emit(cycle_data)
                return

            self.onNewCycleData(cycle_data)

        # unknown parameter
        msg = f"Received unknown acquisition for {parameter}@{selector}."
        raise ValueError(msg)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"[{cycle_data}]: Received data, saving to buffer."
        log.debug(msg)

        self._cycle_data_buffer[cycle_data.user] = cycle_data
