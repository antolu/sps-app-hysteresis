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


PARAM_I_PROG = "rmi://virtual_sps/MBI/IREF"
PARAM_B_PROG = "rmi://virtual_sps/SPSBEAM/B"
TRIGGER = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/XTIM.UCAP.SCY-500/Acquisition"


class AddProgrammedEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions=[
                Subscription("SCY", TRIGGER, ignore_first_updates=True),
            ],
            buffered_subscriptions=[
                BufferedSubscription("I_PROG", PARAM_I_PROG),
                BufferedSubscription("B_PROG", PARAM_B_PROG),
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
        selector = str(fspv.query.context)

        if parameter == TRIGGER:
            if selector not in self._cycle_data_buffer:
                log.error(f"Received trigger for cycle without data: {selector}.")
                return

            cycle_data = self._cycle_data_buffer[selector]

            if cycle_data.cycle.endswith("ECO"):
                msg = f"[{cycle_data}]: ECO cycle has already had programs updated."
                log.debug(msg)
                self.cycleDataAvailable.emit(cycle_data)
                return

            prog_i_df = self._get_buffered_data(PARAM_I_PROG, selector).value.get(
                "value"
            )
            prog_b_df = self._get_buffered_data(PARAM_B_PROG, selector).value.get(
                "value"
            )

            cycle_data.current_prog = np.vstack((prog_i_df.xs, prog_i_df.ys))
            cycle_data.field_prog = np.vstack((prog_b_df.xs, prog_b_df.ys))

            msg = f"[{cycle_data}]: Added programmed data."
            log.debug(msg)

            self.onNewCycleData(cycle_data)

        # unknown parameter
        msg = f"Received unknown acquisition for {parameter}@{selector}."
        raise ValueError(msg)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"[{cycle_data}]: Received data, saving to buffer."
        log.debug(msg)

        self._cycle_data_buffer[cycle_data.user] = cycle_data
