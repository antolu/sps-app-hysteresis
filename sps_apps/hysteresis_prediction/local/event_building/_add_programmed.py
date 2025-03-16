from __future__ import annotations

import logging

import hystcomp_utils.cycle_data
import numpy as np
import pyda
import pyda.access
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
TRIGGER = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/XTIM.UCAP.SCY-CT-500/Acquisition"


class AddProgrammedEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        param_i_prog: str = PARAM_I_PROG,
        param_b_prog: str = PARAM_B_PROG,
        trigger: str = TRIGGER,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions=[
                Subscription("SCY", trigger, ignore_first_updates=True),
            ],
            buffered_subscriptions=[
                BufferedSubscription("I_PROG", param_i_prog),
                BufferedSubscription("B_PROG", param_b_prog),
            ],
            provider=provider,
            parent=parent,
            no_metadata_source=True,
        )

        self.param_i_prog = param_i_prog
        self.param_b_prog = param_b_prog

        self._cycle_data_buffer: dict[str, hystcomp_utils.cycle_data.CycleData] = {}

    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        parameter = str(fspv.query.endpoint)
        selector = str(fspv.header.selector)

        if parameter != TRIGGER:
            msg = f"Received unknown acquisition for {parameter}@{selector}."
            raise ValueError(msg)

        if selector not in self._cycle_data_buffer:
            msg = (
                f"Received trigger for cycle without data: {selector}. "
                "Perhaps cycle predictions are disabled?"
            )
            log.warning(msg)
            return

        cycle_data = self._cycle_data_buffer[selector]

        if cycle_data.economy_mode is not hystcomp_utils.cycle_data.EconomyMode.NONE:
            msg = f"[{cycle_data}]: economy cycle has already had programs updated."
            log.debug(msg)
            self.cycleDataAvailable.emit(cycle_data)
            return

        prog_i_df = self._get_buffered_data(self.param_i_prog, selector).data["value"]
        prog_b_df = self._get_buffered_data(self.param_b_prog, selector).data["value"]

        cycle_data.current_prog = np.vstack((prog_i_df.xs, prog_i_df.ys))
        cycle_data.field_prog = np.vstack((prog_b_df.xs, prog_b_df.ys))

        msg = f"[{cycle_data}]: Added programmed data."
        log.debug(msg)

        self.cycleDataAvailable.emit(cycle_data)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"[{cycle_data}]: Received data, saving to buffer."
        log.debug(msg)

        self._cycle_data_buffer[cycle_data.user] = cycle_data
