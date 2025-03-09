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
PARAM_BHYS_CORRECTION = (
    "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPSBEAM.BHYS-CORRECTION/Acquisition"
)


class CreateCycleEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        cycle_warning: str = "SX.CZERO-CTML/CycleWarning",
        param_i_prog: str = PARAM_I_PROG,
        param_b_prog: str = PARAM_B_PROG,
        param_b_correction: str = PARAM_BHYS_CORRECTION,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            [
                Subscription("CycleWarning", cycle_warning, ignore_first_updates=True),
            ],
            buffered_subscriptions=[
                BufferedSubscription("IREF", param_i_prog),
                BufferedSubscription("B", param_b_prog),
                BufferedSubscription("BHYS-CORRECTION", param_b_correction),
            ],
            provider=provider,
            parent=parent,
            no_metadata_source=True,
        )

        self._trigger = cycle_warning
        self.param_i_prog = param_i_prog
        self.param_b_prog = param_b_prog
        self.param_b_correction = param_b_correction

    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        selector = str(fspv.header.selector)
        msg = f"Received {fspv.query.endpoint} with {fspv.data.get('value')}"
        log.debug(msg)

        if str(fspv.query.endpoint) == self._trigger:
            if (
                not self._buffer_has_data(self.param_i_prog, selector)
                or not self._buffer_has_data(self.param_b_prog, selector)
                or not self._buffer_has_data(self.param_b_correction, selector)
            ):
                msg = f"Missing data for {selector}. Skipping cycle creation."
                log.error(msg)
                return

            current_prog_fspv = self._get_buffered_data(self.param_i_prog, selector)
            field_prog_fspv = self._get_buffered_data(self.param_b_prog, selector)
            bhys_corr_fspv = self._get_buffered_data(self.param_b_correction, selector)

            current_prog = np.vstack((
                current_prog_fspv.data["value"].xs,
                current_prog_fspv.data["value"].ys,
            ))
            field_prog = np.vstack((
                field_prog_fspv.data["value"].xs,
                field_prog_fspv.data["value"].ys,
            ))
            bhys_corr = np.vstack((
                bhys_corr_fspv.data["value"].xs,
                bhys_corr_fspv.data["value"].ys,
            ))

            cycle_data = hystcomp_utils.cycle_data.CycleData(
                cycle=str(fspv.data["lsaCycleName"]),
                user=selector,
                cycle_timestamp=fspv.header.cycle_timestamp,
                current_prog=current_prog,
                field_prog=field_prog,
                correction=bhys_corr,
            )

            self.cycleDataAvailable.emit(cycle_data)
            return

        # unknown endpoint
        log.error(f"Unknown endpoint: {fspv.query.endpoint}")
        return
