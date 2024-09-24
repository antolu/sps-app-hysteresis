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
PARAM_BHYS_CORRECTION = (
    "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPSBEAM.BHYS-CORRECTION/Acquisition"
)


class CreateCycleEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        cycle_warning: str = "SX.CZERO-CTML/CycleWarning",
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            [
                Subscription("CycleWarning", cycle_warning, ignore_first_updates=True),
            ],
            buffered_subscriptions=[
                BufferedSubscription("IREF", PARAM_I_PROG),
                BufferedSubscription("B", PARAM_B_PROG),
                BufferedSubscription("BHYS-CORRECTION", PARAM_BHYS_CORRECTION),
            ],
            provider=provider,
            parent=parent,
            no_metadata_source=True,
        )

        self._trigger = cycle_warning

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        selector = str(fspv.value.header.selector)
        msg = f"Received {fspv.query.endpoint} with {fspv.value.get('value')}"
        log.debug(msg)

        if str(fspv.query.endpoint) == self._trigger:
            if (
                not self._buffer_has_data(PARAM_I_PROG, selector)
                or not self._buffer_has_data(PARAM_B_PROG, selector)
                or not self._buffer_has_data(PARAM_BHYS_CORRECTION, selector)
            ):
                msg = f"Missing data for {selector}. Skipping cycle creation."
                log.error(msg)
                return

            current_prog_fspv = self._get_buffered_data(PARAM_I_PROG, selector)
            field_prog_fspv = self._get_buffered_data(PARAM_B_PROG, selector)
            bhys_corr_fspv = self._get_buffered_data(PARAM_BHYS_CORRECTION, selector)

            current_prog = np.vstack(
                (
                    current_prog_fspv.value.get("value").xs,
                    current_prog_fspv.value.get("value").ys,
                )
            )
            field_prog = np.vstack(
                (
                    field_prog_fspv.value.get("value").xs,
                    field_prog_fspv.value.get("value").ys,
                )
            )
            bhys_corr = np.vstack(
                (
                    bhys_corr_fspv.value.get("value").xs,
                    bhys_corr_fspv.value.get("value").ys,
                )
            )

            cycle_data = hystcomp_utils.cycle_data.CycleData(
                cycle=str(fspv.value.get("lsaCycleName")),
                user=selector,
                cycle_timestamp=fspv.value.header.cycle_timestamp,
                current_prog=current_prog,
                field_prog=field_prog,
                correction=bhys_corr,
            )

            self.cycleDataAvailable.emit(cycle_data)
            return

        # unknown endpoint
        log.error(f"Unknown endpoint: {fspv.query.endpoint}")
        return
