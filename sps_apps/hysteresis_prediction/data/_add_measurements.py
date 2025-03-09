from __future__ import annotations

import logging

import hystcomp_utils.cycle_data
import pyda
import pyda.access
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import BufferedSubscription, BufferedSubscriptionEventBuilder

log = logging.getLogger(__name__)


PARAM_I_MEAS = "MBI/LOG.I.MEAS"
PARAM_B_MEAS = "SR.BMEAS-SP-B-SD/CycleSamples#samples"


class AddMeasurementsEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        param_i_meas: str = PARAM_I_MEAS,
        param_b_meas: str | None = PARAM_B_MEAS,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        assert param_b_meas is not None, "B_MEAS parameter must be provided."
        super().__init__(
            buffered_subscriptions=[
                BufferedSubscription("I_MEAS", param_i_meas),
                BufferedSubscription("B_MEAS", param_b_meas),
            ],
            provider=provider,
            parent=parent,
        )

    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        msg = f"{self.__class__.__name__} does not subscribe to triggers."
        raise NotImplementedError(msg)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        selector = cycle_data.user

        msg = f"Received cycle data for {selector}"
        log.debug(msg)

        if not self._buffer_has_data(
            PARAM_I_MEAS, selector
        ) or not self._buffer_has_data(PARAM_B_MEAS, selector):
            msg = f"Missing measurements for {selector}"
            log.error(msg)
            return

        i_meas = self._get_buffered_data(PARAM_I_MEAS, selector).value.get("value")
        b_meas = (
            self._get_buffered_data(PARAM_B_MEAS, selector).value.get("value") / 1e4
        )  # G to T

        cycle_data.current_meas = i_meas
        cycle_data.field_meas = b_meas

        msg = f"Added measurements to cycle data for {selector}: I_MEAS={i_meas}, B_MEAS={b_meas}"
        log.debug(msg)

        self.cycleDataAvailable.emit(cycle_data)
