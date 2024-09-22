from __future__ import annotations

import logging

import hystcomp_utils.cycle_data
import pyda
import pyda.data
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import BufferedSubscriptionEventBuilder, BufferedSubscription


log = logging.getLogger(__name__)


PARAM_I_MEAS = "MBI/LOG.I.MEAS"
PARAM_B_MEAS = "SR.BMEAS-SP-B-SD/CycleSamples#samples"


class AddMeasurementsEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            buffered_subscriptions=[
                BufferedSubscription("I_MEAS", PARAM_I_MEAS),
                BufferedSubscription("B_MEAS", PARAM_B_MEAS),
            ],
            provider=provider,
            parent=parent,
        )

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        msg = f"{self.__class__.__name__} does not subscribe to triggers."
        raise NotImplementedError(msg)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        selector = cycle_data.user

        msg = f"Received cycle data for {selector}"
        log.debug(msg)

        i_meas = self._get_buffered_data(PARAM_I_MEAS, selector).value.get("value")
        b_meas = (
            self._get_buffered_data(PARAM_B_MEAS, selector).value.get("value") / 1e4
        )  # G to T

        cycle_data.current_meas = i_meas
        cycle_data.field_meas = b_meas

        msg = f"Added measurements to cycle data for {selector}: I_MEAS={i_meas}, B_MEAS={b_meas}"
        log.debug(msg)

        self.cycleDataAvailable.emit(cycle_data)
