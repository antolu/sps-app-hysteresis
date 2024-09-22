from __future__ import annotations

import logging

import hystcomp_utils.cycle_data
import pyda
import pyda.data
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import (
    EventBuilderAbc,
    Subscription,
)

log = logging.getLogger(__name__)


TRIGGER = "XTIM.SX.SCY-CT/Acquisition"


class StartCycleEventBuilder(EventBuilderAbc):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions=[Subscription("SCY", TRIGGER)],
            provider=provider,
            parent=parent,
        )

        self._cycle_data_buffer: dict[str, hystcomp_utils.cycle_data.CycleData] = {}

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        parameter = str(fspv.query.endpoint)
        selector = str(fspv.query.context)

        if parameter != TRIGGER:
            msg = f"Received unknown acquisition for {parameter}@{selector}."
            raise ValueError(msg)

        if selector not in self._cycle_data_buffer:
            log.error(f"Received trigger for cycle without data: {selector}.")
            return

        cycle_data = self._cycle_data_buffer[selector]
        msg = f"[{cycle_data}]: Cycle is starting."
        log.debug(msg)

        self.onNewCycleData(cycle_data)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"[{cycle_data}]: Received data, saving to buffer."
        log.debug(msg)

        self._cycle_data_buffer[cycle_data.user] = cycle_data
