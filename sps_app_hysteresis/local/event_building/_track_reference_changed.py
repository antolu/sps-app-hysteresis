"""
Tracks a trigger, and when the trigger is received, find out which
cycle the trigger selector belongs to, and send the cycle as a signal.
"""

from __future__ import annotations

import logging
import sys

import pyda
import pyda.access
import pyda_japc
from qtpy import QtCore

from ._event_builder_abc import (
    BufferedSubscription,
    BufferedSubscriptionEventBuilder,
    Subscription,
)
from ._start_cycle import TRIGGER as START_CYCLE

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


log = logging.getLogger(__name__)


TRIGGER = "rmi://virtual_sps/SPSBEAM/B"


class TrackReferenceChangedEventBuilder(BufferedSubscriptionEventBuilder):
    resetReference = QtCore.Signal(str)
    """ Emitted when the reference is/should be reset. """

    def __init__(
        self,
        param_trigger: str = TRIGGER,
        param_start_cycle: str = START_CYCLE,
        *,
        provider: pyda_japc.JapcProvider | None = None,
        no_metadata_source=True,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions=[
                Subscription("TRIGGER", param_trigger, ignore_first_updates=True)
            ],
            buffered_subscriptions=[
                BufferedSubscription("START_CYCLE", param_start_cycle)
            ],
            provider=provider,
            no_metadata_source=no_metadata_source,
            parent=parent,
        )

        self._trigger_param = param_trigger
        self._start_cycle_param = param_start_cycle

    @override
    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        parameter = str(fspv.query.endpoint)
        selector = str(fspv.header.selector)

        if parameter != self._trigger_param:
            log.error(f"Received unknown acquisition for {parameter}@{selector}.")
            return

        if self._start_cycle_param not in self._buffers:
            log.error(
                f"No buffered acquisition received for {self._start_cycle_param}, can't determine cycle to user mapping."
            )
            return

        if selector not in self._buffers[self._start_cycle_param]:
            log.debug(
                f"Received reset reference signal for cycle without buffered data: {selector}."
            )
            return

        fspv = self._buffers[self._start_cycle_param][selector]
        cycle = fspv.data["lsaCycleName"]

        log.info(f"Resetting reference for {cycle} when {self._trigger_param} changed.")
        self.resetReference.emit(cycle)
