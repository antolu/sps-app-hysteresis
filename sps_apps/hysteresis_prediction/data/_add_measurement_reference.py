from __future__ import annotations

import datetime
import logging

import hystcomp_utils.cycle_data
import numpy as np
import numpy.typing as npt
import pyda
import pyda.data
from qtpy import QtCore

from ._event_builder_abc import EventBuilderAbc

NS2S = 1e-9


log = logging.getLogger(__name__)


class AddMeasurementReferencesEventBuilder(EventBuilderAbc):
    def __init__(
        self,
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            parent=parent,
        )

        self._reference_timestamps: dict[str, float] = {}
        self._reference_fields: dict[str, npt.NDArray[np.float64]] = {}

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        msg = f"{self.__class__.__name__} does not subscribe to triggers."
        raise NotImplementedError(msg)

    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:

        log.debug(f"{cycle_data}: Triggered to add reference")

        self._maybe_save_reference(cycle_data)

        log.debug(f"{cycle_data}: Adding reference field to the cycle data")
        if cycle_data.cycle not in self._reference_timestamps:
            log.error(f"{cycle_data}: No reference found")
            return None

        cycle_data.field_meas_ref = self._reference_fields[cycle_data.cycle]

        ref_time = datetime.datetime.fromtimestamp(
            self._reference_timestamps[cycle_data.cycle] * NS2S
        )
        log.debug(f"{cycle_data}: Added reference field from " f"timestamp {ref_time}")

        self.cycleDataAvailable.emit(cycle_data)

    def _maybe_save_reference(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        if cycle_data.cycle not in self._reference_timestamps:
            if cycle_data.field_meas is None:
                log.error(f"{cycle_data}: field_meas is None")
                return

            log.debug(f"{cycle_data}: Saving reference for the first time.")

            self._reference_timestamps[cycle_data.cycle] = cycle_data.cycle_timestamp
            self._reference_fields[cycle_data.cycle] = cycle_data.field_meas
            return

        if cycle_data.reference_timestamp is None:
            log.error(f"{cycle_data}: reference_timestamp is None")
            return

        if (
            cycle_data.reference_timestamp
            != self._reference_timestamps[cycle_data.cycle]
        ):
            # update reference
            old_reference_stamp = self._reference_timestamps[cycle_data.cycle]
            old_time = datetime.datetime.fromtimestamp(old_reference_stamp * NS2S)
            new_time = datetime.datetime.fromtimestamp(
                cycle_data.cycle_timestamp * NS2S
            )

            log.debug(
                f"{cycle_data}: Reference timestamp changed from {old_time} -> {new_time}"
                " and field updated."
            )

            self._reference_timestamps[cycle_data.cycle] = cycle_data.cycle_timestamp
            self._reference_fields[cycle_data.cycle] = cycle_data.field_meas
            return

    def resetReference(self, cycle_name: str | None = None) -> None:
        if cycle_name is None or cycle_name == "all":
            log.info("Resetting all references")
            self._reference_timestamps = {}
            self._reference_fields = {}
        else:
            log.info(f"Resetting reference for {cycle_name}")
            self._reference_timestamps.pop(cycle_name, None)
            self._reference_fields.pop(cycle_name, None)
