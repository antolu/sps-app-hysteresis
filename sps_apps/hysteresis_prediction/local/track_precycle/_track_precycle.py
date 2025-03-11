"""
This converter tracks precycle sequences, which by default is
LHCPILOT - MD1 - LHCPILOT - MD1 - LHCPILOT - MD1
"""

from __future__ import annotations

import logging
import os
import pathlib
import typing

import hystcomp_utils.cycle_data
import pandas as pd
from qtpy import QtCore

from ...local.event_building import EventBuilderAbc

DEFAULT_PRECYCLE_SEQUENCE = [
    "SPS.USER.LHCPILOT",
    "SPS.USER.MD1",
]


_HERE = pathlib.Path(__file__).parent


log = logging.getLogger(__name__)


class TrackPrecycleEventBuilder(EventBuilderAbc):
    precycleStarted = QtCore.Signal()
    precycleEnded = QtCore.Signal()

    def __init__(
        self,
        precycle_sequence: typing.Sequence[str] | None = None,
        patience: int = 2,
        ref_field_path: str | os.PathLike = _HERE,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent=parent)

        self._precycle_sequence = precycle_sequence or DEFAULT_PRECYCLE_SEQUENCE

        self._patience = patience
        self._patience_counter = 0
        self._precycle_index = 0
        self._precycle_active = False

        # read parquet files and assume the file names correspond to users
        self._ref_fields = {
            path.stem: pd.read_parquet(path)
            for path in pathlib.Path(ref_field_path).glob("*.parquet")
        }

    @property
    def precycle_active(self) -> bool:
        return self._precycle_active

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        next_user = self._precycle_sequence[self._precycle_index]

        if cycle_data.user == next_user:
            log.info(
                f"Expected user: {cycle_data.user} == {next_user}, in precycle sequence."
            )
            self._precycle_index += 1
        elif self._precycle_index == 0:
            log.info(
                f"Unexpected user: {cycle_data.user} != {next_user}, no longer in precycle sequence."
            )
            self.reset_precycle()
            self.precycleEnded.emit()

        if self._precycle_index == len(self._precycle_sequence):
            self._precycle_index = 0
            self._patience_counter += 1

            if self._patience_counter >= self._patience and not self._precycle_active:
                log.info("Precycle sequence active.")
                self._precycle_active = True
                self.precycleStarted.emit()

        if self._precycle_active:
            ref_data = self._ref_fields.get(cycle_data.user)

            if ref_data is None:
                log.warning(f"Reference data not found for {cycle_data.user}.")
            else:
                log.info(
                    f"Updating cycle data with reference data for {cycle_data.user}."
                )
                cycle_data.field_meas = ref_data["B_meas_T"].to_numpy()
                cycle_data.current_meas = ref_data["I_meas_A"].to_numpy()

        self.cycleDataAvailable.emit(cycle_data)

    def reset_precycle(self) -> None:
        self._precycle_index = 0
        self._patience_counter = 0
        self._precycle_active = False
