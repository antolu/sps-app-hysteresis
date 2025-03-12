from __future__ import annotations

import logging
import typing

from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore

log = logging.getLogger(__name__)


class ReferenceCycles(QtCore.QObject):
    referenceChanged = QtCore.Signal(CycleData)
    """ Emitted when the reference is changed. The cycle information is present in the CycleData Object """

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        self._references: dict[str, CycleData] = {}

    @QtCore.Slot(CycleData)
    def onNewCycleData(self, cycle_data: CycleData) -> None:
        if self.is_reference(cycle_data):
            if cycle_data.cycle in self._references:
                msg = (
                    f"Reference for {cycle_data.cycle} at "
                    f"{self._references[cycle_data.cycle].cycle_time} already exists."
                    f"Will be replaced by {cycle_data.cycle_time}"
                )
                log.info(msg)
            else:
                log.info(
                    f"Reference for {cycle_data.cycle} at {cycle_data.cycle_time} created."
                )
            self._references[cycle_data.cycle] = cycle_data

            self.referenceChanged.emit(cycle_data)
        log.debug(f"Reference is not changed for {cycle_data.cycle}")

    @staticmethod
    def is_reference(cycle_data: CycleData) -> bool:
        return cycle_data.cycle_timestamp == cycle_data.reference_timestamp

    def __getitem__(self, key: str) -> CycleData:
        return self._references[key]

    def __iter__(self) -> typing.Iterator[str]:
        return iter(self._references)

    def __len__(self) -> int:
        return len(self._references)
