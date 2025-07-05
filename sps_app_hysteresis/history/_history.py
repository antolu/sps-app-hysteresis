from __future__ import annotations

import logging

from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore

from ..widgets.history_widget._unified_model import CycleListModel
from ._reference_cycles import ReferenceCycles

log = logging.getLogger(__package__)


class PredictionHistory(QtCore.QObject):
    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

        # map from cycle name to list model
        self._history: dict[str, CycleListModel] = {}
        self._references = ReferenceCycles(parent=self)

        self._references.referenceChanged.connect(self.onReferenceChanged)

    @property
    def references(self) -> ReferenceCycles:
        return self._references

    def add_cycle(self, cycle_data: CycleData) -> None:
        """
        Function to be called when new predictions are made, and a new cycle
        should be added to the history.
        """
        if cycle_data.cycle not in self._history:
            self._history[cycle_data.cycle] = CycleListModel(parent=self)

        self._history[cycle_data.cycle].append(cycle_data)

    def update_cycle(self, cycle_data: CycleData) -> None:
        """
        Function to be called whenever fields in a CycleData is updated, and the corresponding plots
        should be updated as well.
        """
        if cycle_data.cycle in self._history:
            log.debug(f"[{cycle_data}]: Updating cycle data in history.")
            self._history[cycle_data.cycle].update(cycle_data)
        elif cycle_data.field_pred is None:  # no prediction
            log.debug(
                f"Cycle {cycle_data.cycle} not found in history, no prediction to update."
            )
        elif cycle_data.field_meas is None:  # no measurement
            log.debug(
                f"Cycle {cycle_data.cycle} not found in history, no measurement to update."
            )
        else:
            msg = f"Cycle {cycle_data.cycle} not found in history, cannot update."
            log.error(msg)

    def model(self, cycle: str) -> CycleListModel:
        # create new model if cycle does not exist, to always keep track of history
        if cycle not in self._history:
            self._history[cycle] = CycleListModel(parent=self)

        return self._history[cycle]

    @QtCore.Slot(CycleData)
    def onReferenceChanged(self, cycle_data: CycleData) -> None:
        if cycle_data.cycle not in self._history:
            self._history[cycle_data.cycle] = CycleListModel(parent=self)

        self._history[cycle_data.cycle].set_reference(cycle_data)
