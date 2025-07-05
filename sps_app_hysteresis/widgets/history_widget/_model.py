"""
This module contains the model for the prediction analysis widget.
"""

from __future__ import annotations

import logging
import typing

from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore, QtGui

from ...history import HistoryListModel
from ._dataclass import PlotItem

log = logging.getLogger(__package__)


BEAM_IN = "SIX.MC-CTML/ControlValue#controlValue"
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue#controlValue"


class PredictionListModel(QtCore.QAbstractListModel):
    """
    Model for list of predicted cycles.

    DEPRECATED: Use CycleListModel instead. This class wraps HistoryListModel
    and adds plot metadata, but CycleListModel provides the same functionality
    in a single, more efficient implementation.

    On click, the cycle is plotted and highlighted with a
    unique color.
    """

    itemAdded = QtCore.Signal(PlotItem)
    """ Emitted when an item is added to the list. """
    itemUpdated = QtCore.Signal(PlotItem)
    """ Emitted when an item is updated in the list. """
    itemRemoved = QtCore.Signal(PlotItem)
    """ Emitted when an item is removed from the list. """

    referenceChanged = QtCore.Signal(PlotItem)
    """ Emitted when the reference is changed. """

    def __init__(
        self, data_source: HistoryListModel, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self.data_source = data_source
        self.data_source.referenceChanged.connect(self.onReferenceChanged)
        self.data_source.itemAdded.connect(self.newItem)
        self.data_source.itemUpdated.connect(self.updateItem)
        self.data_source.itemRemoved.connect(self.removeItem)

        self.data_source.rowsAboutToBeInserted.connect(self.rowsAboutToBeInserted.emit)
        self.data_source.rowsInserted.connect(self.rowsInserted.emit)
        self.data_source.rowsAboutToBeRemoved.connect(self.rowsAboutToBeRemoved.emit)
        self.data_source.rowsRemoved.connect(self.rowsRemoved.emit)
        self.data_source.dataChanged.connect(self.dataChanged.emit)
        self.data_source.modelReset.connect(self.modelReset.emit)

        self._plot_metadata: dict[int, PlotItem] = {}
        self._reference: PlotItem | None = None

        for cycle_data in self.data_source._data:  # noqa: SLF001
            self.newItem(cycle_data)
        self.modelReset.emit()

    @property
    def reference(self) -> PlotItem | None:
        return self._reference

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> typing.Any:
        cycle_data = self.data_source.itemAt(index)
        value = self.data_source.data(index, role)

        if value is not None:
            return value
        if value is None and role == QtCore.Qt.ItemDataRole.BackgroundColorRole:
            item = self._plot_metadata.get(cycle_data.cycle_timestamp)
            if item is None:
                return None
            return QtGui.QColor(item.color or "white")

        return None

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        type(parent)  # avoid unused variable warning
        return self.data_source.rowCount()

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0
    ) -> typing.Any:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return "Cycle"
            if orientation == QtCore.Qt.Vertical:
                return str(section + 1)

        return None

    def itemAt(self, index: QtCore.QModelIndex) -> PlotItem:
        cycle = self.data_source.itemAt(index)
        return self._plot_metadata[cycle.cycle_timestamp]

    def newItem(self, cycle_data: CycleData) -> None:
        item = PlotItem(cycle_data=cycle_data)

        self._plot_metadata[cycle_data.cycle_timestamp] = item
        self.itemAdded.emit(item)

        if (
            cycle_data.cycle_timestamp == cycle_data.reference_timestamp
            and cycle_data == self.data_source.reference
        ):
            self._reference = item
            self.referenceChanged.emit(item)

    def updateItem(self, cycle_data: CycleData) -> None:
        item = self._plot_metadata[cycle_data.cycle_timestamp]
        item.cycle_data = cycle_data

    def removeItem(self, cycle_data: CycleData) -> None:
        """
        Slot to remove internal data based on cycle timestamp, should be
        connected to data source.
        """
        item = self._plot_metadata.pop(cycle_data.cycle_timestamp)

        self.itemRemoved.emit(item)

    def clear(self) -> None:
        self.data_source.clear()
        for data in self._plot_metadata.values():
            self.itemRemoved.emit(data)

        self._plot_metadata.clear()

    @QtCore.Slot(QtCore.QModelIndex)
    def clicked(self, index: QtCore.QModelIndex) -> None:
        self.dataChanged.emit(index, index, [QtCore.Qt.BackgroundRole])

    @QtCore.Slot(CycleData)
    def onReferenceChanged(self, cycle_data: CycleData) -> None:
        # find the item in the list
        item = self._plot_metadata.get(cycle_data.cycle_timestamp)
        if item is None:
            log.debug(
                f"Reference cycle {cycle_data.cycle} not found in list. Waiting for it to come."
            )
            return

        self._reference = item
        self.referenceChanged.emit(item)
