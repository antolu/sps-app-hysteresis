from __future__ import annotations

import logging
import typing
from collections import deque

from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore, QtGui

log = logging.getLogger(__package__)


class HistoryListModel(QtCore.QAbstractListModel):
    """
    Model for list of predicted cycles.

    On click, the cycle is plotted and highlighted with a
    unique color.
    """

    itemAdded = QtCore.Signal(CycleData)
    """ Emitted when an item is added to the list. """
    itemUpdated = QtCore.Signal(CycleData)
    itemRemoved = QtCore.Signal(CycleData)
    """ Emitted when an item is removed from the list. """

    referenceChanged = QtCore.Signal(CycleData)

    def __init__(
        self,
        max_len: int = 10,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._data: deque[CycleData] = deque(maxlen=max_len)
        self._reference: CycleData | None = None

    def set_reference(self, reference: CycleData) -> None:
        old_reference = self._reference
        if old_reference == reference:
            log.debug("Reference is the same as the old reference.")
            return

        # get the current reference index and emit dataChanged
        if old_reference is not None:
            idx = self._deque_idx(old_reference)
            model_index = self.index(idx, 0)
            self.dataChanged.emit(model_index, model_index, [QtCore.Qt.FontRole])

        self._reference = reference
        self.referenceChanged.emit(reference)

    def reference(self) -> CycleData | None:
        return self._reference

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> typing.Any:
        row = self._calc_real_row(index.row())

        cycle_data = self._data[row]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return f"{str(cycle_data.cycle_time)[:-7]}"
        if role == QtCore.Qt.ItemDataRole.FontRole and (
            self._reference is not None
            and cycle_data.cycle_timestamp == self._reference.cycle_timestamp
        ):
            font = QtGui.QFont()
            font.setBold(True)
            return font

        return None

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        type(parent)  # avoid unused variable warning
        return len(self._data)

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0
    ) -> typing.Any:
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return "Cycle"
            if orientation == QtCore.Qt.Vertical:
                return str(section + 1)

        return None

    def itemAt(self, index: QtCore.QModelIndex) -> CycleData:
        row = self._calc_real_row(index.row())

        return self._data[row]

    def set_max_len(self, max_len: int) -> None:
        if max_len < 1:
            msg = "Max length must be greater than 1."
            raise ValueError(msg)

        assert self._data.maxlen is not None
        if max_len < self._data.maxlen and len(self._data) > max_len:
            to_remove = list(self._data)[max_len:]
            self._data = deque(list(self._data)[:max_len], maxlen=max_len)

            for data in to_remove:
                self.itemRemoved.emit(data)
        else:
            self._data = deque(self._data, maxlen=max_len)

    def get_max_len(self) -> int:
        assert self._data.maxlen is not None
        return self._data.maxlen

    def append(self, data: CycleData) -> None:
        if len(self._data) == self._data.maxlen:
            self.remove_last(keep_reference=True)

        self._data.append(data)
        index = (
            QtCore.QModelIndex(),
            len(self._data) - 1,
            len(self._data) - 1,
        )
        self.rowsAboutToBeInserted.emit(*index)
        self.rowsInserted.emit(*index)
        self.itemAdded.emit(data)

    def remove_last(self, *, keep_reference: bool = True) -> CycleData:
        if not self._data:
            msg = "No items to remove."
            raise IndexError(msg)

        data = self._data.pop()
        index = self.index(len(self._data), 0)
        self.rowsAboutToBeRemoved.emit(*index)
        self.rowsRemoved.emit(*index)

        if data == self._reference and keep_reference:
            reference = data
            data = self._data.pop()

            # move the reference to the end of the list and emit dataChanged
            self._data.appendleft(reference)
            ref_idx = (self.index(len(self._data), 0), 0, 0)
            self.dataChanged.emit(*ref_idx, [QtCore.Qt.FontRole, QtCore.Qt.DisplayRole])

        self.itemRemoved.emit(data)

    def update(self, data: CycleData) -> None:
        # check if the cycle is already in the list
        idx = self._deque_idx(data)

        if idx == -1:
            msg = f"Cycle {data.cycle_timestamp} not found in the list. Cannot update entry."
            log.error(msg)
            return

        self._data[idx] = data
        model_index = self.index(idx, 0)

        self.dataChanged.emit(model_index, model_index, [QtCore.Qt.DisplayRole])
        self.itemUpdated.emit(data)

    def _deque_idx(self, data: CycleData) -> int:
        for idx, item in enumerate(self._data):
            if item.cycle_timestamp == data.cycle_timestamp:
                return idx

        return -1

    def clear(self) -> None:
        for data in self._data:
            self.itemRemoved.emit(data)

        self._data.clear()
        self.modelReset.emit()

    max_len = property(get_max_len, set_max_len)

    @QtCore.Slot(QtCore.QModelIndex)
    def clicked(self, index: QtCore.QModelIndex) -> None:
        self.dataChanged.emit(index, index, [QtCore.Qt.BackgroundRole])

    @property
    def buffered_data(self) -> list[CycleData]:
        return list(self._data)

    def _calc_real_row(self, row: int) -> int:
        """
        Show the data in reverse order, i.e. the last item is shown first.
        """
        if row >= len(self._data):
            msg = "Index out of range."
            raise IndexError(msg)

        return len(self._data) - row - 1
