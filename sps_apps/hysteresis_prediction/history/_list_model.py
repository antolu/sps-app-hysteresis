from __future__ import annotations

import logging
import typing
from collections import deque

from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore

log = logging.getLogger(__name__)


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

    def __init__(self, max_len: int = 10, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

        self._data: deque[CycleData] = deque(maxlen=max_len)

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> typing.Any:
        row = self._calc_real_row(index.row())

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            cycle_data = self._data[row]

            return f"{str(cycle_data.cycle_time)[:-7]}"

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
            to_remove = self._data.popleft()
            index = (QtCore.QModelIndex(), 0, 0)
            self.rowsAboutToBeRemoved.emit(*index)
            self.rowsRemoved.emit(*index)
            self.itemRemoved.emit(to_remove)

        self._data.append(data)
        index = (
            QtCore.QModelIndex(),
            len(self._data) - 1,
            len(self._data) - 1,
        )
        self.rowsAboutToBeInserted.emit(*index)
        self.rowsInserted.emit(*index)
        self.itemAdded.emit(data)

    def update(self, data: CycleData) -> None:
        # check if the cycle is already in the list
        idx = self._deque_idx(data)

        if idx == -1:
            msg = f"Cycle {data.cycle_timestamp} not found in the list. Cannot update entry."
            log.error(msg)
            return

        self._data[idx] = data

        index = (
            QtCore.QModelIndex(),
            idx,
            idx,
        )
        self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole])
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
