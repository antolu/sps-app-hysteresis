from __future__ import annotations

import typing
from collections import deque

from qtpy import QtCore

from hystcomp_utils.cycle_data import CycleData
from ._dataclass import PredictionItem


class PredictionListModel(QtCore.QAbstractListModel):
    """
    Model for list of predicted cycles.

    On click, the cycle is plotted and highlighted with a
    unique color.
    """

    itemAdded = QtCore.Signal(PredictionItem)
    """ Emitted when an item is added to the list. """
    itemRemoved = QtCore.Signal(PredictionItem)
    """ Emitted when an item is removed from the list. """

    def __init__(
        self, max_len: int = 10, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self._data: deque[PredictionItem] = deque(maxlen=max_len)

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> typing.Any:
        row = self._calc_real_row(index.row())

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            cycle_data = self._data[row].cycle_data

            return f"{str(cycle_data.cycle_time)[:-7]}"
        elif role == QtCore.Qt.ItemDataRole.BackgroundRole:
            return self._data[row].color

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
            elif orientation == QtCore.Qt.Vertical:
                return str(section + 1)

        return None

    def itemAt(self, index: QtCore.QModelIndex) -> PredictionItem:
        row = self._calc_real_row(index.row())

        return self._data[row]

    def set_max_len(self, max_len: int) -> None:
        if max_len < 1:
            raise ValueError("Max length must be greater than 0.")

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

        item = PredictionItem(cycle_data=data)
        self._data.append(item)
        index = (
            QtCore.QModelIndex(),
            len(self._data) - 1,
            len(self._data) - 1,
        )
        self.rowsAboutToBeInserted.emit(*index)
        self.rowsInserted.emit(*index)
        self.itemAdded.emit(item)

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
    def buffered_data(self) -> list[PredictionItem]:
        return list(self._data)

    def _calc_real_row(self, row: int) -> int:
        """
        Show the data in reverse order, i.e. the last item is shown first.
        """
        if row >= len(self._data):
            raise IndexError("Index out of range.")

        return len(self._data) - row - 1
