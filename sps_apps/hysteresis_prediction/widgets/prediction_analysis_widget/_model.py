"""
This module contains the model for the prediction analysis widget.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any

from qtpy.QtCore import QAbstractListModel, QModelIndex, QObject, Qt, Signal

from ...data import SingleCycleData

log = logging.getLogger(__name__)


class PredictionListModel(QAbstractListModel):
    def __init__(
        self, max_len: int = 10, parent: QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self._data: deque[SingleCycleData] = deque(maxlen=max_len)

    def data(self, index: QModelIndex, role: int = 0) -> Any:
        row = index.row()

        if role == Qt.ItemDataRole.DisplayRole:
            cycle_data = self._data[row]

            return f"{cycle_data.user}: {str(cycle_data.cycle_time)[:-7]}"

        return None

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        type(parent)
        return len(self._data)

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = 0
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Horizontal:
                return "Cycle"
            elif orientation == Qt.Vertical:
                return str(section + 1)

        return None

    def set_max_len(self, max_len: int) -> None:
        if max_len < 1:
            raise ValueError("Max length must be greater than 0.")

        assert self._data.maxlen is not None
        if max_len < self._data.maxlen and len(self._data) > max_len:
            self._data = deque(list(self._data)[:max_len], maxlen=max_len)
        else:
            self._data = deque(self._data, maxlen=max_len)

    def get_max_len(self) -> int:
        assert self._data.maxlen is not None
        return self._data.maxlen

    def append(self, data: SingleCycleData) -> None:
        self._data.append(data)

        self.modelReset.emit()

    max_len = property(get_max_len, set_max_len)

    @property
    def buffered_data(self) -> list[SingleCycleData]:
        return list(self._data)


class PredictionAnalysisModel(QObject):
    newData = Signal(SingleCycleData)
    """ Triggered by new data acquisition (new predictions) """

    maxBufferSizeChanged = Signal(int)
    """ Triggered when the buffer size changes """

    def __init__(
        self, max_buffer_samples: int = 10, parent: QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self._list_model = PredictionListModel(max_len=max_buffer_samples)

        self._watch_supercycle = False
        self._supercycle_patience = 0

        self.maxBufferSizeChanged.connect(self.set_max_buffer_samples)

    @property
    def list_model(self) -> PredictionListModel:
        return self._list_model

    def set_max_buffer_samples(self, max_buffer_samples: int) -> None:
        self._list_model.max_len = max_buffer_samples

    def get_max_buffer_samples(self) -> int:
        return self._list_model.max_len

    max_buffer_samples = property(
        get_max_buffer_samples, set_max_buffer_samples
    )

    def set_watch_supercycle(self, watch_supercycle: bool) -> None:
        self._watch_supercycle = watch_supercycle

    def get_watch_supercycle(self) -> bool:
        return self._watch_supercycle

    watch_supercycle = property(get_watch_supercycle, set_watch_supercycle)

    def set_supercycle_patience(self, supercycle_patience: int) -> None:
        self._supercycle_patience = supercycle_patience

    def get_supercycle_patience(self) -> int:
        return self._supercycle_patience

    supercycle_patience = property(
        get_supercycle_patience, set_supercycle_patience
    )
