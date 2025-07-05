"""
Model for cycle data management.

This model provides a single source of truth for cycle data
with integrated plot metadata management.
"""

from __future__ import annotations

import logging
import typing
from collections import deque

from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore, QtGui

from ._dataclass import PlotItem

log = logging.getLogger(__package__)


class CycleListModel(QtCore.QAbstractListModel):
    """
    Model for cycle data with list management and plot metadata.

    This provides comprehensive cycle data management with integrated
    plot metadata support.
    """

    # Data signals
    itemAdded = QtCore.Signal(CycleData)
    itemUpdated = QtCore.Signal(CycleData)
    itemRemoved = QtCore.Signal(CycleData)

    # Plot signals
    plotItemAdded = QtCore.Signal(PlotItem)
    plotItemUpdated = QtCore.Signal(PlotItem)
    plotItemRemoved = QtCore.Signal(PlotItem)

    # Reference signals
    referenceChanged = QtCore.Signal(CycleData)

    def __init__(
        self,
        max_len: int = 10,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        # Data storage
        self._cycle_data: deque[CycleData] = deque(maxlen=max_len)
        self._plot_metadata: dict[int, PlotItem] = {}  # timestamp -> PlotItem

        # Reference tracking
        self._reference: CycleData | None = None

    @property
    def reference(self) -> CycleData | None:
        """Get the current reference cycle."""
        return self._reference

    @property
    def reference_plot_item(self) -> PlotItem | None:
        """Get the PlotItem for the current reference cycle."""
        if self._reference is None:
            return None
        return self._plot_metadata.get(self._reference.cycle_timestamp)

    def set_reference(self, reference: CycleData) -> None:
        """Set the reference cycle and update UI accordingly."""
        old_reference = self._reference
        if (
            old_reference is not None
            and old_reference.cycle_timestamp == reference.cycle_timestamp
        ):
            log.debug("Reference is the same as the old reference.")
            return

        # Update old reference display
        if old_reference is not None:
            idx = self._get_cycle_index(old_reference)
            if idx != -1:
                model_index = self.index(idx, 0)
                self.dataChanged.emit(model_index, model_index, [QtCore.Qt.FontRole])

        self._reference = reference

        # Update new reference display
        idx = self._get_cycle_index(reference)
        if idx != -1:
            model_index = self.index(idx, 0)
            self.dataChanged.emit(model_index, model_index, [QtCore.Qt.FontRole])

        self.referenceChanged.emit(reference)

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> typing.Any:
        """Get data for the list view display."""
        if not index.isValid():
            return None

        row = self._calc_real_row(index.row())
        if row >= len(self._cycle_data) or row < 0:
            return None

        cycle_data = self._cycle_data[row]

        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            return f"{str(cycle_data.cycle_time)[:-7]}"
        if role == QtCore.Qt.ItemDataRole.FontRole:
            # Bold if this is the reference
            if (
                self._reference is not None
                and cycle_data.cycle_timestamp == self._reference.cycle_timestamp
            ):
                font = QtGui.QFont()
                font.setBold(True)
                return font
        elif role == QtCore.Qt.ItemDataRole.BackgroundColorRole:
            # Show plot color if item has one
            plot_item = self._plot_metadata.get(cycle_data.cycle_timestamp)
            if plot_item is not None and plot_item.color is not None:
                return plot_item.color

        return None

    def rowCount(self, parent: QtCore.QModelIndex | None = None) -> int:
        """Get the number of cycles in the model."""
        return len(self._cycle_data)

    def headerData(
        self, section: int, orientation: QtCore.Qt.Orientation, role: int = 0
    ) -> typing.Any:
        """Get header data for the list view."""
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return "Cycle"
            if orientation == QtCore.Qt.Vertical:
                return str(section + 1)
        return None

    def get_cycle_at(self, index: QtCore.QModelIndex) -> CycleData:
        """Get the CycleData at the given index."""
        row = self._calc_real_row(index.row())
        return self._cycle_data[row]

    def get_plot_item_at(self, index: QtCore.QModelIndex) -> PlotItem:
        """Get the PlotItem at the given index."""
        cycle = self.get_cycle_at(index)
        return self._plot_metadata[cycle.cycle_timestamp]

    def append(self, cycle_data: CycleData) -> None:
        """Add a new cycle to the model."""
        # Handle queue overflow
        if len(self._cycle_data) == self._cycle_data.maxlen:
            self._remove_oldest(keep_reference=True)

        # Add new data
        self._cycle_data.append(cycle_data)

        # Create plot metadata
        plot_item = PlotItem(cycle_data=cycle_data)
        self._plot_metadata[cycle_data.cycle_timestamp] = plot_item

        # Notify views
        insert_row = len(self._cycle_data) - 1
        self.beginInsertRows(QtCore.QModelIndex(), insert_row, insert_row)
        self.endInsertRows()

        # Emit signals
        self.itemAdded.emit(cycle_data)
        self.plotItemAdded.emit(plot_item)

        # Auto-set reference if this matches the reference timestamp
        if (
            cycle_data.cycle_timestamp == cycle_data.reference_timestamp
            and self._reference is None
        ):
            self.set_reference(cycle_data)

    def update(self, cycle_data: CycleData) -> None:
        """Update an existing cycle in the model."""
        idx = self._get_cycle_index(cycle_data)
        if idx == -1:
            log.error(f"[{cycle_data}] not found in the list. Cannot update entry.")
            return

        # Update data
        self._cycle_data[idx] = cycle_data

        # Update plot metadata
        plot_item = self._plot_metadata[cycle_data.cycle_timestamp]
        plot_item.cycle_data = cycle_data

        # Notify views
        model_index = self.index(self._calc_display_row(idx), 0)
        self.dataChanged.emit(model_index, model_index, [QtCore.Qt.DisplayRole])

        # Emit signals
        self.itemUpdated.emit(cycle_data)
        self.plotItemUpdated.emit(plot_item)

        log.debug(f"[{cycle_data}] Updated in history.")

    def clear(self) -> None:
        """Clear all data from the model."""
        self.beginResetModel()

        # Emit removal signals for all items
        for cycle_data in self._cycle_data:
            self.itemRemoved.emit(cycle_data)
            plot_item = self._plot_metadata[cycle_data.cycle_timestamp]
            self.plotItemRemoved.emit(plot_item)

        self._cycle_data.clear()
        self._plot_metadata.clear()
        self._reference = None

        self.endResetModel()

    def set_max_len(self, max_len: int) -> None:
        """Set the maximum length of the cycle history."""
        if max_len < 1:
            msg = "Max length must be greater than 1."
            raise ValueError(msg)

        old_maxlen = self._cycle_data.maxlen
        if old_maxlen is None:
            old_maxlen = len(self._cycle_data)

        if max_len < old_maxlen and len(self._cycle_data) > max_len:
            # Remove excess items
            to_remove = list(self._cycle_data)[max_len:]
            self._cycle_data = deque(list(self._cycle_data)[:max_len], maxlen=max_len)

            for cycle_data in to_remove:
                plot_item = self._plot_metadata.pop(cycle_data.cycle_timestamp)
                self.itemRemoved.emit(cycle_data)
                self.plotItemRemoved.emit(plot_item)
        else:
            self._cycle_data = deque(self._cycle_data, maxlen=max_len)

    def get_max_len(self) -> int:
        """Get the maximum length of the cycle history."""
        maxlen = self._cycle_data.maxlen
        return maxlen if maxlen is not None else len(self._cycle_data)

    max_len = property(get_max_len, set_max_len)

    @QtCore.Slot(QtCore.QModelIndex)
    def clicked(self, index: QtCore.QModelIndex) -> None:
        """Handle item click - toggle plot visibility."""
        plot_item = self.get_plot_item_at(index)

        # Toggle visibility state
        plot_item.is_shown = not plot_item.is_shown

        # Update background color
        self.dataChanged.emit(index, index, [QtCore.Qt.BackgroundRole])

        # Let the plot model handle the actual show/hide
        if plot_item.is_shown:
            self.plotItemAdded.emit(plot_item)
        else:
            self.plotItemRemoved.emit(plot_item)

    def _remove_oldest(self, *, keep_reference: bool = True) -> CycleData:
        """Remove the oldest item, optionally preserving reference."""
        if not self._cycle_data:
            msg = "No items to remove."
            raise IndexError(msg)

        # Check if we should preserve the reference
        if (
            keep_reference
            and self._reference is not None
            and self._reference.cycle_timestamp == self._cycle_data[0].cycle_timestamp
        ):
            log.debug(
                f"[{self._cycle_data[0]}] Reference is first item. Removing second item."
            )
            if len(self._cycle_data) > 1:
                removed_data = self._cycle_data[1]
                del self._cycle_data[1]
                display_row = self._calc_display_row(1)
            else:
                # Only one item and it's the reference - can't remove
                return self._cycle_data[0]
        else:
            log.debug(f"[{self._cycle_data[0]}] Removed from history.")
            removed_data = self._cycle_data.popleft()
            display_row = self._calc_display_row(0)

        # Clean up plot metadata
        plot_item = self._plot_metadata.pop(removed_data.cycle_timestamp)

        # Notify views
        self.beginRemoveRows(QtCore.QModelIndex(), display_row, display_row)
        self.endRemoveRows()

        # Emit signals
        self.itemRemoved.emit(removed_data)
        self.plotItemRemoved.emit(plot_item)

        return removed_data

    def _get_cycle_index(self, cycle_data: CycleData) -> int:
        """Get the internal index of a cycle by timestamp."""
        for idx, item in enumerate(self._cycle_data):
            if item.cycle_timestamp == cycle_data.cycle_timestamp:
                return idx
        return -1

    def _calc_real_row(self, display_row: int) -> int:
        """Convert display row (reversed) to internal storage row."""
        if display_row >= len(self._cycle_data) or display_row < 0:
            msg = "Index out of range."
            raise IndexError(msg)
        return len(self._cycle_data) - display_row - 1

    def _calc_display_row(self, real_row: int) -> int:
        """Convert internal storage row to display row (reversed)."""
        return len(self._cycle_data) - real_row - 1

    @property
    def buffered_data(self) -> list[CycleData]:
        """Get all cycle data as a list."""
        return list(self._cycle_data)

    @property
    def plot_items(self) -> list[PlotItem]:
        """Get all plot items."""
        return list(self._plot_metadata.values())
