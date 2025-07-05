"""
Refactored plot model using the adapter pattern.

This simplified plot model focuses purely on managing plot visibility
and communicating with the plot widgets, while delegating data transformation
to the PlotDataAdapter.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pyqtgraph as pg
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore

from ._dataclass import PlotItem
from ._plot_adapter import PlotDataAdapter, PlotType

if TYPE_CHECKING:
    from ._unified_model import CycleListModel

log = logging.getLogger(__package__)


class UnifiedPlotModel(QtCore.QObject):
    """
    Plot model that manages plot visibility using the adapter pattern.

    This model:
    - Uses adapter for data transformations
    - Focuses on plot management
    - Has clear separation of concerns
    """

    # Plot addition signals
    measuredCurrentAdded = QtCore.Signal(pg.PlotCurveItem)
    measuredFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    predictedFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    deltaFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    refMeasuredFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    refPredictedFieldAdded = QtCore.Signal(pg.PlotCurveItem)

    # Plot removal signals
    measuredCurrentRemoved = QtCore.Signal(pg.PlotCurveItem)
    measuredFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    predictedFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    deltaFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    refMeasuredFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    refPredictedFieldRemoved = QtCore.Signal(pg.PlotCurveItem)

    # Axis control signals
    setXRange = QtCore.Signal(float, float)
    setYRange = QtCore.Signal(float, float)

    # Reference signals
    newReference = QtCore.Signal(PlotItem)

    def __init__(
        self, cycle_model: CycleListModel, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self._cycle_model = cycle_model
        self._adapter = PlotDataAdapter()

        # Track which items are currently plotted
        self._plotted_items: set[PlotItem] = set()
        self._current_reference: PlotItem | None = None

        # Connect to the unified model
        self._connect_model_signals()

    def _connect_model_signals(self) -> None:
        """Connect to signals from the unified cycle model."""
        self._cycle_model.plotItemAdded.connect(self.show_plot_item)
        self._cycle_model.plotItemUpdated.connect(self.update_plot_item)
        self._cycle_model.plotItemRemoved.connect(self.hide_plot_item)
        self._cycle_model.referenceChanged.connect(self.set_reference)

    @QtCore.Slot(PlotItem)
    def show_plot_item(self, plot_item: PlotItem) -> None:
        """Show plots for the given item."""
        if plot_item in self._plotted_items:
            log.debug(f"[{plot_item.cycle_data}] Already shown, skipping.")
            return

        log.debug(f"[{plot_item.cycle_data}] Creating plots.")

        try:
            # Generate all possible curves for this item
            curves = self._adapter.create_plot_curves(plot_item)

            # Store curves in the plot item for later removal
            self._store_curves_in_item(plot_item, curves)

            # Emit signals to add curves to appropriate plots
            self._emit_addition_signals(curves)

            # Track this item as plotted
            self._plotted_items.add(plot_item)
            plot_item.is_shown = True

            # Update axes if this is the first item
            if len(self._plotted_items) == 1:
                self.reset_axes()

        except Exception:
            log.exception(f"Failed to create plots for {plot_item.cycle_data}")

    @QtCore.Slot(PlotItem)
    def update_plot_item(self, plot_item: PlotItem) -> None:
        """Update plots for an existing item."""
        if plot_item not in self._plotted_items:
            log.debug(f"[{plot_item.cycle_data}] Not shown, skipping update.")
            return

        log.debug(f"[{plot_item.cycle_data}] Updating plots.")

        # Remove old plots and create new ones
        self.hide_plot_item(plot_item)
        self.show_plot_item(plot_item)

    @QtCore.Slot(PlotItem)
    def hide_plot_item(self, plot_item: PlotItem) -> None:
        """Hide plots for the given item."""
        if plot_item not in self._plotted_items:
            return

        log.debug(f"[{plot_item.cycle_data}] Removing plots.")

        # Remove all curves from plots
        self._remove_all_curves(plot_item)

        # Return color to pool
        if plot_item.color is not None:
            self._adapter.return_color(plot_item.color)
            plot_item.color = None

        # Update tracking
        self._plotted_items.remove(plot_item)
        plot_item.is_shown = False

    @QtCore.Slot()
    def hide_all(self) -> None:
        """Hide all currently shown plots."""
        for plot_item in self._plotted_items.copy():
            self.hide_plot_item(plot_item)

    @QtCore.Slot(object)  # CycleData
    def set_reference(self, cycle_data: CycleData) -> None:
        """Set the reference cycle and update plot widths."""
        # Find the plot item for this cycle
        plot_item = None
        for item in self._plotted_items:
            if item.cycle_data.cycle_timestamp == cycle_data.cycle_timestamp:
                plot_item = item
                break

        if plot_item is None:
            log.debug(f"Reference cycle {cycle_data.cycle} not currently plotted.")
            return

        old_reference = self._current_reference
        self._current_reference = plot_item

        # Update line widths
        self._update_reference_widths(old_reference, plot_item)

        # Emit signal
        self.newReference.emit(plot_item)

        log.debug(f"[{cycle_data}] Set as plotting reference.")

    @QtCore.Slot()
    def reset_axes(self) -> None:
        """Reset plot axes based on currently shown data."""
        if not self._plotted_items:
            return

        # Calculate appropriate ranges
        x_min, x_max = self._calculate_x_range()
        y_min, y_max = self._calculate_y_range()

        self.setXRange.emit(x_min, x_max)
        self.setYRange.emit(y_min, y_max)

    def _store_curves_in_item(
        self, plot_item: PlotItem, curves: dict[PlotType, pg.PlotCurveItem]
    ) -> None:
        """Store curve references in the plot item for later removal."""
        plot_item.raw_current_plt = curves.get(PlotType.MEASURED_CURRENT)
        plot_item.raw_meas_plt = curves.get(PlotType.MEASURED_FIELD)
        plot_item.raw_pred_plt = curves.get(PlotType.PREDICTED_FIELD)
        plot_item.delta_plt = curves.get(PlotType.DELTA_FIELD)
        plot_item.ref_meas_plt = curves.get(PlotType.REF_MEASURED_DIFF)
        plot_item.ref_pred_plt = curves.get(PlotType.REF_PREDICTED_DIFF)

    def _emit_addition_signals(self, curves: dict[PlotType, pg.PlotCurveItem]) -> None:
        """Emit appropriate signals to add curves to plots."""
        signal_map = {
            PlotType.MEASURED_CURRENT: self.measuredCurrentAdded,
            PlotType.MEASURED_FIELD: self.measuredFieldAdded,
            PlotType.PREDICTED_FIELD: self.predictedFieldAdded,
            PlotType.DELTA_FIELD: self.deltaFieldAdded,
            PlotType.REF_MEASURED_DIFF: self.refMeasuredFieldAdded,
            PlotType.REF_PREDICTED_DIFF: self.refPredictedFieldAdded,
        }

        for plot_type, curve in curves.items():
            if plot_type in signal_map:
                signal_map[plot_type].emit(curve)

    def _remove_all_curves(self, plot_item: PlotItem) -> None:
        """Remove all curves for a plot item from the plots."""
        curve_signal_map = [
            (plot_item.raw_current_plt, self.measuredCurrentRemoved),
            (plot_item.raw_meas_plt, self.measuredFieldRemoved),
            (plot_item.raw_pred_plt, self.predictedFieldRemoved),
            (plot_item.delta_plt, self.deltaFieldRemoved),
            (plot_item.ref_meas_plt, self.refMeasuredFieldRemoved),
            (plot_item.ref_pred_plt, self.refPredictedFieldRemoved),
        ]

        for curve, signal in curve_signal_map:
            if curve is not None:
                signal.emit(curve)

        # Clear curve references
        plot_item.raw_current_plt = None
        plot_item.raw_meas_plt = None
        plot_item.raw_pred_plt = None
        plot_item.delta_plt = None
        plot_item.ref_meas_plt = None
        plot_item.ref_pred_plt = None

    def _update_reference_widths(
        self, old_reference: PlotItem | None, new_reference: PlotItem
    ) -> None:
        """Update line widths when reference changes."""
        # Make new reference curves thicker
        for curve in self._get_all_curves(new_reference):
            if curve is not None:
                self._adapter.update_curve_width(curve, 4)

        # Make old reference curves normal width
        if old_reference is not None:
            for curve in self._get_all_curves(old_reference):
                if curve is not None:
                    self._adapter.update_curve_width(curve, 2)

    def _get_all_curves(self, plot_item: PlotItem) -> list[pg.PlotCurveItem | None]:
        """Get all curve references from a plot item."""
        return [
            plot_item.raw_current_plt,
            plot_item.raw_meas_plt,
            plot_item.raw_pred_plt,
            plot_item.delta_plt,
            plot_item.ref_meas_plt,
            plot_item.ref_pred_plt,
        ]

    def _calculate_x_range(self) -> tuple[float, float]:
        """Calculate appropriate X range for all plotted items."""
        if not self._plotted_items:
            return 0.0, 1.0

        # Use the number of samples from any plotted item
        first_item = next(iter(self._plotted_items))
        num_samples = first_item.cycle_data.num_samples
        return 0.0, float(num_samples)

    def _calculate_y_range(self) -> tuple[float, float]:
        """Calculate appropriate Y range for field plots."""
        if not self._plotted_items:
            return 0.0, 2.1

        y_values = []
        for item in self._plotted_items:
            if item.cycle_data.field_pred is not None:
                field_data = item.cycle_data.field_pred[1, :]
                y_values.extend([field_data.min(), field_data.max()])

        if not y_values:
            return 0.0, 2.1

        y_min, y_max = min(y_values), max(y_values)
        margin = (y_max - y_min) * 0.1  # 10% margin
        return y_min - margin, y_max + margin


# Alternative plot model implementation
class PredictionPlotModel(QtCore.QObject):
    """
    Plot model for prediction visualization.

    This provides the standard API for managing prediction plots.
    """

    # Same signals as the original for compatibility
    newReference = QtCore.Signal(PlotItem)
    measuredCurrentAdded = QtCore.Signal(pg.PlotCurveItem)
    measuredFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    predictedFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    deltaFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    refMeasuredFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    refPredictedFieldAdded = QtCore.Signal(pg.PlotCurveItem)
    measuredCurrentRemoved = QtCore.Signal(pg.PlotCurveItem)
    measuredFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    predictedFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    deltaFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    refMeasuredFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    refPredictedFieldRemoved = QtCore.Signal(pg.PlotCurveItem)
    setXRange = QtCore.Signal(float, float)
    setYRange = QtCore.Signal(float, float)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

        # Implementation using adapter pattern
        self._plotted_items: set[PlotItem] = set()
        self._reference: PlotItem | None = None
        self._adapter = PlotDataAdapter()

    @QtCore.Slot(PlotItem)
    def showCycle(self, item: PlotItem) -> None:
        """Show a cycle."""
        if item in self._plotted_items:
            return

        try:
            curves = self._adapter.create_plot_curves(item)
            self._store_curves_in_item(item, curves)
            self._emit_addition_signals(curves)
            self._plotted_items.add(item)
            item.is_shown = True
        except Exception:
            log.exception(f"Failed to create plots for {item.cycle_data}")

    @QtCore.Slot(PlotItem)
    def updateCycle(self, item: PlotItem) -> None:
        """Update a cycle."""
        if item not in self._plotted_items:
            return
        self.removeCycle(item)
        self.showCycle(item)

    @QtCore.Slot(PlotItem)
    def removeCycle(self, item: PlotItem) -> None:
        """Remove a cycle."""
        if item not in self._plotted_items:
            return

        self._remove_all_curves(item)
        if item.color is not None:
            self._adapter.return_color(item.color)
            item.color = None
        self._plotted_items.remove(item)
        item.is_shown = False

    @QtCore.Slot()
    def removeAll(self) -> None:
        """Remove all cycles."""
        for item in self._plotted_items.copy():
            self.removeCycle(item)

    def _store_curves_in_item(
        self, plot_item: PlotItem, curves: dict[PlotType, pg.PlotCurveItem]
    ) -> None:
        """Store curve references in the plot item for later removal."""
        plot_item.raw_current_plt = curves.get(PlotType.MEASURED_CURRENT)
        plot_item.raw_meas_plt = curves.get(PlotType.MEASURED_FIELD)
        plot_item.raw_pred_plt = curves.get(PlotType.PREDICTED_FIELD)
        plot_item.delta_plt = curves.get(PlotType.DELTA_FIELD)
        plot_item.ref_meas_plt = curves.get(PlotType.REF_MEASURED_DIFF)
        plot_item.ref_pred_plt = curves.get(PlotType.REF_PREDICTED_DIFF)

    def _emit_addition_signals(self, curves: dict[PlotType, pg.PlotCurveItem]) -> None:
        """Emit appropriate signals to add curves to plots."""
        signal_map = {
            PlotType.MEASURED_CURRENT: self.measuredCurrentAdded,
            PlotType.MEASURED_FIELD: self.measuredFieldAdded,
            PlotType.PREDICTED_FIELD: self.predictedFieldAdded,
            PlotType.DELTA_FIELD: self.deltaFieldAdded,
            PlotType.REF_MEASURED_DIFF: self.refMeasuredFieldAdded,
            PlotType.REF_PREDICTED_DIFF: self.refPredictedFieldAdded,
        }

        for plot_type, curve in curves.items():
            if plot_type in signal_map:
                signal_map[plot_type].emit(curve)

    def _remove_all_curves(self, plot_item: PlotItem) -> None:
        """Remove all curves for a plot item from the plots."""
        curve_signal_map = [
            (plot_item.raw_current_plt, self.measuredCurrentRemoved),
            (plot_item.raw_meas_plt, self.measuredFieldRemoved),
            (plot_item.raw_pred_plt, self.predictedFieldRemoved),
            (plot_item.delta_plt, self.deltaFieldRemoved),
            (plot_item.ref_meas_plt, self.refMeasuredFieldRemoved),
            (plot_item.ref_pred_plt, self.refPredictedFieldRemoved),
        ]

        for curve, signal in curve_signal_map:
            if curve is not None:
                signal.emit(curve)

        # Clear curve references
        plot_item.raw_current_plt = None
        plot_item.raw_meas_plt = None
        plot_item.raw_pred_plt = None
        plot_item.delta_plt = None
        plot_item.ref_meas_plt = None
        plot_item.ref_pred_plt = None
