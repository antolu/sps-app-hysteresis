"""
Plot data adapter for converting cycle data to plot-ready format.

This adapter handles the transformation from raw CycleData to plot curves
without duplicating data storage, providing a clean separation between
data management and plot visualization.
"""

from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtGui

if TYPE_CHECKING:
    from ._dataclass import PlotItem

log = logging.getLogger(__package__)


class PlotType(enum.Enum):
    """Types of plots that can be generated."""

    MEASURED_CURRENT = enum.auto()
    MEASURED_FIELD = enum.auto()
    PREDICTED_FIELD = enum.auto()
    DELTA_FIELD = enum.auto()
    REF_MEASURED_DIFF = enum.auto()
    REF_PREDICTED_DIFF = enum.auto()


class PlotDataAdapter:
    """
    Adapter for converting CycleData to plot curves.

    This class handles all the data transformation logic for creating
    pyqtgraph plot items from cycle data, without storing the data itself.
    """

    def __init__(self) -> None:
        self._color_pool = ColorPool()

    def create_plot_curves(
        self, plot_item: PlotItem
    ) -> dict[PlotType, pg.PlotCurveItem]:
        """
        Create all possible plot curves for a PlotItem.

        Args:
            plot_item: The PlotItem containing cycle data and metadata

        Returns:
            Dictionary mapping plot types to curve items
        """
        curves = {}
        cycle_data = plot_item.cycle_data

        # Ensure item has a color
        if plot_item.color is None:
            plot_item.color = self._color_pool.get_color()

        # Determine line width (reference items are thicker)
        width = 4 if self._is_reference(plot_item) else 2

        # Create each type of curve if data is available
        if cycle_data.current_meas is not None:
            curves[PlotType.MEASURED_CURRENT] = self._create_measured_current_curve(
                plot_item, width
            )

        if cycle_data.field_meas is not None:
            curves[PlotType.MEASURED_FIELD] = self._create_measured_field_curve(
                plot_item, width
            )

        if cycle_data.field_pred is not None:
            curves[PlotType.PREDICTED_FIELD] = self._create_predicted_field_curve(
                plot_item, width
            )

        if cycle_data.delta_applied is not None:
            curves[PlotType.DELTA_FIELD] = self._create_delta_curve(plot_item, width)

        if cycle_data.field_meas_ref is not None and cycle_data.field_meas is not None:
            curves[PlotType.REF_MEASURED_DIFF] = self._create_ref_measured_diff_curve(
                plot_item, width
            )

        if cycle_data.field_ref is not None and cycle_data.field_pred is not None:
            curves[PlotType.REF_PREDICTED_DIFF] = self._create_ref_predicted_diff_curve(
                plot_item, width
            )

        return curves

    def update_curve_width(self, curve: pg.PlotCurveItem, width: int) -> None:
        """Update the width of an existing curve."""
        pen = curve.opts.get("pen", pg.mkPen())
        color = pen.color() if hasattr(pen, "color") else pen
        curve.setPen(width=width, color=color)

    def return_color(self, color: QtGui.QColor) -> None:
        """Return a color to the pool for reuse."""
        self._color_pool.return_color(color)

    def _create_measured_current_curve(
        self, plot_item: PlotItem, width: int
    ) -> pg.PlotCurveItem:
        """Create curve for measured current data."""
        cycle_data = plot_item.cycle_data
        x, y = self._make_measurement_curve(cycle_data, cycle_data.current_meas)
        return self._make_curve_item(x, y, plot_item.color, width)

    def _create_measured_field_curve(
        self, plot_item: PlotItem, width: int
    ) -> pg.PlotCurveItem:
        """Create curve for measured field data."""
        cycle_data = plot_item.cycle_data
        x, y = self._make_measurement_curve(cycle_data, cycle_data.field_meas)
        return self._make_curve_item(x, y, plot_item.color, width)

    def _create_predicted_field_curve(
        self, plot_item: PlotItem, width: int
    ) -> pg.PlotCurveItem:
        """Create curve for predicted field data."""
        x, y = self._make_prediction_curve(plot_item, use_reference=False)
        return self._make_curve_item(x, y, plot_item.color, width)

    def _create_delta_curve(self, plot_item: PlotItem, width: int) -> pg.PlotCurveItem:
        """Create curve for delta (trim) data."""
        cycle_data = plot_item.cycle_data
        x = cycle_data.delta_applied[0]
        y = cycle_data.delta_applied[1] * 1e4  # Convert to 1e-4 T units
        return self._make_curve_item(x, y, plot_item.color, width)

    def _create_ref_measured_diff_curve(
        self, plot_item: PlotItem, width: int
    ) -> pg.PlotCurveItem:
        """Create curve for measured field difference from reference."""
        cycle_data = plot_item.cycle_data
        diff = (cycle_data.field_meas_ref - cycle_data.field_meas) * 1e4
        x, y = self._make_measurement_curve(cycle_data, diff)
        return self._make_curve_item(x, y, plot_item.color, width)

    def _create_ref_predicted_diff_curve(
        self, plot_item: PlotItem, width: int
    ) -> pg.PlotCurveItem:
        """Create curve for predicted field difference from reference."""
        ref_x, ref_y = self._make_prediction_curve(plot_item, use_reference=True)
        _, pred_y = self._make_prediction_curve(plot_item, use_reference=False)

        # Calculate difference in 1e-4 T units
        diff_y = (ref_y - pred_y) * 1e4
        return self._make_curve_item(ref_x, diff_y, plot_item.color, width)

    def _make_measurement_curve(
        self, cycle_data: CycleData, y_data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create x,y data for measurement plots."""
        x = self._make_time_axis(cycle_data)
        y = y_data.flatten()
        return x, y

    def _make_prediction_curve(
        self, plot_item: PlotItem, *, use_reference: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create x,y data for prediction plots."""
        cycle_data = plot_item.cycle_data

        if use_reference:
            if cycle_data.field_ref is None:
                msg = f"[{cycle_data}] No reference field data found."
                raise ValueError(msg)
            field_data = cycle_data.field_ref
        else:
            if cycle_data.field_pred is None:
                msg = f"[{cycle_data}] No field prediction found."
                raise ValueError(msg)
            field_data = cycle_data.field_pred

        time_axis = self._make_time_axis(cycle_data)
        field_values = field_data[1, :]

        # Handle downsampling if needed
        x = time_axis[:: self._calc_downsample(time_axis, field_values)]
        field_interp = np.interp(time_axis, x, field_values)

        return time_axis, field_interp

    def _make_time_axis(self, cycle_data: CycleData) -> np.ndarray:
        """Create time axis for a cycle."""
        x = np.arange(0, cycle_data.num_samples + 1)

        # Handle edge case for specific lengths
        if str(len(x)).endswith("1"):
            x = x[:-1]

        return x

    def _calc_downsample(self, high: np.ndarray, low: np.ndarray) -> int:
        """Calculate downsampling factor."""
        return int(np.ceil(len(high) / len(low)))

    def _make_curve_item(
        self, x: np.ndarray, y: np.ndarray, color: QtGui.QColor, width: int = 2
    ) -> pg.PlotCurveItem:
        """Create a pyqtgraph curve item."""
        return pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color=color, width=width))

    def _is_reference(self, plot_item: PlotItem) -> bool:
        """Check if this plot item represents the reference cycle."""
        cycle_data = plot_item.cycle_data
        return cycle_data.cycle_timestamp == cycle_data.reference_timestamp


class ColorPool:
    """
    Simple color pool for managing plot colors.

    This is a simplified version that could be imported from utils
    if it already exists in the codebase.
    """

    def __init__(self) -> None:
        self._available_colors = [
            QtGui.QColor("#1f77b4"),  # Blue
            QtGui.QColor("#ff7f0e"),  # Orange
            QtGui.QColor("#2ca02c"),  # Green
            QtGui.QColor("#d62728"),  # Red
            QtGui.QColor("#9467bd"),  # Purple
            QtGui.QColor("#8c564b"),  # Brown
            QtGui.QColor("#e377c2"),  # Pink
            QtGui.QColor("#7f7f7f"),  # Gray
            QtGui.QColor("#bcbd22"),  # Olive
            QtGui.QColor("#17becf"),  # Cyan
        ]
        self._used_colors: set[QtGui.QColor] = set()

    def get_color(self) -> QtGui.QColor:
        """Get an available color from the pool."""
        for color in self._available_colors:
            if color not in self._used_colors:
                self._used_colors.add(color)
                return color

        # If all colors are used, cycle through them
        return self._available_colors[
            len(self._used_colors) % len(self._available_colors)
        ]

    def return_color(self, color: QtGui.QColor) -> None:
        """Return a color to the pool."""
        self._used_colors.discard(color)
