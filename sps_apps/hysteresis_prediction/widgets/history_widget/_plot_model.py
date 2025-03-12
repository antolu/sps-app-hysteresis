from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore, QtGui

from ...utils import ColorPool
from ._dataclass import PlotItem

log = logging.getLogger(__package__)


class PredictionPlotModel(QtCore.QObject):
    newReference = QtCore.Signal(PlotItem)
    """ Triggered when a new reference is set """

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
    """ Set X range of all plots """
    setYRange = QtCore.Signal(float, float)
    """ Set Y range of field plots (not current) """

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent=parent)

        self._plotted_items: set[PlotItem] = set()

        self._reference: PlotItem | None = None
        self._color_pool = ColorPool()

    @QtCore.Slot(PlotItem)
    def showCycle(self, item: PlotItem) -> None:
        """
        Add a *new* item to the plot.

        The item is sent to the plotAdded signal if it is successfully plotted.

        If the reference is not set, the main plot will not be created,
        and left as None. It will be automatically updated by the
        :meth:`set_reference`, and :meth:`update_item` methods.

        :param item: The item to add to the plot.
        """
        if item.color is None:
            color = self._color_pool.get_color()
            item.color = color

        width = 4 if item is self._reference else 2

        if item.cycle_data.current_meas is not None and item.raw_current_plt is None:
            item.raw_current_plt = _make_curve_item(
                *_make_meas_curve(item.cycle_data, item.cycle_data.current_meas),
                item.color,
                width=width,
            )
            log.debug(f"[{item.cycle_data}] Creating current plot.")
            self.measuredCurrentAdded.emit(item.raw_current_plt)
        elif item.cycle_data.current_meas is None:
            log.debug(f"[{item.cycle_data}] No current data, skipping.")

        if item.cycle_data.field_meas is not None and item.raw_meas_plt is None:
            item.raw_meas_plt = _make_curve_item(
                *_make_meas_curve(item.cycle_data, item.cycle_data.field_meas),
                item.color,
                width=width,
            )
            log.debug(f"[{item.cycle_data}] Creating field plot.")
            self.measuredFieldAdded.emit(item.raw_meas_plt)
        elif item.cycle_data.field_meas is None:
            log.debug(f"[{item.cycle_data}] No meas data, skipping.")

        if item.cycle_data.field_pred is not None and item.raw_pred_plt is None:
            item.raw_pred_plt = _make_curve_item(*_make_pred_curve(item), item.color)
            self.predictedFieldAdded.emit(item.raw_pred_plt)
        elif item.cycle_data.field_pred is None:
            log.debug(f"[{item.cycle_data}] No pred data, skipping.")

        if item.cycle_data.delta_applied is not None and item.delta_plt is None:
            item.delta_plt = _make_curve_item(
                item.cycle_data.delta_applied[0],
                item.cycle_data.delta_applied[1] * 1e4,
                item.color,
                width=width,
            )
            log.debug(f"[{item.cycle_data}] Creating delta plot.")
            self.deltaFieldAdded.emit(item.delta_plt)

        if (
            item.ref_meas_plt is None
            and item.cycle_data.field_meas_ref is not None
            and item.cycle_data.field_meas is not None
        ):
            item.ref_meas_plt = _make_curve_item(
                *_make_meas_curve(
                    item.cycle_data,
                    (item.cycle_data.field_meas_ref - item.cycle_data.field_meas) * 1e4,
                ),
                item.color,
                width=width,
            )

            log.debug(f"[{item.cycle_data}] Creating ref meas plot.")
            self.refMeasuredFieldAdded.emit(item.ref_meas_plt)
        elif item.cycle_data.field_meas_ref is None:
            log.debug(f"[{item.cycle_data}] No ref meas data, skipping.")

        if (
            item.ref_pred_plt is None
            and item.cycle_data.field_ref is not None
            and item.cycle_data.field_pred is not None
        ):
            ref_x, ref_y = _make_pred_curve(item)
            _, pred_y = _make_pred_curve(item)

            item.ref_pred_plt = _make_curve_item(
                ref_x, (ref_y - pred_y) * 1e4, item.color, width=width
            )

            log.debug(f"[{item.cycle_data}] Creating ref pred plot.")
            self.refPredictedFieldAdded.emit(item.ref_pred_plt)
        elif item.cycle_data.field_ref is None:
            log.debug(f"[{item.cycle_data}] No ref pred data, skipping.")

        self._plotted_items.add(item)

        item.is_shown = True
        if not item.is_shown and len(self._plotted_items) == 1:
            # first item, set axes
            # self.setReference(item)
            self.resetAxes()

    def updateCycle(self, item: PlotItem) -> None:
        if not item.is_shown:
            log.debug(f"Item {item} not shown, not updating.")
            return

        log.debug(f"[{item.cycle_data}] Updating plots with.")
        self.showCycle(item)

    @QtCore.Slot(PlotItem)
    def removeCycle(self, item: PlotItem) -> None:
        """
        Remove an existing item from the plot.

        The item is sent to the plotRemoved Signal if it was plotted,
        and to the plotRemoved_dpp Signal. The plot is then removed
        from the index, and the color returned to the pool.

        The plot change is communicated to the plot widget with the
        plotRemoved signals.
        """
        if item not in self._plotted_items:
            return

        log.debug(f"Removing plot for cycle {item.cycle_data.cycle_time}")

        if item.ref_pred_plt is not None:
            log.debug(f"[{item.cycle_data}] Removing ref pred plot.")
            self.refPredictedFieldRemoved.emit(item.ref_pred_plt)
            item.ref_pred_plt = None

        if item.raw_pred_plt is not None:
            log.debug(f"[{item.cycle_data}] Removing pred plot.")
            self.predictedFieldRemoved.emit(item.raw_pred_plt)
            item.raw_pred_plt = None

        if item.raw_meas_plt is not None:
            log.debug(f"[{item.cycle_data}] Removing meas plot.")
            self.measuredFieldRemoved.emit(item.raw_meas_plt)
            item.raw_meas_plt = None

        if item.raw_current_plt is not None:
            log.debug(f"[{item.cycle_data}] Removing current plot.")
            self.measuredCurrentRemoved.emit(item.raw_current_plt)
            item.raw_current_plt = None

        if item.delta_plt is not None:
            log.debug(f"[{item.cycle_data}] Removing delta plot.")
            self.deltaFieldRemoved.emit(item.delta_plt)
            item.delta_plt = None

        if item.ref_meas_plt is not None:
            log.debug(f"[{item.cycle_data}] Removing ref meas plot.")
            self.refMeasuredFieldRemoved.emit(item.ref_meas_plt)
            item.ref_meas_plt = None

        if item.color is not None:
            log.debug(f"Returning color {item.color} to pool.")
            self._color_pool.return_color(item.color)
            item.color = None

        self._plotted_items.remove(item)
        item.is_shown = False

    @QtCore.Slot()
    def removeAll(self) -> None:
        """
        Trigger a full removal of all plots.
        """
        for item in self._plotted_items.copy():
            self.removeCycle(item)

    @QtCore.Slot(PlotItem)
    def setReference(self, item: PlotItem) -> None:
        """
        Set the reference item. Makes the reference curve wider, and plots it if not shown.

        :param item: The item to use as reference.
        """
        log.debug(f"[{item.cycle_data}] Setting plotting reference.")
        current_reference = self._reference

        if not item.is_shown:
            log.debug(f"New reference {item.cycle_data} not shown, plotting now.")
            self.showCycle(item)

        if current_reference is item:
            return

        self._reference = item
        log.debug(f"Updating reference w.r.t. {item.cycle_data}")

        # make reference curve wider
        for attr in (
            "ref_pred_plt",
            "ref_meas_plt",
            "delta_plt",
            "raw_pred_plt",
            "raw_meas_plt",
            "raw_current_plt",
        ):
            if getattr(item, attr) is not None:
                self.setCurveWidth(getattr(item, attr), 4)

            if (
                current_reference is not None
                and getattr(current_reference, attr) is not None
            ):
                self.setCurveWidth(getattr(current_reference, attr), 2)

        self.newReference.emit(item)

    @staticmethod
    def setCurveWidth(pg_curve: pg.PlotCurveItem, width: int) -> None:
        """
        Set the width of a curve.

        :param pg_curve: The curve to set the width of.
        :param width: The width to set.
        """
        pg_curve.setPen(width=width, color=pg_curve.opts["pen"].color())

    @QtCore.Slot()
    def resetAxes(self) -> None:
        if len(self._plotted_items) == 0:
            return

        max_val = max(
            item.cycle_data.field_pred[1, :].max()
            for item in self._plotted_items
            if item.cycle_data.field_pred is not None
        )
        min_val = min(
            item.cycle_data.field_pred[1, :].min()
            for item in self._plotted_items
            if item.cycle_data.field_pred is not None
        )

        self.setYRange.emit(min_val - 0.1, max_val + 0.1)
        self.setXRange.emit(0, next(iter(self._plotted_items)).cycle_data.num_samples)


def calc_downsample(high: np.ndarray, low: np.ndarray) -> int:
    return int(np.ceil(len(high) / len(low)))


def _make_pred_curve(item: PlotItem) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a curve for the dp/p plot.

    :param item: The item to create the curve for.
    """
    data = item.cycle_data
    assert data.field_pred is not None

    time_axis = _make_time_axis(item)
    field_pred = data.field_pred[1, :]
    x = time_axis[:: calc_downsample(time_axis, data.field_pred[1, :])]

    field_pred = np.interp(time_axis, x, field_pred)

    return time_axis, field_pred


def _make_curve_item(
    x: np.ndarray, y: np.ndarray, color: QtGui.QColor, width: int = 2
) -> pg.PlotCurveItem:
    """
    Create a new curve item.

    :param x: The x data.
    :param y: The y data.
    :param color: The color of the curve.
    """
    return pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color=color, width=width))


def _make_meas_curve(item: PlotItem, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _make_time_axis(item)
    y = y.flatten()

    return x, y


def _update_curve(x: np.ndarray, y: np.ndarray, curve: pg.PlotCurveItem) -> None:
    """
    Update the data of a curve.

    :param x: The x data.
    :param y: The y data.
    :param curve: The curve to update.
    """
    curve.setData(x=x, y=y)


def _make_time_axis(item: PlotItem | CycleData) -> np.ndarray:
    cycle = item.cycle_data if isinstance(item, PlotItem) else item
    x = np.arange(0, cycle.num_samples + 1)

    if str(len(x)).endswith("1"):
        x = x[:-1]

    return x
