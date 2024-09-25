from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore, QtGui

from ._colors import ColorPool
from ._dataclass import DiffPlotMode, MeasPlotMode, Plot, PredictionItem

log = logging.getLogger(__name__)


class PredictionPlotModel(QtCore.QObject):
    newReference = QtCore.Signal(PredictionItem)
    """ Triggered when a new reference is set """

    plotAdded = QtCore.Signal(pg.PlotCurveItem, Plot)
    """ Triggered when a new plot is added """

    plotRemoved = QtCore.Signal(pg.PlotCurveItem, Plot)
    """ Triggered when an existing plot is removed """

    zoomFlatTop = QtCore.Signal()
    zoomFlatBottom = QtCore.Signal()
    zoomBeamIn = QtCore.Signal()
    resetAxes = QtCore.Signal()

    setXRange = QtCore.Signal(float, float)
    setYRange = QtCore.Signal(float, float)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent=parent)

        self._plotted_items: set[PredictionItem] = set()

        self.newReference.connect(self.set_reference)
        self.zoomFlatTop.connect(self._zoom_flat_top)
        self.zoomFlatBottom.connect(self._zoom_flat_bottom)
        self.zoomBeamIn.connect(self._zoom_beam_in)
        self.resetAxes.connect(self._reset_axes)

        self._diff_plot_mode = DiffPlotMode.PredVsMeas
        self._meas_plot_mode = MeasPlotMode.RawMeas
        self._reference: PredictionItem | None = None
        self._color_pool = ColorPool()

        self.beam_in: int = 0
        self.beam_out: int = 0

    @QtCore.Slot(PredictionItem)
    def show_cycle(self, item: PredictionItem) -> None:
        """
        Add a *new* item to the plot.

        The item is sent to the plotAdded signal if it is successfully plotted.

        If the reference is not set, the main plot will not be created,
        and left as None. It will be automatically updated by the
        :meth:`set_reference`, and :meth:`update_item` methods.

        :param item: The item to add to the plot.
        """
        if item in self._plotted_items:
            log.warning(f"Item {item} already plotted.")
            return

        color = item.color or self._color_pool.get_color()
        item.color = color

        try:
            x, y = _make_diff_curve(item, self._reference, self._diff_plot_mode)
            item.diff_plot_item = _make_curve_item(x, y, color)
        except ValueError:  # missing reference
            log.debug("Reference not set, not plotting main plot.")
        try:
            item.pred_plot_item = _make_curve_item(*_make_pred_curve(item), color)

            assert item.cycle_data.field_meas is not None
            assert item.cycle_data.current_meas is not None
            item.meas_i_plot_item = _make_curve_item(
                *_make_meas_curve(
                    item, item.cycle_data.current_meas, self._meas_plot_mode
                ),
                color,
            )
            item.meas_b_plot_item = _make_curve_item(
                *_make_meas_curve(
                    item, item.cycle_data.field_meas, self._meas_plot_mode
                ),
                color,
            )
            item.delta_plot_item = _make_curve_item(
                item.cycle_data.delta_applied[0],
                item.cycle_data.delta_applied[1] * 1e4,
                color,
            )
            item.ref_diff_plot_item = _make_curve_item(
                *_make_diff_pred_vs_meas(item),
                color,
            )
        except Exception:
            log.exception(f"Failed to plot item {item}.")
            return

        if item.diff_plot_item is not None:
            self.plotAdded.emit(item.diff_plot_item, Plot.Diff)

        self.plotAdded.emit(item.pred_plot_item, Plot.Pred)
        self.plotAdded.emit(item.meas_i_plot_item, Plot.MeasI)
        self.plotAdded.emit(item.meas_b_plot_item, Plot.MeasB)
        self.plotAdded.emit(item.delta_plot_item, Plot.Delta)
        self.plotAdded.emit(item.ref_diff_plot_item, Plot.RefDiff)

        self._plotted_items.add(item)
        item.is_shown = True

    @QtCore.Slot(PredictionItem)
    def update_diff_plot(self, item: PredictionItem) -> None:
        """
        Update an existing item in the plot.

        The item is sent to the plotAdded Signal if it was not
        plotted before. Otherwise the change is handled in the
        individual plot items.

        If the reference is not set, the main plot will not be created.
        If a plot existed previously, it will remain unchanged (until
        the reference is set).

        :param item: The item to update.
        """
        log.debug(f"Updating plot for cycle {item.cycle_data.cycle_time}")

        if item not in self._plotted_items:
            log.warning(f"Item {item} not plotted.")
            return

        try:
            x, y = _make_diff_curve(item, self._reference, self._diff_plot_mode)

            # only add if not already plotted
            if item.diff_plot_item is None:
                item.diff_plot_item = pg.PlotCurveItem(
                    x=x,
                    y=y,
                    pen=pg.mkPen(color=item.color, width=1),
                )
                self.plotAdded.emit(item.diff_plot_item, Plot.Diff)
            else:
                _update_curve(x, y, item.diff_plot_item)
        except ValueError:  # missing reference
            log.exception("Reference not set, not updating main plot.")
            log.debug("Reference not set, not updating main plot.")

    @QtCore.Slot(PredictionItem)
    def update_meas_plot(self, item: PredictionItem) -> None:
        """
        Update an existing item in the plot.

        The item is sent to the plotAdded Signal if it was not
        plotted before. Otherwise the change is handled in the
        individual plot items.

        If the reference is not set, the main plot will not be created.
        If a plot existed previously, it will remain unchanged (until
        the reference is set).

        :param item: The item to update.
        """
        log.debug(f"Updating plot for cycle {item.cycle_data.cycle_time}")

        if item not in self._plotted_items:
            log.warning(f"Item {item} not plotted.")
            return

        assert item.cycle_data.field_meas is not None
        assert item.cycle_data.current_meas is not None
        x, y = _make_meas_curve(item, item.cycle_data.field_meas, self._meas_plot_mode)
        _update_curve(x, y, item.meas_b_plot_item)

        x, y = _make_meas_curve(
            item, item.cycle_data.current_meas, self._meas_plot_mode
        )
        _update_curve(x, y, item.meas_i_plot_item)

    @QtCore.Slot(PredictionItem)
    def remove_cycle(self, item: PredictionItem) -> None:
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

        if item.diff_plot_item is not None:
            self.plotRemoved.emit(item.diff_plot_item, Plot.Diff)
            item.diff_plot_item = None
        if item.pred_plot_item is not None:
            self.plotRemoved.emit(item.pred_plot_item, Plot.Pred)
            item.pred_plot_item = None
        if item.meas_i_plot_item is not None:
            self.plotRemoved.emit(item.meas_i_plot_item, Plot.MeasI)
            item.meas_i_plot_item = None
        if item.meas_b_plot_item is not None:
            self.plotRemoved.emit(item.meas_b_plot_item, Plot.MeasB)
            item.meas_b_plot_item = None
        if item.delta_plot_item is not None:
            self.plotRemoved.emit(item.delta_plot_item, Plot.Delta)
            item.delta_plot_item = None
        if item.ref_diff_plot_item is not None:
            self.plotRemoved.emit(item.ref_diff_plot_item, Plot.RefDiff)
            item.ref_diff_plot_item = None

        if item.color is not None:
            self._color_pool.return_color(item.color)
            item.color = None

        self._plotted_items.remove(item)
        item.is_shown = False

    def update_all_diff(self) -> None:
        """
        Trigger a full update of all (main) plots.
        """
        for item in self._plotted_items:
            self.update_diff_plot(item)

    def update_all_meas(self) -> None:
        """
        Trigger a full update of all measurement plots.
        """
        for item in self._plotted_items:
            self.update_meas_plot(item)

    def remove_all(self) -> None:
        """
        Trigger a full removal of all plots.
        """
        for item in self._plotted_items.copy():
            self.remove_cycle(item)

    def set_reference(self, item: PredictionItem) -> None:
        """
        Set the reference item for the plot.

        If changed, this will trigger an update of all plots.

        :param item: The item to use as reference.
        """
        current_reference = self._reference

        self._reference = item

        if current_reference is item:
            self.update_all_diff()

    def set_plot_mode(self, mode: DiffPlotMode | MeasPlotMode) -> None:
        """
        Change the plot mode.

        If changed, this will trigger an update of all plots.

        :param mode: The new plot mode.
        """
        if isinstance(mode, DiffPlotMode):
            current_mode = self._diff_plot_mode

            self._diff_plot_mode = mode

            log.debug(f"Plot mode changed to {mode.name}. Redrawing plots.")
            if current_mode != mode:
                self.update_all_diff()
        elif isinstance(mode, MeasPlotMode):
            current_meas_mode = self._meas_plot_mode

            self._meas_plot_mode = mode

            log.debug(f"Plot mode changed to {mode.name}. Redrawing plots.")
            if current_meas_mode != mode:
                self.update_all_meas()

    def get_plot_mode(self) -> DiffPlotMode:
        return self._diff_plot_mode

    plot_mode = property(get_plot_mode, set_plot_mode)

    def _zoom_flat_top(self) -> None:
        if len(self._plotted_items) == 0:
            log.debug("No items plotted, cannot zoom flat top.")
            return

        # find max and min flat top values for all recorded data
        max_val = max(
            [
                item.cycle_data.field_pred.max()
                for item in self._plotted_items
                if item.cycle_data.field_pred is not None
            ]
        )
        min_val = min(
            [
                item.cycle_data.field_pred.max()
                for item in self._plotted_items
                if item.cycle_data.field_pred is not None
            ]
        )

        self.setYRange.emit(0.997 * min_val, 1.0005 * max_val)

    def _zoom_flat_bottom(self) -> None: ...

    def _zoom_beam_in(self) -> None:
        if len(self._plotted_items) == 0:
            log.debug("No items plotted, cannot zoom beam in.")
            return

        self.setXRange.emit(self.beam_in, self.beam_out)

    def _reset_axes(self) -> None:
        if len(self._plotted_items) == 0:
            return

        max_val = max(
            [
                item.cycle_data.field_pred[1, :].max()
                for item in self._plotted_items
                if item.cycle_data.field_pred is not None
            ]
        )
        min_val = min(
            [
                item.cycle_data.field_pred[1, :].min()
                for item in self._plotted_items
                if item.cycle_data.field_pred is not None
            ]
        )

        self.setYRange.emit(min_val - 0.1, max_val + 0.1)
        self.setXRange.emit(0, next(iter(self._plotted_items)).cycle_data.num_samples)


def calc_dpp(
    reference: np.ndarray, value: np.ndarray, scale: float = 1.0
) -> np.ndarray:
    return (reference - value) / reference * scale


def calc_abs_diff(reference: np.ndarray, value: np.ndarray) -> np.ndarray:
    return reference - value


def calc_downsample(high: np.ndarray, low: np.ndarray) -> int:
    return int(np.ceil(len(high) / len(low)))


def make_pred_vs_pred(
    cycle_data: CycleData,
) -> tuple[np.ndarray, np.ndarray]:
    assert cycle_data.field_pred is not None
    y = cycle_data.field_pred[1, :]

    time_axis = _make_time_axis(cycle_data)
    x = time_axis[:: calc_downsample(time_axis, y)]
    y = np.interp(time_axis, x, y)

    return time_axis, y


def make_meas_vs_meas(
    cycle_data: CycleData,
) -> tuple[np.ndarray, np.ndarray]:
    assert cycle_data.field_meas is not None
    y = cycle_data.field_meas.flatten()

    x = np.arange(0, cycle_data.num_samples)

    return x, y


def make_pred_vs_meas(
    cycle_data: CycleData,
) -> tuple[np.ndarray, np.ndarray]:
    assert cycle_data.field_pred is not None
    assert cycle_data.field_meas is not None

    field_pred = cycle_data.field_pred[1, :]
    field_meas = cycle_data.field_meas.flatten()

    downsample = calc_downsample(field_meas, field_pred)

    time_axis = _make_time_axis(cycle_data)
    time_axis_downsampled = time_axis[::downsample]

    field_pred = np.interp(time_axis, time_axis_downsampled, field_pred)
    y = calc_abs_diff(field_meas, field_pred) * 1e4

    return time_axis, y


def make_pred_vs_ref(
    cycle_data: CycleData,
    reference: CycleData,
) -> tuple[np.ndarray, np.ndarray]:
    assert cycle_data.field_pred is not None
    assert reference.field_pred is not None

    field_pred = cycle_data.field_pred[1, :]
    field_ref = reference.field_pred[1, :]

    time_axis = _make_time_axis(cycle_data)
    y = calc_abs_diff(field_ref, field_pred) * 1e4
    x = time_axis[:: calc_downsample(time_axis, field_pred)]

    return time_axis, np.interp(time_axis, x, y)


def make_meas_vs_ref(
    cycle_data: CycleData,
    reference: CycleData,
) -> tuple[np.ndarray, np.ndarray]:
    assert cycle_data.field_meas is not None
    assert reference.field_meas is not None

    field_meas = cycle_data.field_meas.flatten()
    field_ref = reference.field_meas.flatten()

    y = calc_abs_diff(field_ref, field_meas) * 1e4
    x = np.arange(0, cycle_data.num_samples)

    return x, y


def _make_diff_curve(
    item: PredictionItem,
    reference: PredictionItem | None,
    diff_mode: DiffPlotMode,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Makes a new curve for the item based on the current plot mode.

    The curve is not added to the plot widget, in this method,
    but is returned to the caller.

    If a plot item already exists, it will be updated with the
    new data.

    :param item: The item to make the curve for.

    :raises ValueError: If the plot mode or reference is not set.
    """
    log.debug(
        "Creating/updating plot for cycle "
        f"{item.cycle_data.cycle_time}"
        " with mode "
        f"{diff_mode.name}"
    )
    cycle_data = item.cycle_data

    if diff_mode is DiffPlotMode.PredVsPred:
        return make_pred_vs_pred(cycle_data)
    elif diff_mode is DiffPlotMode.PredVsMeas:
        return make_pred_vs_meas(cycle_data)
    elif diff_mode is DiffPlotMode.PredVsRef:
        if reference is None:
            raise ValueError("No reference set.")
        return make_pred_vs_ref(cycle_data, reference.cycle_data)
    else:
        raise TypeError(f"Invalid plot mode {diff_mode}")


def _make_pred_curve(item: PredictionItem) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a curve for the dp/p plot.

    :param item: The item to create the curve for.
    """
    log.debug(f"Creating pred plot for cycle {item.cycle_data.cycle_time}")

    data = item.cycle_data
    assert data.field_pred is not None

    time_axis = _make_time_axis(item)
    field_pred = data.field_pred[1, :]
    x = time_axis[:: calc_downsample(time_axis, data.field_pred[1, :])]

    field_pred = np.interp(time_axis, x, field_pred)

    return time_axis, field_pred


def _make_curve_item(
    x: np.ndarray, y: np.ndarray, color: QtGui.QColor
) -> pg.PlotCurveItem:
    """
    Create a new curve item.

    :param x: The x data.
    :param y: The y data.
    :param color: The color of the curve.
    """
    return pg.PlotCurveItem(x=x, y=y, pen=pg.mkPen(color=color, width=1))


def _make_meas_curve(
    item: PredictionItem, y: np.ndarray, meas_mode: MeasPlotMode
) -> tuple[np.ndarray, np.ndarray]:
    x = _make_time_axis(item)
    y = y.flatten()
    if meas_mode is MeasPlotMode.RawMeas:
        return x, y
    elif meas_mode is MeasPlotMode.DownsampledMeas:
        assert item.cycle_data.field_pred is not None
        downsample = calc_downsample(x, item.cycle_data.field_pred[1, :])
        y = y[::downsample]

        return x[::downsample], y
    else:
        raise ValueError(f"Invalid plot mode {meas_mode.name}")


def _make_diff_pred_vs_meas(
    item: PredictionItem,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates:
        delta1 = field_pred_ref - field_pred
        delta2 = field_meas_ref - field_meas
        delta = delta2 - delta1
    """
    cycle = item.cycle_data
    assert cycle.field_meas is not None
    assert cycle.field_meas_ref is not None
    assert cycle.field_pred is not None
    assert cycle.field_ref is not None

    ref_t = cycle.field_ref[0, :] * 1e3
    delta1 = calc_abs_diff(cycle.field_ref[1, :], cycle.field_pred[1, :])

    meas_t = _make_time_axis(item)

    delta2 = calc_abs_diff(cycle.field_meas_ref, cycle.field_meas).flatten()
    delta2 = np.interp(ref_t, meas_t, delta2)

    delta = delta2 - delta1

    return ref_t, delta * 1e4


def _update_curve(x: np.ndarray, y: np.ndarray, curve: pg.PlotCurveItem) -> None:
    """
    Update the data of a curve.

    :param x: The x data.
    :param y: The y data.
    :param curve: The curve to update.
    """
    curve.setData(x=x, y=y)


def _make_time_axis(item: PredictionItem | CycleData) -> np.ndarray:
    cycle = item.cycle_data if isinstance(item, PredictionItem) else item
    x = np.arange(0, cycle.num_samples + 1)

    if str(len(x)).endswith("1"):
        x = x[:-1]

    return x
