"""
This module contains the model for the prediction analysis widget.
"""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import numpy as np
import pandas as pd
import pyda
import pyda_japc
import pyqtgraph as pg
from qtpy import QtCore, QtGui

from ...data import CycleData

log = logging.getLogger(__name__)


BEAM_IN = "SIX.MC-CTML/ControlValue#controlValue"
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue#controlValue"


class PlotMode(Enum):
    PredictedOnly = auto()
    VsMeasured = auto()
    dpp = auto()


@dataclass
class PredictionItem:
    cycle_data: CycleData
    plot_item: pg.PlotCurveItem | None = None
    plot_item_dpp: pg.PlotCurveItem | None = None
    color: QtGui.QColor | None = None
    shown: bool = False

    def __hash__(self) -> int:
        return hash(self.cycle_data.cycle_time)


class ColorPool:
    CM = pg.colormap.getFromMatplotlib("tab20")

    def __init__(self) -> None:
        assert self.CM is not None
        color_list = list(self.CM.getColors())
        random.shuffle(color_list)
        self._colors: deque[QtGui.QColor] = deque(
            [QtGui.QColor(*val) for val in color_list], maxlen=20
        )

    def get_color(self) -> QtGui.QColor:
        return self._colors.popleft()

    def return_color(self, color: QtGui.QColor) -> None:
        self._colors.append(color)


class PredictionListModel(QtCore.QAbstractListModel):
    itemAdded = QtCore.Signal(PredictionItem)
    itemRemoved = QtCore.Signal(PredictionItem)
    itemClicked = QtCore.Signal(QtCore.QModelIndex)

    def __init__(
        self, max_len: int = 10, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self._data: deque[PredictionItem] = deque(maxlen=max_len)

        # trigger row changed
        self.itemClicked.connect(
            lambda x: self.dataChanged.emit(x, x, [QtCore.Qt.BackgroundRole])
        )

    def data(self, index: QtCore.QModelIndex, role: int = 0) -> Any:
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
    ) -> Any:
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


class PredictionPlotModel(QtCore.QObject):
    addPlot = QtCore.Signal(PredictionItem)
    """ Triggered when a new plot is added """

    updatePlot = QtCore.Signal(PredictionItem)
    """ Triggered when a plot is updated """

    removePlot = QtCore.Signal(PredictionItem)
    """ Triggered when an existing plot is removed """

    newReference = QtCore.Signal(PredictionItem)
    """ Triggered when a new reference is set """

    plotAdded = QtCore.Signal(pg.PlotCurveItem)
    """ Triggered when a new plot is added """

    plotRemoved = QtCore.Signal(pg.PlotCurveItem)
    """ Triggered when an existing plot is removed """

    plotAdded_dpp = QtCore.Signal(pg.PlotCurveItem)
    """ Triggered when a new plot is added """

    plotRemoved_dpp = QtCore.Signal(pg.PlotCurveItem)
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

        self.addPlot.connect(self.add_item)
        self.removePlot.connect(self.remove_item)
        self.newReference.connect(self.set_reference)
        self.zoomFlatTop.connect(self._zoom_flat_top)
        self.zoomFlatBottom.connect(self._zoom_flat_bottom)
        self.zoomBeamIn.connect(self._zoom_beam_in)
        self.resetAxes.connect(self._reset_axes)

        self._plot_mode = PlotMode.PredictedOnly
        self._reference: PredictionItem | None = None
        self._color_pool = ColorPool()

        self.beam_in: int = 0
        self.beam_out: int = 0

    def add_item(self, item: PredictionItem) -> None:
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

        try:
            curve = self._make_curve(item)
        except ValueError:  # missing reference
            curve = None
        curve_dpp = self._make_curve_dpp(item)

        item.plot_item_dpp = curve_dpp

        self._plotted_items.add(item)
        item.shown = True

        if curve is not None:
            item.plot_item = curve
            self.plotAdded.emit(curve)
        self.plotAdded_dpp.emit(item.plot_item_dpp)

    def update_item(self, item: PredictionItem) -> None:
        """
        Update an existing item in the plot.

        The item is sent to the plotAdded Signal if it was not
        plotted before.

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
            curve = self._make_curve(item)
        except ValueError:  # missing reference
            curve = None

        if curve is not None and item.plot_item is None:
            item.plot_item = curve
            self.plotAdded.emit(item)

    def remove_item(self, item: PredictionItem) -> None:
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

        if item.plot_item is not None:
            self.plotRemoved.emit(item.plot_item)
            item.plot_item = None
        if item.plot_item_dpp is not None:
            self.plotRemoved_dpp.emit(item.plot_item_dpp)
            item.plot_item_dpp = None

        if item.color is not None:
            self._color_pool.return_color(item.color)
            item.color = None

        self._plotted_items.remove(item)
        item.shown = False

    def update_all(self) -> None:
        """
        Trigger a full update of all (main) plots.
        """
        for item in self._plotted_items:
            self.update_item(item)

    def remove_all(self) -> None:
        """
        Trigger a full removal of all plots.
        """
        for item in self._plotted_items.copy():
            self.remove_item(item)

    def set_reference(self, item: PredictionItem) -> None:
        """
        Set the reference item for the plot.

        If changed, this will trigger an update of all plots.

        :param item: The item to use as reference.
        """
        current_reference = self._reference

        self._reference = item

        if current_reference is not None:
            self.update_all()

    def set_plot_mode(self, mode: PlotMode) -> None:
        """
        Change the plot mode.

        If changed, this will trigger an update of all plots.

        :param mode: The new plot mode.
        """
        current_mode = self._plot_mode

        self._plot_mode = mode

        log.debug(f"Plot mode changed to {mode.name}. Redrawing plots.")
        if current_mode != mode:
            self.update_all()

    def get_plot_mode(self) -> PlotMode:
        return self._plot_mode

    plot_mode = property(get_plot_mode, set_plot_mode)

    def _make_curve(self, item: PredictionItem) -> pg.PlotCurveItem:
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
            "Creating/updating plot for cycle " f"{item.cycle_data.cycle_time}"
        )
        if item.color is None:
            item.color = self._color_pool.get_color()

        cycle_data = item.cycle_data
        assert cycle_data.field_pred is not None

        x = self._make_time_axis(item)
        if self._plot_mode == PlotMode.PredictedOnly:
            y = cycle_data.field_pred[1, :]
            x = x[:: len(x) // len(y)]
        elif self._plot_mode == PlotMode.VsMeasured:
            assert cycle_data.field_meas is not None

            downsample = cycle_data.num_samples // len(
                cycle_data.field_pred[0, :]
            )
            y = (
                cycle_data.field_meas[::downsample]
                - cycle_data.field_pred[1, :]
            )
            x = x[:: len(x) // len(y)]
        elif self._plot_mode == PlotMode.dpp:
            if self._reference is None:
                raise ValueError("No reference set.")
            reference = self._reference

            assert reference.cycle_data.field_pred is not None

            y = (
                (reference.cycle_data.field_pred - cycle_data.field_pred)
                / reference.cycle_data.field_pred
                * 1e4
            )[1, :]
            x = x[:: len(x) // len(y)]
        else:
            raise ValueError(f"Invalid plot mode {self._plot_mode.name}")

        args = {"x": x, "y": y, "pen": pg.mkPen(color=item.color, width=1)}

        if item.plot_item is not None:
            log.debug(
                "Updating existing plot for cycle "
                f"{item.cycle_data.cycle_time}"
            )
            args.pop("pen")
            item.plot_item.updateData(**args)
            curve = item.plot_item
        else:
            log.debug(
                f"Creating new plot for cycle {item.cycle_data.cycle_time}"
            )
            curve = pg.PlotCurveItem(**args)

        return curve

    def _make_curve_dpp(self, item: PredictionItem) -> pg.PlotCurveItem:
        """
        Create a curve for the dp/p plot.

        :param item: The item to create the curve for.
        """
        log.debug(f"Creating dp/p plot for cycle {item.cycle_data.cycle_time}")
        if item.color is None:
            item.color = self._color_pool.get_color()

        data = item.cycle_data
        assert data.field_meas is not None
        assert data.field_pred is not None

        time_axis = self._make_time_axis(item)
        dp_p = item.cycle_data.dp_p * 1e4

        downsample = time_axis.size // dp_p.size

        x = time_axis[::downsample]

        curve = pg.PlotCurveItem(
            x=x,
            y=dp_p,
            pen=pg.mkPen(color=item.color, width=1),
        )

        return curve

    @staticmethod
    def _make_time_axis(item: PredictionItem) -> np.ndarray:
        return np.arange(0, item.cycle_data.num_samples)

    def _zoom_flat_top(self) -> None:
        if len(self._plotted_items) == 0:
            log.debug("No items plotted, cannot zoom flat top.")
            return

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

    def _zoom_flat_bottom(self) -> None:
        ...

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
        self.setXRange.emit(
            0, next(iter(self._plotted_items)).cycle_data.num_samples
        )


class PredictionAnalysisModel(QtCore.QObject):
    """
    Model for the prediction analysis widget.

    The model contains the model for the QListView,
    and filters incoming data based on selector,
    and only appends data if the selector matches the one
    saved in the model.
    """

    newData = QtCore.Signal(CycleData)
    """ Triggered by new data acquisition (new predictions) """

    superCycleChanged = QtCore.Signal()
    """ Triggered when a supercycle is detected (externally) """

    maxBufferSizeChanged = QtCore.Signal(int)
    """ Triggered when the buffer size changes """

    userChanged = QtCore.Signal(str)
    """ Triggered when the user changes """

    plotModeChanged = QtCore.Signal(PlotMode)

    def __init__(
        self,
        max_buffer_samples: int = 10,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._list_model = PredictionListModel(max_len=max_buffer_samples)
        self._plot_model = PredictionPlotModel()

        self._selector: str | None = None

        # state
        self._watch_supercycle = False
        self._supercycle_patience = 0
        self._acq_enabled: bool = False

        # active flags
        self._count_acq_supercycle: bool = False
        self._n_acq_since_supercycle = 0

        # connect signals
        self.maxBufferSizeChanged.connect(self.set_max_buffer_samples)
        self.newData.connect(self._on_new_data_received)
        self.superCycleChanged.connect(self._on_supercycle_changed)
        self.plotModeChanged.connect(self._on_plot_mode_changed)
        self.list_model.itemRemoved.connect(self.plot_model.removePlot.emit)
        self.list_model.modelReset.connect(self.plot_model.remove_all)

        self._da = pyda.SimpleClient(provider=pyda_japc.JapcProvider())

    @property
    def list_model(self) -> PredictionListModel:
        return self._list_model

    @property
    def plot_model(self) -> PredictionPlotModel:
        return self._plot_model

    def set_max_buffer_samples(self, max_buffer_samples: int) -> None:
        self._list_model.max_len = max_buffer_samples

    def get_max_buffer_samples(self) -> int:
        return self._list_model.max_len

    max_buffer_samples = property(
        get_max_buffer_samples, set_max_buffer_samples
    )

    def set_watch_supercycle(self, watch_supercycle: bool) -> None:
        log.debug(f"Setting watch supercycle to {watch_supercycle}.")
        self._watch_supercycle = watch_supercycle

    def get_watch_supercycle(self) -> bool:
        return self._watch_supercycle

    watch_supercycle = property(get_watch_supercycle, set_watch_supercycle)

    def set_supercycle_patience(self, supercycle_patience: int) -> None:
        log.debug(
            f"Setting supercycle watch patience to {supercycle_patience}."
        )
        self._supercycle_patience = supercycle_patience

    def get_supercycle_patience(self) -> int:
        return self._supercycle_patience

    supercycle_patience = property(
        get_supercycle_patience, set_supercycle_patience
    )

    def set_selector(self, selector: str | None) -> None:
        current_selector = self._selector

        self._selector = selector

        if current_selector != selector:
            log.debug(f"Selector changed to {selector}. Clearing model.")
            self._list_model.clear()

            if selector is not None:
                beam_in = self._da.get(BEAM_IN, context=selector).value[
                    "value"
                ]
                beam_out = self._da.get(BEAM_OUT, context=selector).value[
                    "value"
                ]

                self.plot_model.beam_in = beam_in
                self.plot_model.beam_out = beam_out

    def get_selector(self) -> str | None:
        return self._selector

    selector = property(get_selector, set_selector)

    def enable_acquisition(self, enable: bool = True) -> None:
        if enable:
            log.debug("Enabling acquisition.")
            self._acq_enabled = True
        else:
            log.debug("Disabling acquisition.")
            self._acq_enabled = False

    def disable_acquisition(self) -> None:
        self.enable_acquisition(False)

    def _on_new_data_received(self, cycle_data: CycleData) -> None:
        if self._selector is None:
            log.debug("No selector set. Discarding new data.")
            return
        elif self._selector != cycle_data.user:
            log.debug(
                f"Selector {self._selector} does not match "
                f"{cycle_data.user}. Discarding it."
            )
            return
        # else:

        if not self._acq_enabled:
            log.debug("Acquisition is disabled. Discarding new data.")
            return

        if self._count_acq_supercycle:
            if self._n_acq_since_supercycle >= self._supercycle_patience:
                log.debug(
                    "Acquired more data than patience, discarding new data."
                )
                return
            else:
                self._n_acq_since_supercycle += 1

        if cycle_data.field_pred is None:
            log.debug(
                f"No field prediction for {cycle_data.user}. Discarding."
            )
            return

        log.debug(f"Adding new data to model for {cycle_data.user}.")
        self._list_model.append(cycle_data)

    def _on_supercycle_changed(self) -> None:
        log.debug("New supercycle detected. Clearing model.")

        if self._count_acq_supercycle:
            log.debug(
                "Already counting supercycles. "
                f"Count: {self._n_acq_since_supercycle}"
            )
            return

        self._count_acq_supercycle = True
        self._n_acq_since_supercycle = 0

    def _on_plot_mode_changed(self, plot_mode: PlotMode) -> None:
        self._plot_model.plot_mode = plot_mode

    def item_clicked(self, index: QtCore.QModelIndex) -> None:
        item = self._list_model.itemAt(index)
        if item.shown:
            self._plot_model.removePlot.emit(item)
        else:
            self._plot_model.addPlot.emit(item)

        self._list_model.itemClicked.emit(index)

    def clear(self) -> None:
        self._list_model.clear()

        self._n_acq_since_supercycle = 0

    def to_pandas(self) -> pd.DataFrame:
        """
        Export the currently saved predictions to Pandas.
        """
        predictions = self.list_model.buffered_data

        df = pd.concat(
            [o.cycle_data.to_pandas() for o in predictions],
        )

        return df

    def from_pandas(self, df: pd.DataFrame) -> None:
        """
        Load predictions from a Pandas DataFrame.
        """
        log.debug("Loading predictions from Pandas DataFrame.")

        predictions = [
            CycleData.from_pandas(row.to_frame()) for _, row in df.iterrows()
        ]

        user = predictions[0].user
        self.userChanged.emit(user)

        self.clear()
        for pred in predictions:
            self._list_model.append(pred)
