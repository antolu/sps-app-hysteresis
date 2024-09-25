"""
This module contains the model for the prediction analysis widget.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pyda
import pyda_japc
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore

from ._dataclass import DiffPlotMode, MeasPlotMode
from ._list_model import PredictionListModel
from ._plot_model import PredictionPlotModel

log = logging.getLogger(__name__)


BEAM_IN = "SIX.MC-CTML/ControlValue#controlValue"
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue#controlValue"


class PredictionAnalysisModel(QtCore.QObject):
    """
    Model for the prediction analysis widget.

    The model contains the model for the QListView,
    and filters incoming data based on selector,
    and only appends data if the selector matches the one
    saved in the model.
    """

    superCycleChanged = QtCore.Signal()
    """ Triggered when a supercycle is detected (externally) """

    maxBufferSizeChanged = QtCore.Signal(int)
    """ Triggered when the buffer size changes """

    userChanged = QtCore.Signal(str)
    """ Triggered when the user changes """

    diffPlotModeChanged = QtCore.Signal(DiffPlotMode)
    """ Triggered when the diff plot mode changes """

    measPlotModeChanged = QtCore.Signal(MeasPlotMode)
    """ Triggered when the meas plot mode changes """

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
        self.superCycleChanged.connect(self._on_supercycle_changed)
        self.diffPlotModeChanged.connect(self._on_plot_mode_changed)
        self.measPlotModeChanged.connect(self._on_plot_mode_changed)
        self.list_model.itemRemoved.connect(self.plot_model.remove_cycle)
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

    max_buffer_samples = property(get_max_buffer_samples, set_max_buffer_samples)

    def set_watch_supercycle(self, watch_supercycle: bool) -> None:
        log.debug(f"Setting watch supercycle to {watch_supercycle}.")
        self._watch_supercycle = watch_supercycle

    def get_watch_supercycle(self) -> bool:
        return self._watch_supercycle

    watch_supercycle = property(get_watch_supercycle, set_watch_supercycle)

    def set_supercycle_patience(self, supercycle_patience: int) -> None:
        log.debug(f"Setting supercycle watch patience to {supercycle_patience}.")
        self._supercycle_patience = supercycle_patience

    def get_supercycle_patience(self) -> int:
        return self._supercycle_patience

    supercycle_patience = property(get_supercycle_patience, set_supercycle_patience)

    def set_selector(self, selector: str | None) -> None:
        current_selector = self._selector

        self._selector = selector

        if current_selector != selector:
            log.debug(f"Selector changed to {selector}. Clearing model.")
            self._list_model.clear()

            if selector is not None:
                beam_in = self._da.get(BEAM_IN, context=selector).value["value"]
                beam_out = self._da.get(BEAM_OUT, context=selector).value["value"]

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

    @QtCore.Slot(CycleData)
    def onNewMeasuredData(self, cycle_data: CycleData) -> None:
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
                log.debug("Acquired more data than patience, discarding new data.")
                return
            else:
                self._n_acq_since_supercycle += 1

        if cycle_data.field_pred is None:
            log.debug(f"[{cycle_data}]: No field prediction. Discarding.")
            return

        log.debug(f"[{cycle_data}]: Adding new data to model.")
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

    def _on_plot_mode_changed(self, plot_mode: DiffPlotMode) -> None:
        self._plot_model.plot_mode = plot_mode

    def item_clicked(self, index: QtCore.QModelIndex) -> None:
        item = self._list_model.itemAt(index)
        if item.is_shown:
            self._plot_model.remove_cycle(item)
        else:
            self._plot_model.show_cycle(item)

        self._list_model.clicked(index)

    def clear(self) -> None:
        self._list_model.clear()

        self._n_acq_since_supercycle = 0

    def to_pandas(self) -> pd.DataFrame:
        """
        Export the currently saved predictions to Pandas.
        """
        predictions = self.list_model.buffered_data

        df = pd.concat(
            [to_pandas(o.cycle_data) for o in predictions],
        )

        return df

    def from_pandas(self, df: pd.DataFrame) -> None:
        """
        Load predictions from a Pandas DataFrame.
        """
        log.debug("Loading predictions from Pandas DataFrame.")

        predictions = [from_pandas(row.to_frame()) for _, row in df.iterrows()]

        user = predictions[0].user
        self.userChanged.emit(user)

        self.clear()
        for pred in predictions:
            self._list_model.append(pred)


def to_dict(
    data: CycleData,
) -> dict[str, np.ndarray | int | float | str | datetime | None]:
    return {
        "cycle": data.cycle,
        "user": data.user,
        "cycle_time": data.cycle_time,
        "cycle_timestamp": data.cycle_timestamp,
        "cycle_length": data.cycle_length,
        "current_prog": data.current_prog.flatten(),
        "field_prog": data.field_prog.flatten(),
        "current_input": (
            data.current_input if hasattr(data, "current_input") else None
        ),
        "field_ref": (data.field_ref.flatten() if data.field_ref is not None else None),
        "field_pred": (
            data.field_pred.flatten() if data.field_pred is not None else None
        ),
        "current_meas": data.current_meas,
        "field_meas": data.field_meas,
        "num_samples": data.num_samples,
        "correction": (
            data.correction.flatten() if data.correction is not None else None
        ),
    }


def from_dict(d: dict) -> CycleData:  # type: ignore
    current_prog = d["current_prog"]
    field_prog = d["field_prog"]

    current_prog = from_1d_array(current_prog)
    field_prog = from_1d_array(field_prog)
    item = CycleData(
        d["cycle"],
        d["user"],
        d["cycle_timestamp"],
        current_prog,
        field_prog,
    )

    item.current_input = d["current_input"]
    item.field_pred = (
        from_1d_array(d["field_pred"]) if d["field_pred"] is not None else None
    )
    item.field_ref = (
        from_1d_array(d["field_ref"]) if d["field_ref"] is not None else None
    )
    item.current_meas = d["current_meas"]
    item.field_meas = d["field_meas"]
    item.correction = (
        from_1d_array(d["correction"]) if d["correction"] is not None else None
    )

    return item


def to_pandas(data: CycleData) -> pd.DataFrame:
    """
    Export cycle data to a Pandas DataFrame.
    """
    return pd.DataFrame.from_dict({k: [v] for k, v in to_dict(data).items()})


def from_pandas(df: pd.DataFrame) -> CycleData:
    """
    Load predictions from a Pandas DataFrame.
    """
    if len(df) != 1:
        raise ValueError("DataFrame must have only one row")

    return from_dict({k: v[0] for k, v in df.to_dict().items()})


def from_1d_array(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(2, arr.size // 2)
