"""
This module contains the model for the prediction analysis widget.
"""

from __future__ import annotations

import logging

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
