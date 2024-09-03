from __future__ import annotations


import logging
import typing

import numpy as np
from hysteresis_scripts.predict import PETEPredictor
from qtpy import QtCore, QtWidgets

from ..data import CycleData
from ..utils import ThreadWorker, load_cursor, run_in_thread, time_execution

MS = int(1e3)
NS = int(1e9)

log = logging.getLogger(__name__)


_thread: QtCore.QThread | None = None


USE_PROGRAMMED_CURRENT = True


def upscale_programmed_current(data: CycleData) -> np.ndarray:
    I_prog = data.current_prog[1]
    t_prog = data.current_prog[0]
    t = np.arange(0, data.num_samples + 1, 1)
    I = np.interp(t, t_prog, I_prog)  # noqa F741

    return I


def filter_(data: np.ndarray) -> np.ndarray:
    return data
    # return scipY.NDIMAGE.MEDIAN_FILTER(DATA, SIZE=51, MODE="NEAREST")


def inference_thread() -> QtCore.QThread:
    global _thread
    if _thread is None:
        _thread = QtCore.QThread()
    return _thread


class Inference(QtCore.QObject):
    load_model = QtCore.Signal(str, str, str)  # ckpt path, device
    cycle_predicted = QtCore.Signal(CycleData, np.ndarray)
    model_loaded = QtCore.Signal()

    started = QtCore.Signal()
    completed = QtCore.Signal()

    def __init__(
        self,
        device: typing.Literal["cpu", "cuda", "auto"] = "cpu",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._predictor = PETEPredictor(device=device)

        self._do_inference = False
        self._autoregressive = False
        self._use_programmed_current = True

        self.load_model.connect(self.on_load_model)

    def on_load_model(
        self, model_name: str, ckpt_path: str, device: str = "cpu"
    ) -> None:
        worker = ThreadWorker(
            self._on_load_model, model_name, ckpt_path, device
        )

        def on_exception(e: Exception) -> None:
            log.exception("Error loading model.")
            QtWidgets.QMessageBox.critical(
                None, "Error loading model", str(e), QtWidgets.QMessageBox.Ok
            )

        worker.exception.connect(on_exception)

        QtCore.QThreadPool.globalInstance().start(worker)

    def _on_load_model(
        self, model_name: str, ckpt_path: str, device: str = "cpu"
    ) -> None:
        with load_cursor():
            try:
                self._predictor.device = device
                self._predictor.load_checkpoint(ckpt_path)
            except:  # noqa F722
                log.exception("Error occurred.")
                return

            with self._predictor.lock:
                self._ckpt_path = ckpt_path

        self.model_loaded.emit()

    @run_in_thread(inference_thread)
    def predict_last_cycle(self, cycle_data: list[CycleData]) -> None:
        if not self._do_inference:
            log.debug("Inference is disabled. Not predicting.")
            return None

        if self._predictor.busy:
            log.warning(
                "Inference is already underway. " "Cannot do more in parallel."
            )
            return None

        self.started.emit()
        # first check if all data has current set
        try:
            predictions = self._predict_last_cycle(cycle_data)
            last_cycle = cycle_data[-1]

            last_cycle.field_pred = predictions
            self.cycle_predicted.emit(last_cycle, predictions)
        except:  # noqa F722
            log.exception("Inference failed.")
        finally:
            self.completed.emit()

    def _predict_last_cycle(self, cycle_data: list[CycleData]) -> np.ndarray:
        """
        Predict the field for the last cycle in the given data.

        :param cycle_data: The data to predict the field for.

        :return: The predicted field of the last cycle.

        :raises ValueError: If not all data has input current set.
        """

        log.debug("Running inference.")
        with time_execution() as timer:
            predictions = self._predictor.predict_last_cycle(
                cycle_data,
                autoregressive=self.autoregressive,
                use_programmed_current=self.use_programmed_current,
            )
        log.info("Inference took: %f s", timer.duration)

        time_axis = (
            np.arange(cycle_data[-1].num_samples) / MS
            + cycle_data[-1].cycle_timestamp / NS
        )
        time_axis = time_axis[
            :: int(cycle_data[-1].num_samples / len(predictions))
        ]

        log.debug(
            f"Made time axis of length {len(time_axis)} for {len(predictions)} predictions"
        )

        return np.stack((time_axis, predictions), axis=0)

    @property
    def model_is_loaded(self) -> bool:
        return self._predictor._module is not None

    def set_do_inference(self, state: bool) -> None:
        with self._predictor.lock:
            self._do_inference = state

    def set_autoregressive(self, state: bool) -> None:
        with self._predictor.lock:
            self._autoregressive = state

    @property
    def autoregressive(self) -> bool:
        return self._autoregressive

    @autoregressive.setter
    def autoregressive(self, value: bool) -> None:
        self.set_autoregressive(value)

    def set_use_programmed_current(self, state: bool) -> None:
        with self._predictor.lock:
            self._use_programmed_current = state

    @property
    def use_programmed_current(self) -> bool:
        return self._use_programmed_current

    @use_programmed_current.setter
    def use_programmed_current(self, value: bool) -> None:
        self.set_use_programmed_current(value)
