from __future__ import annotations

import logging
from threading import Lock

import numpy as np
import pandas as pd
import scipy.signal
from hysteresis_scripts.predict import Predictor
from qtpy import QtCore, QtWidgets

from ..data import CycleData
from ..utils import ThreadWorker, load_cursor, run_in_thread, time_execution

MS = int(1e3)
NS = int(1e9)

log = logging.getLogger(__name__)


_thread: QtCore.QThread | None = None


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
        self, device: str = "cpu", parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        # self.do_inference.connect(self._do_inference)

        self._predictor = Predictor(device=device)

        self._lock = Lock()
        self._do_inference = False
        self._doing_inference = False

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
                self, "Error loading model", str(e), QtWidgets.QMessageBox.Ok
            )

        worker.exception.connect(on_exception)

        QtCore.QThreadPool.globalInstance().start(worker)

    def _on_load_model(
        self, model_name: str, ckpt_path: str, device: str = "cpu"
    ) -> None:
        with load_cursor():
            try:
                self._predictor.device = device
                self._predictor.load_from_checkpoint(model_name, ckpt_path)
            except:  # noqa F722
                log.exception("Error occurred.")
                return

            with self._lock:
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
        for data in cycle_data:
            if data.current_input is None:
                raise ValueError("Not all data has input current set.")

        current_input = np.concatenate(
            [data.current_input for data in cycle_data]
        )

        last_cycle = cycle_data[-1]
        log.debug(f"Running prediction on {len(current_input)} samples.")

        past_current = np.concatenate(
            [data.current_input for data in cycle_data[:-1]]
        )
        for data in cycle_data[:-1]:
            if data.field_meas is None:
                raise RuntimeError(
                    "Not all data in the past has measured field set."
                )
        past_field = np.concatenate(
            [
                data.field_meas
                for data in cycle_data[:-1]
                if data.field_meas is not None
            ]
        )
        past_covariates = pd.DataFrame(
            {
                "I_meas_A": past_current,
                "I_meas_A_dot": calc_time_derivative(past_current),
            }
        )
        future_covariates = pd.DataFrame(
            {
                "I_meas_A": cycle_data[-1].current_input,
                "I_meas_A_dot": calc_time_derivative(
                    cycle_data[-1].current_input
                ),
            }
        )

        log.debug("Running inference.")
        with time_execution() as timer:
            predictions = self._predictor.predict(
                past_covariates,
                future_covariates,
                past_field,
                exception_on_failure=True,
                upsample=False,
            )
        log.info("Inference took: %f s", timer.duration)

        time_axis = (
            np.arange(len(future_covariates)) / MS
            + last_cycle.cycle_timestamp / NS
        )
        time_axis = time_axis[:: int(len(future_covariates) / len(predictions))]

        return np.stack((time_axis, predictions), axis=0)

    @property
    def model_is_loaded(self) -> bool:
        return self._predictor._module is not None

    def set_do_inference(self, state: bool) -> None:
        with self._lock:
            self._do_inference = state


def calc_time_derivative(column: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(column, pd.Series):
        column = column.to_numpy()
    col_smoothed = scipy.signal.medfilt(column, kernel_size=25)
    gradient = np.gradient(col_smoothed)

    return gradient
