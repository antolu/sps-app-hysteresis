from __future__ import annotations

import logging
import typing
from threading import Lock

import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.signal
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
                self._predictor.load_from_checkpoint(ckpt_path)
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

        if USE_PROGRAMMED_CURRENT:
            # past_current = []  hack flat MD1
            # for data in cycle_data[:-1]:
            # if data.user != "SPS.USER.MD1":
            #     past_current.append(upscale_programmed_current(data)[:-1])
            # else:
            #     I_prog = data.current_prog[1]
            #     # take first point and make it last point, then interpolate
            #     I_prog = np.array([I_prog[0], I_prog[0]])
            #     t_prog = np.array(
            #         [data.current_prog[0][0], data.current_prog[0][-1]]
            #     )
            #     t = np.arange(0, data.current_input.size + 1, 1)
            #     I = np.interp(t, t_prog, I_prog)

            #     past_current.append(I[:-1])
            # past_current = np.concatenate(past_current)
            past_current = np.concatenate(
                [
                    upscale_programmed_current(data)[:-1]
                    for data in cycle_data[:-1]
                ]
            )
            future_current = upscale_programmed_current(cycle_data[-1])[:-1]
        else:
            for data in cycle_data:
                if data.current_input is None:
                    raise ValueError("Not all data has input current set.")
            past_current = np.concatenate(
                [filter_(data.current_input) for data in cycle_data[:-1]]
            )
            future_current = filter_(cycle_data[-1].current_input)
        past_covariates = pd.DataFrame(
            {
                "I_meas_A": past_current,
                "I_meas_A_dot": calc_time_derivative(past_current),
            }
        )
        future_covariates = pd.DataFrame(
            {
                "I_meas_A": filter_(future_current),
                "I_meas_A_dot": calc_time_derivative(filter_(future_current)),
            }
        )
        if USE_PROGRAMMED_CURRENT:
            future_covariates["I_meas_A"] = upscale_programmed_current(
                cycle_data[-1]
            )[:-1]
            future_covariates["I_meas_A_dot"] = calc_time_derivative(
                future_covariates["I_meas_A"]
            )

        for data in cycle_data[:-1]:
            if data.field_meas is None:
                raise RuntimeError(
                    "Not all data in the past has measured field set."
                )
        past_field = np.concatenate(
            [filter_(data.field_meas) for data in cycle_data[:-1]]  # type: ignore[arg-type]
        )

        input_columns = self._predictor._datamodule.hparams["input_columns"]
        past_covariates = past_covariates.rename(
            columns={
                "I_meas_A": input_columns[0],
                "I_meas_A_dot": input_columns[1],
            }
        )
        future_covariates = future_covariates.rename(
            columns={
                "I_meas_A": input_columns[0],
                "I_meas_A_dot": input_columns[1],
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
            + cycle_data[-1].cycle_timestamp / NS
        )
        time_axis = time_axis[
            :: int(len(future_covariates) / len(predictions))
        ]

        log.debug(
            f"Made time axis of length {len(time_axis)} for {len(predictions)} predictions"
        )

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
    col_smoothed = scipy.signal.medfilt(column, kernel_size=5)
    gradient = np.gradient(col_smoothed)

    return gradient
