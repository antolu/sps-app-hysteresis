from __future__ import annotations


import logging
import typing

import numpy as np
from sps_mlp_hystcomp import PETEPredictor, TFTPredictor
from qtpy import QtCore, QtWidgets

from hystcomp_utils.cycle_data import CycleData
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
    cyclePredicted = QtCore.Signal(CycleData, np.ndarray)
    model_loaded = QtCore.Signal()

    started = QtCore.Signal()
    completed = QtCore.Signal()

    def __init__(
        self,
        device: typing.Literal["cpu", "cuda", "auto"] = "cpu",
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._predictor: PETEPredictor | TFTPredictor | None = None

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
        if model_name == "PETE":
            predictor_cls = PETEPredictor
        elif model_name == "TemporalFusionTransformer":
            predictor_cls = TFTPredictor
        else:
            msg = f"Unknown model name: {model_name}"
            log.exception(msg)
            raise ValueError(msg)

        with load_cursor():
            try:
                self._predictor = predictor_cls.load_from_checkpoint(
                    ckpt_path, device=device
                )
            except:  # noqa F722
                log.exception("Error occurred.")
                return

            self._ckpt_path = ckpt_path

        self.model_loaded.emit()

    @run_in_thread(inference_thread)
    def predict_last_cycle(self, cycle_data: list[CycleData]) -> None:
        if self._predictor is None:
            log.error("Model not loaded. Cannot predict.")
            return None

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
            self.cyclePredicted.emit(last_cycle)
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
        if self._predictor is None:
            msg = "Model not loaded. Cannot predict."
            log.error(msg)
            raise ValueError(msg)

        log.debug("Running inference.")
        with time_execution() as timer:
            time_axis, predictions = self._predictor.predict_last_cycle(
                cycle_data,
                autoregressive=self.autoregressive,
                use_programmed_current=self.use_programmed_current,
            )
        log.info("Inference took: %f s", timer.duration)

        time_axis += cycle_data[-1].cycle_timestamp / NS

        return np.stack((time_axis, predictions), axis=0)

    @property
    def model_is_loaded(self) -> bool:
        return (
            self._predictor is not None and self._predictor._module is not None
        )  # noqa: SLF001

    def set_do_inference(self, state: bool) -> None:
        self._do_inference = state

    def set_autoregressive(self, state: bool) -> None:
        self._autoregressive = state

    @property
    def autoregressive(self) -> bool:
        return self._autoregressive

    @autoregressive.setter
    def autoregressive(self, value: bool) -> None:
        self.set_autoregressive(value)

    def set_use_programmed_current(self, state: bool) -> None:
        self._use_programmed_current = state

    @property
    def use_programmed_current(self) -> bool:
        return self._use_programmed_current

    @use_programmed_current.setter
    def use_programmed_current(self, value: bool) -> None:
        self.set_use_programmed_current(value)

    def reset_state(self) -> None:
        if self._predictor is not None:
            self._predictor.reset_state()
        else:
            log.error("Model not loaded. Cannot reset state.")
