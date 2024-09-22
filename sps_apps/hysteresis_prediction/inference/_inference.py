from __future__ import annotations

import logging
import typing

import numpy as np
import pyda.data
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore, QtWidgets
from sps_mlp_hystcomp import PETEPredictor, TFTPredictor

from ..data import EventBuilderAbc
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


class Inference(EventBuilderAbc):
    model_loaded = QtCore.Signal()

    predictionStarted = QtCore.Signal()
    predictionFinished = QtCore.Signal()

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

        self._prev_state: typing.Any | None = None

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        pass

    @QtCore.Slot(str, str, str)
    def loadModel(self, model_name: str, ckpt_path: str, device: str = "cpu") -> None:
        worker = ThreadWorker(self._on_load_model, model_name, ckpt_path, device)

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
                self._predictor.prog_t_phase = 0.15 * 1e-3
            except:  # noqa F722
                log.exception("Error occurred.")
                return

            self._ckpt_path = ckpt_path

        self.model_loaded.emit()

    @QtCore.Slot(list[CycleData])
    def onNewCycleDataBuffer(self, cycle_data: list[CycleData]) -> None:
        return self.predict_last_cycle(cycle_data)

    def predict_last_cycle(self, cycle_data: list[CycleData]) -> None:
        if self._predictor is None:
            log.debug("Model not loaded. Cannot predict.")
            return None

        if not self._do_inference:
            log.debug("Inference is disabled. Not predicting.")
            return None

        if self._predictor.busy:
            log.warning("Inference is already underway. " "Cannot do more in parallel.")
            return None

        self.predictionStarted.emit()
        # first check if all data has current set
        try:
            with time_execution() as t:
                predictions = self._predict_last_cycle(cycle_data)
            log.info(f"[{cycle_data[-1]}]: Prediction took {t.duration * 1e3:.2f} ms.")
            last_cycle = cycle_data[-1]

            predictions[0] = np.round([predictions[0]], 3)  # round to ms
            last_cycle.field_pred = predictions
            self.cycleDataAvailable.emit(last_cycle)
        except:  # noqa F722
            log.exception("Inference failed.")
        finally:
            self.predictionFinished.emit()

    @run_in_thread(inference_thread)
    def _predict_last_cycle(self, buffer: list[CycleData]) -> np.ndarray:
        """
        Predict the field for the last cycle in the given data.

        :param cycle_data: The data to predict the field for.

        :return: The predicted field of the last cycle.

        :raises ValueError: If not all data has input current set.
        """
        assert self._predictor is not None

        last_cycle = buffer[-1]

        if self._prev_state is None:  # no way we can be in ECO cycle
            if last_cycle.cycle.endswith("ECO"):
                msg = (
                    f"[{last_cycle}]: ECO cycle detected, but previous state is not set."
                    f"This should not happen."
                )
                log.debug(msg)
                raise ValueError(msg)

            self._predictor.set_cycled_initial_state(
                buffer[:-1],
                use_programmed_current=self._use_programmed_current,
            )
            self._prev_state = self._predictor.state

            return self._predictor.predict_cycle(
                cycle=last_cycle,
                save_state=True,
                use_programmed_current=self._use_programmed_current,
            )
        if last_cycle.cycle.endswith("ECO"):
            msg = f"[{last_cycle}]: ECO cycle detected, using previous state to predict again."
            log.debug(msg)

            # doesn't matter if we are going autoregressive or not since
            # the state was kept from the last cycle
            self._predictor.state = self._prev_state
            return self._predictor.predict_cycle(
                cycle=last_cycle,
                save_state=True,
                use_programmed_current=self._use_programmed_current,
            )

        # no need to save the state again since the next prediction will not
        # be an ECO cycle
        if self._autoregressive:  # previous state is set already, no need to check
            msg = f"[{last_cycle}]: Autoregressive mode enabled, using previous state to predict."
            log.debug(msg)

            # save the state before prediction if we need to re-predict
            self._prev_state = self._predictor.state
            return self._predictor.predict_cycle(
                cycle=last_cycle,
                save_state=True,
                use_programmed_current=self._use_programmed_current,
            )

        msg = f"[{last_cycle}]: Autoregressive mode disabled, re-initializing state."
        log.debug(msg)

        self._predictor.set_cycled_initial_state(
            buffer[:-1], use_programmed_current=self._use_programmed_current
        )
        log.debug(f"[{last_cycle}]: Saving state for next prediction.")
        self._prev_state = self._predictor.state

        log.debug(f"[{last_cycle}]: Predicting next cycle.")
        return self._predictor.predict_cycle(
            cycle=last_cycle,
            save_state=True,
            use_programmed_current=self._use_programmed_current,
        )

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
