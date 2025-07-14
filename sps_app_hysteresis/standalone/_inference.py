from __future__ import annotations

import enum
import functools
import logging
import typing

import mlp_client
import numpy as np
import pyda.access
from hystcomp_utils.cycle_data import CycleData, EconomyMode
from qtpy import QtCore, QtWidgets
from sps_mlp_hystcomp import (
    EddyCurrentPredictor,
    PETEPredictor,
    PFTFTPredictor,
    TFTPredictor,
)

from ..utils import ThreadWorker, load_cursor, run_in_thread, time_execution
from .event_building import EventBuilderAbc

MS = int(1e3)
NS = int(1e9)

log = logging.getLogger(__name__)

_thread: QtCore.QThread | None = None


def _mlp_client() -> mlp_client.Client:
    """Get the MLP client for the current profile."""
    return mlp_client.Client(profile=mlp_client.Profile.PRO)


def inference_thread() -> QtCore.QThread:
    global _thread  # noqa: PLW0603
    if _thread is None:
        _thread = QtCore.QThread()
    return _thread


class PredictionMode(enum.Enum):
    """Prediction mode enumeration."""

    COMBINED = "combined"  # Hysteresis + Eddy Current
    HYSTERESIS_ONLY = "hysteresis_only"  # Hysteresis predictions only
    EDDY_CURRENT_ONLY = "eddy_current_only"  # Eddy current predictions only


class InferenceFlags:
    """Flags for the inference module."""

    def __init__(self) -> None:
        self._do_inference = False
        self._autoregressive = False
        self._use_programmed_current = True
        self._prediction_mode = PredictionMode.COMBINED

    def set_do_inference(self, state: bool) -> None:  # noqa: FBT001
        self._do_inference = state

    def set_autoregressive(self, state: bool) -> None:  # noqa: FBT001
        self._autoregressive = state

    @property
    def autoregressive(self) -> bool:
        return self._autoregressive

    @autoregressive.setter
    def autoregressive(self, value: bool) -> None:
        self.set_autoregressive(value)

    def set_use_programmed_current(self, *, state: bool) -> None:
        self._use_programmed_current = state

    @property
    def use_programmed_current(self) -> bool:
        return self._use_programmed_current

    @use_programmed_current.setter
    def use_programmed_current(self, value: bool) -> None:
        self.set_use_programmed_current(state=value)

    def set_prediction_mode(self, mode: PredictionMode) -> None:
        """Set the prediction mode."""
        self._prediction_mode = mode

    @property
    def prediction_mode(self) -> PredictionMode:
        """Get the current prediction mode."""
        return self._prediction_mode

    @prediction_mode.setter
    def prediction_mode(self, mode: PredictionMode) -> None:
        """Set the prediction mode."""
        self.set_prediction_mode(mode)

    # Keep backward compatibility for now
    @property
    def eddy_current(self) -> bool:
        """Legacy property for backward compatibility."""
        return self._prediction_mode in (
            PredictionMode.COMBINED,
            PredictionMode.EDDY_CURRENT_ONLY,
        )

    @eddy_current.setter
    def eddy_current(self, value: bool) -> None:
        """Legacy setter for backward compatibility."""
        if value:
            self._prediction_mode = PredictionMode.COMBINED
        else:
            self._prediction_mode = PredictionMode.HYSTERESIS_ONLY


class Inference(InferenceFlags, EventBuilderAbc):
    model_loaded = QtCore.Signal()

    predictionStarted = QtCore.Signal()
    predictionFinished = QtCore.Signal()

    def __init__(
        self,
        device: typing.Literal["cpu", "cuda", "auto"] = "cpu",
        parent: QtCore.QObject | None = None,
    ) -> None:
        EventBuilderAbc.__init__(self, parent=parent)
        InferenceFlags.__init__(self)

        self._e_predictor = EddyCurrentPredictor()
        self._predictor: PETEPredictor | TFTPredictor | None = None

        self._prev_state: typing.Any | None = None
        self._prev_e_state: typing.Any | None = None

    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        pass

    def load_eddy_current_model(
        self,
        model_name: str,
        model_version: str,
    ) -> None:
        client = mlp_client.Client(profile=mlp_client.Profile.PRO)

        log.debug(f"Loading Eddy Current model {model_name} version {model_version}.")
        self._e_predictor = typing.cast(
            EddyCurrentPredictor,
            client.create_model(
                EddyCurrentPredictor,
                params_name=model_name,
                params_version=model_version,
            ),
        )

        log.info(f"Loaded eddy current model with {len(self._e_predictor.C)}")

    @QtCore.Slot(str, str, str)
    def loadLocalModel(
        self, model_name: str, ckpt_path: str, device: str = "cpu"
    ) -> None:
        worker = ThreadWorker(self._on_load_local_model, model_name, ckpt_path, device)

        def on_exception(e: Exception) -> None:
            log.exception("Error loading model.")
            QtWidgets.QMessageBox.critical(
                None, "Error loading model", str(e), QtWidgets.QMessageBox.Ok
            )

        worker.exception.connect(on_exception)

        QtCore.QThreadPool.globalInstance().start(worker)

    @QtCore.Slot(str, str, str, str)
    def loadMlpModel(
        self,
        model_name: str,
        params_name: str,
        params_version: str,
        device: str = "cpu",
    ) -> None:
        worker = ThreadWorker(
            self._on_load_mlp_model, model_name, params_name, params_version, device
        )

        def on_exception(e: Exception) -> None:
            log.exception("Error loading model.")
            QtWidgets.QMessageBox.critical(
                None, "Error loading model", str(e), QtWidgets.QMessageBox.Ok
            )

        worker.exception.connect(on_exception)

        QtCore.QThreadPool.globalInstance().start(worker)

    def _on_load_mlp_model(
        self,
        model_name: str,
        params_name: str,
        params_version: str,
        device: typing.Literal["cpu", "cuda", "auto"] = "cpu",
    ) -> None:
        predictor_cls = resolve_predictor_cls(model_name)

        with load_cursor():
            client = _mlp_client()
            try:
                predictor = typing.cast(
                    PETEPredictor | TFTPredictor | PFTFTPredictor,
                    client.create_model(predictor_cls, params_name, params_version),
                )
                predictor.device = device
                predictor.prog_t_phase = 0.1535 * 1e-3
            except Exception:
                log.exception("Error occurred while loading MLP model.")
                raise

            self._predictor = predictor

        self.model_loaded.emit()

    def _on_load_local_model(
        self,
        model_name: str,
        ckpt_path: str,
        device: typing.Literal["cpu", "cuda", "auto"] = "cpu",
    ) -> None:
        predictor_cls = resolve_predictor_cls(model_name)
        with load_cursor():
            try:
                self._predictor = predictor_cls.load_from_checkpoint(
                    ckpt_path, device=device
                )
                self._predictor.prog_t_phase = 0.1535 * 1e-3
            except:
                log.exception("Error occurred.")
                raise

        self.model_loaded.emit()

    def onNewCycleDataBuffer(self, cycle_data: list[CycleData]) -> None:
        return self.predict_last_cycle(cycle_data)

    def predict_last_cycle(self, cycle_data: list[CycleData]) -> None:
        if self._predictor is None:
            log.debug("Model not loaded. Cannot predict.")
            return

        if not self._do_inference:
            log.debug("Inference is disabled. Not predicting.")
            return

        if self._predictor.busy:
            log.warning("Inference is already underway. Cannot do more in parallel.")
            return

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
            if last_cycle.economy_mode is EconomyMode.DYNAMIC:
                msg = (
                    f"[{last_cycle}]: ECO cycle detected, but previous state is not set."
                    f"This should not happen."
                )
                log.debug(msg)
                raise ValueError(msg)

            # run first prediction
            self._e_predictor.set_cycled_initial_state(buffer[:-1])
            self._predictor.set_cycled_initial_state(
                buffer[:-1],
                use_programmed_current=self._use_programmed_current,
            )
            self._prev_state = self._predictor.state
            self._prev_e_state = self._e_predictor.state

            return predict_cycle(
                cycle=last_cycle,
                predictor=self._predictor,
                e_predictor=self._e_predictor,
                use_programmed_current=self._use_programmed_current,
                prediction_mode=self._prediction_mode,
            )

        if last_cycle.economy_mode is EconomyMode.DYNAMIC:
            msg = f"[{last_cycle}]: ECO cycle detected, using previous state to predict again."
            log.debug(msg)

            # doesn't matter if we are going autoregressive or not since
            # the state was kept from the last cycle
            self._predictor.state = self._prev_state
            self._e_predictor.state = self._prev_e_state

            return predict_cycle(
                cycle=last_cycle,
                predictor=self._predictor,
                e_predictor=self._e_predictor,
                use_programmed_current=self._use_programmed_current,
                prediction_mode=self._prediction_mode,
            )

        # no need to save the state again since the next prediction will not
        # be an ECO cycle
        if self._autoregressive:  # previous state is set already, no need to check
            msg = f"[{last_cycle}]: Autoregressive mode enabled, using previous state to predict."
            log.debug(msg)

            # save the state before prediction if we need to re-predict
            self._prev_state = self._predictor.state
            t_pred, b_pred = self._predictor.predict_cycle(
                cycle=last_cycle,
                save_state=True,
                use_programmed_current=self._use_programmed_current,
                prediction_mode=self._prediction_mode,
            )

            t_e_pred, b_e_pred = self._e_predictor.predict_cycle(
                cycle=last_cycle,
                save_state=True,
            )

            b_e_pred_interp = np.interp(
                t_pred,
                t_e_pred,
                b_e_pred.flatten(),
            )
            return np.vstack((t_pred, b_pred + b_e_pred_interp))

        msg = f"[{last_cycle}]: Autoregressive mode disabled, re-initializing state."
        log.debug(msg)

        self._predictor.set_cycled_initial_state(
            buffer[:-1], use_programmed_current=self._use_programmed_current
        )
        log.debug(f"[{last_cycle}]: Saving state for next prediction.")
        self._prev_state = self._predictor.state

        self._e_predictor.set_cycled_initial_state(buffer[:-1])
        self._prev_e_state = self._e_predictor.state

        log.debug(f"[{last_cycle}]: Predicting next cycle.")
        return predict_cycle(
            cycle=last_cycle,
            predictor=self._predictor,
            e_predictor=self._e_predictor,
            use_programmed_current=self._use_programmed_current,
            prediction_mode=self._prediction_mode,
        )

    @property
    def model_is_loaded(self) -> bool:
        return self._predictor is not None and self._predictor._module is not None  # noqa: SLF001

    def reset_state(self) -> None:
        if self._predictor is not None:
            self._predictor.reset_state()
        else:
            log.error("Model not loaded. Cannot reset state.")

        self._e_predictor.reset_state()


def predict_cycle(
    cycle: CycleData,
    predictor: PETEPredictor | TFTPredictor,
    e_predictor: EddyCurrentPredictor | None = None,
    *,
    use_programmed_current: bool = True,
    prediction_mode: PredictionMode = PredictionMode.COMBINED,
) -> np.ndarray:
    """Predict the field for the given cycle and store separated components.

    :param cycle: The cycle to predict the field for.
    :param predictor: The predictor to use.
    :param e_predictor: The EddyCurrentPredictor to use for the eddy current prediction.
    :param use_programmed_current: Whether to use programmed current or not.
    :param prediction_mode: The prediction mode to use.

    :return: The predicted field of the cycle. Time axis in seconds, field in T.
    """
    log.debug(f"[{cycle}]: Predicting field with mode {prediction_mode.value}")

    # Always compute hysteresis prediction when predictor is available
    t_hyst, b_hyst = None, None
    if predictor is not None:
        t_hyst, b_hyst = predictor.predict_cycle(
            cycle=cycle,
            save_state=True,
            use_programmed_current=use_programmed_current,
        )
        cycle.field_pred_hyst = np.vstack((t_hyst, b_hyst))
        log.debug(
            f"[{cycle}]: Hysteresis prediction computed with shape {cycle.field_pred_hyst.shape}"
        )

    # Always compute eddy current prediction when predictor is available
    t_eddy, b_eddy = None, None
    if e_predictor is not None:
        t_eddy, b_eddy = e_predictor.predict_cycle(
            cycle=cycle,
            save_state=True,
        )
        cycle.field_pred_eddy = np.vstack((t_eddy, b_eddy.flatten()))
        log.debug(
            f"[{cycle}]: Eddy current prediction computed with shape {cycle.field_pred_eddy.shape}"
        )

    # Set field_pred based on prediction mode for legacy support
    if prediction_mode == PredictionMode.EDDY_CURRENT_ONLY:
        log.debug(f"[{cycle}]: Using EDDY_CURRENT_ONLY prediction mode.")
        if cycle.field_pred_eddy is None:
            msg = "Eddy current predictor is required for EDDY_CURRENT_ONLY mode"
            raise ValueError(msg)
        cycle.field_pred = cycle.field_pred_eddy
        return cycle.field_pred_eddy

    if prediction_mode == PredictionMode.HYSTERESIS_ONLY:
        log.debug(f"[{cycle}]: Using HYSTERESIS_ONLY prediction mode.")
        if cycle.field_pred_hyst is None:
            msg = "Hysteresis predictor is required for HYSTERESIS_ONLY mode"
            raise ValueError(msg)
        cycle.field_pred = cycle.field_pred_hyst
        return cycle.field_pred_hyst

    # PredictionMode.COMBINED
    log.debug(f"[{cycle}]: Using COMBINED prediction mode.")

    if cycle.field_pred_hyst is None:
        msg = "Hysteresis predictor is required for COMBINED mode"
        raise ValueError(msg)

    if cycle.field_pred_eddy is None:
        log.warning(
            "Eddy current predictor is not provided, using hysteresis prediction only."
        )
        cycle.field_pred = cycle.field_pred_hyst
        return cycle.field_pred_hyst

    # Interpolate eddy current prediction to hysteresis time grid
    b_eddy_interp = np.interp(
        cycle.field_pred_hyst[0],
        cycle.field_pred_eddy[0],
        cycle.field_pred_eddy[1],
    )

    # Combine predictions
    combined_prediction = np.vstack((
        cycle.field_pred_hyst[0],
        cycle.field_pred_hyst[1] + b_eddy_interp,
    ))

    cycle.field_pred = combined_prediction
    log.debug(
        f"[{cycle}]: Combined prediction created with shape {cycle.field_pred.shape}"
    )

    return combined_prediction


@functools.lru_cache
def resolve_predictor_cls(
    model_name: str,
) -> type[EddyCurrentPredictor | PETEPredictor | TFTPredictor]:
    if model_name == "PETE":
        return PETEPredictor
    if model_name == "TemporalFusionTransformer":
        return TFTPredictor
    if model_name == "PFTFT":
        return PFTFTPredictor
    msg = f"Unknown model name: {model_name}"
    log.exception(msg)
    raise ValueError(msg)
