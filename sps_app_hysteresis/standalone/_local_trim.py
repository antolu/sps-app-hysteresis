from __future__ import annotations

import logging
import typing
from datetime import datetime

import numpy as np
import pyda
import pyda.data
from hystcomp_utils.cycle_data import CycleData
from op_app_context import context
from pyda_lsa import LsaCycleContext, LsaEndpoint
from pyda_lsa.data import TrimFlags
from qtpy import QtCore

from ..contexts import app_context
from ..settings import StandaloneTrimSettings
from ..utils import ThreadWorker, cycle_metadata, time_execution

log = logging.getLogger(__package__)


class StandaloneTrim(QtCore.QObject):
    trimApplied = QtCore.Signal(CycleData, datetime, str)
    flatteningApplied = QtCore.Signal(
        CycleData, np.ndarray, datetime, str
    )  # Keep delta for flattening

    def __init__(
        self,
        param_b_corr: str,
        settings: StandaloneTrimSettings,
        *,
        trim_threshold: float | None = None,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(parent)

        self._param_b_corr = param_b_corr
        self._settings = settings

        self._lsa = pyda.SimpleClient(provider=context.lsa_provider)
        self._trim_threshold = trim_threshold or app_context().TRIM_MIN_THRESHOLD
        self._trim_lock = QtCore.QMutex()

        # Flattening state
        self._pending_flattening: dict[str, float] = {}  # cycle -> constant_field

        # Reference to correction system for eddy current reference updates
        self._correction_system: typing.Any = None

    def set_correction_system(self, correction_system: typing.Any) -> None:
        """Set reference to correction system for eddy current reference updates."""
        self._correction_system = correction_system

    @QtCore.Slot(CycleData, name="onNewPrediction")
    def onNewPrediction(self, prediction: CycleData, *_: typing.Any) -> None:
        # Check if there's a pending flattening request for this cycle
        if prediction.cycle in self._pending_flattening:
            constant_field = self._pending_flattening.pop(prediction.cycle)
            log.info(
                f"Applying pending flattening correction for cycle {prediction.cycle}"
            )

            # Apply flattening in a new thread
            worker = ThreadWorker(
                self.apply_flattening_correction, prediction, constant_field
            )
            worker.exception.connect(
                lambda e: log.exception(
                    "Failed to apply flattening correction to LSA:\n" + str(e)
                )
            )
            QtCore.QThreadPool.globalInstance().start(worker)
            return

        # Normal trim logic
        if not self._check_trim_allowed(prediction):
            log.debug(f"Trim not allowed for {prediction}.")
            return

        # trim in a new thread
        worker = ThreadWorker(self.apply_correction, prediction)
        worker.exception.connect(
            lambda e: log.exception("Failed to apply trim to LSA:\n" + str(e))
        )

        QtCore.QThreadPool.globalInstance().start(worker)

    def _check_trim_allowed(self, cycle_data: CycleData) -> bool:
        if not self._settings.trim_enabled[cycle_data.cycle]:
            log.debug(f"[{cycle_data.cycle}] Trim is disabled, skipping trim.")
            return False

        if cycle_data.field_pred is None:
            msg = f"[{cycle_data}] No field prediction found."
            raise ValueError(msg)

        _delta_t, delta_v = cycle_data.delta_applied
        max_val = np.max(np.abs(delta_v))
        if max_val < self._trim_threshold:
            msg = f"Max value in delta {max_val:.2e} < {self._trim_threshold:e}. \
                Skipping trim on {cycle_data}"

            log.info(msg)
            return False

        time_margin = (cycle_data.cycle_time - datetime.now()).total_seconds()
        if time_margin < 1.0:
            log.warning(
                f"[{cycle_data}] Not enough time to send transaction, "
                f"skipping trim (margin {time_margin:.02f}s < 1.0s."
            )
            return False
        log.info(f"[{cycle_data}] Time margin: {time_margin:.02f}s.")

        return True

    def apply_correction(
        self,
        cycle_data: CycleData,
    ) -> None:
        if not self._trim_lock.tryLock():
            log.warning("Already applying trim, skipping.")
            return

        comment = f"Prediction correction {str(cycle_data.cycle_time)[:-7]}"

        try:
            assert cycle_data.correction_applied is not None, "No correction found."
            correction_t = cycle_data.correction_applied[0]
            correction_v = cycle_data.correction_applied[1]

            start = self._settings.trim_start[cycle_data.cycle]
            start = max(start, cycle_metadata.beam_in(cycle_data.cycle))

            end = self._settings.trim_end[cycle_data.cycle]
            end = min(end, cycle_metadata.beam_out(cycle_data.cycle))

            correction_t, correction_v = self.cut_trim_beyond_time(
                correction_t, correction_v, start, end
            )

            log.debug(
                f"[{cycle_data}] Sending trims to LSA with {correction_t.size} points."
            )

            with time_execution() as trim_time:
                trim_time_d = self.send_trim(
                    correction_t, correction_v, cycle=cycle_data.cycle, comment=comment
                )

            trim_time_diff = trim_time.duration
            log.debug(f"Trim applied in {trim_time_diff:.02f}s.")

            self.trimApplied.emit(cycle_data, trim_time_d, comment)
        except:
            log.exception("Failed to apply trim to LSA.")
            raise
        finally:
            self._trim_lock.unlock()

    def cut_trim_beyond_time(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        lower: float,
        upper: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        # add lower and upper bounds to the time axis
        time_axis_new: np.ndarray = np.concatenate(
            ([lower], xs, [upper])  # type: ignore[arg-type]
        )
        time_axis_new = np.sort(np.unique(time_axis_new))
        ys = np.interp(time_axis_new, xs, ys)

        valid_indices = (lower <= time_axis_new) & (time_axis_new <= upper)
        time_axis_trunc = time_axis_new[valid_indices]
        correction_trunc = ys[valid_indices]

        return time_axis_trunc, correction_trunc

    def send_trim(
        self,
        time_axis: np.ndarray,
        correction: np.ndarray,
        cycle: str,
        comment: str | None = None,
    ) -> datetime:
        now = datetime.now()
        if comment is None:
            comment = "Hysteresis prediction correction " + str(now)[:-7]

        msg = f"Sending trim to LSA with {time_axis}: {correction}"
        log.debug(msg)

        func: pyda.data.DiscreteFunction = pyda.data.DiscreteFunction(
            time_axis, correction
        )
        resp_set = self._lsa.set(
            endpoint=LsaEndpoint(
                device_name=self._param_b_corr.split("/")[0],
                property_name=self._param_b_corr.split("/")[1],
                setting_part="CORRECTION",
            ),
            data={"correction": func},
            context=LsaCycleContext(
                cycle=cycle,
                comment=comment,
                flags=TrimFlags(
                    transient=True,
                    drive=context.lsa_server not in {"next", "dev"},
                    propagate_to_children=context.lsa_server not in {"next", "dev"},
                ),
            ),
        )

        if resp_set.exception is not None:
            raise resp_set.exception

        return now

    @QtCore.Slot(str, float, name="onFlatteningRequested")
    def onFlatteningRequested(self, cycle: str, constant_field: float) -> None:
        """Schedule flattening correction for the next prediction of the given cycle."""
        log.info(
            f"Scheduling flattening correction for cycle {cycle} with constant field {constant_field}"
        )
        self._pending_flattening[cycle] = constant_field

    def apply_flattening_correction(
        self,
        cycle_data: CycleData,
        constant_field: float,
    ) -> None:
        """Apply flattening correction by computing difference between prediction and constant field."""
        if not self._trim_lock.tryLock():
            log.warning("Already applying trim, skipping flattening correction.")
            return

        comment = f"Flattening correction (constant={constant_field:.2f}) {str(cycle_data.cycle_time)[:-7]}"

        try:
            assert cycle_data.field_pred is not None, (
                "No field prediction found for flattening."
            )

            # Get the predicted field
            pred_time = cycle_data.field_pred[0]
            pred_field = cycle_data.field_pred[1]

            # Compute the difference from constant field
            field_diff = pred_field - constant_field

            # Apply gain to the difference
            gain = self._settings.gain[cycle_data.cycle]
            correction_v = gain * field_diff

            # Add to existing current correction if it exists
            if cycle_data.correction is not None:
                current_corr_time = cycle_data.correction[0]
                current_corr_v = cycle_data.correction[1]

                # Interpolate existing correction to prediction time grid
                existing_correction = np.interp(
                    pred_time, current_corr_time, current_corr_v
                )
                correction_v = correction_v + existing_correction

            # Apply time limits
            start = self._settings.trim_start[cycle_data.cycle]
            start = max(start, cycle_metadata.beam_in(cycle_data.cycle))

            end = self._settings.trim_end[cycle_data.cycle]
            end = min(end, cycle_metadata.beam_out(cycle_data.cycle))

            correction_t, correction_v = self.cut_trim_beyond_time(
                pred_time, correction_v, start, end
            )

            log.debug(
                f"[{cycle_data}] Sending flattening correction to LSA with {correction_t.size} points."
            )

            with time_execution() as trim_time:
                trim_time_d = self.send_trim(
                    correction_t, correction_v, cycle=cycle_data.cycle, comment=comment
                )

            trim_time_diff = trim_time.duration
            log.debug(f"Flattening correction applied in {trim_time_diff:.02f}s.")

            # Calculate delta for plotting
            delta = np.vstack((correction_t, correction_v))

            self.flatteningApplied.emit(cycle_data, delta, trim_time_d, comment)

        except:
            log.exception("Failed to apply flattening correction to LSA.")
            raise
        finally:
            self._trim_lock.unlock()
