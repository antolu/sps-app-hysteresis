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
from ..settings import LocalTrimSettings
from ..utils import ThreadWorker, cycle_metadata, time_execution

log = logging.getLogger(__package__)


class LocalTrim(QtCore.QObject):
    trimApplied = QtCore.Signal(CycleData, np.ndarray, datetime, str)

    def __init__(
        self,
        param_b_corr: str,
        settings: LocalTrimSettings,
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

    @QtCore.Slot(CycleData, name="onNewPrediction")
    def onNewPrediction(self, prediction: CycleData, *_: typing.Any) -> None:
        if not self._check_trim_allowed(prediction):
            log.debug(f"Trim not allowed for {prediction}.")
            return

        # trim in a new thread
        worker = ThreadWorker(self.apply_correction, prediction)
        worker.exception.connect(
            lambda e: log.exception("Failed to apply trim to LSA.:\n" + str(e))
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

        comment = f"Hysteresis prediction correction {str(cycle_data.cycle_time)[:-7]}"

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

            # calculating the deltas is for purely plotting purposes
            # any real usage should still be done with the CycleData.correction_applied field
            delta = self.calc_delta(cycle_data, start, end)

            self.trimApplied.emit(cycle_data, delta, trim_time_d, comment)
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

    def calc_delta(self, cycle_data: CycleData, start: float, end: float) -> np.ndarray:
        assert cycle_data.delta_applied is not None, "No delta applied found."
        delta_t, delta_v = self.cut_trim_beyond_time(
            cycle_data.delta_applied[0],
            cycle_data.delta_applied[1],
            lower=start,
            upper=end,
        )

        delta_v = self._settings.gain[cycle_data.cycle] * delta_v

        return np.vstack((delta_t, delta_v))
