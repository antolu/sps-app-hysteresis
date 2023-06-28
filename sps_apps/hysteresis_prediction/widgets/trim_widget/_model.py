from __future__ import annotations

import logging
import time
import typing
from datetime import datetime

import numpy as np
import numpy.typing as npt
from pyda import SimpleClient
from pyda.data import DiscreteFunction
from pyda_japc import JapcProvider
from pyda_lsa import LsaCycleContext, LsaEndpoint, LsaProvider
from qtpy import QtCore
from sps_projects.hysteresis_compensation.utils import signal

from ...core.application_context import context
from ...data import CycleData
from ...utils import ThreadWorker

log = logging.getLogger(__name__)


DEV_LSA_B = "SPSBEAM"
PROP_LSA_B = "BHYS"
DEV_LSA_I = "MBI/IREF"

TRIM_THRESHOLD = 0.1
TRIM_SOFT_THRESHOLD = 0.01

BEAM_IN = "SIX.MC-CTML/ControlValue#controlValue"
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue#controlValue"


class time_execution:
    """
    Convenience class for timing execution. Used simply as
    >>> with time_execution() as t:
    >>>     # some code to time
    >>> print(t.duration)
    """

    def __init__(self):
        self.start = 0
        self.end = 0
        self.duration = 0

    def __enter__(self):
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.end = time.time()
        self.duration = self.end - self.start


class TrimModel(QtCore.QObject):
    newPredictedData = QtCore.Signal(CycleData, np.ndarray)

    trimApplied = QtCore.Signal(tuple, datetime, str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent=parent)

        # to get beam in/out
        self._da = SimpleClient(provider=JapcProvider())

        # to send trims
        self._lsa = SimpleClient(
            provider=LsaProvider(server=context.lsa_server)
        )

        self._beam_in: int = 0
        self._beam_out: int = 0
        self._selector: str | None = None

        self._trim_enabled = False

        self._trim_lock = QtCore.QMutex()

        self.newPredictedData.connect(self.on_new_prediction)

    def on_new_prediction(self, prediction: CycleData, *_) -> None:
        if not self._trim_enabled:
            log.debug("Trim is disabled, skipping trim.")
            return

        if self._selector is None:
            log.debug("No selector set, skipping trim.")
            return

        if self._selector != prediction.user:
            log.debug(
                f"Selector {self._selector} != {prediction.user}, "
                "skipping trim."
            )
            return

        if prediction.field_ref is None:
            log.info(
                f"[{prediction}] No field reference found, skipping trim."
            )
            return

        if prediction.field_pred is None:
            raise ValueError(f"[{prediction}] No field prediction found.")

        correction = prediction.field_ref - prediction.field_pred

        correction = signal.perona_malik_smooth(correction, 10.0, 5e-2, 5.0)

        time_margin = (prediction.cycle_time - datetime.now()).total_seconds()
        if time_margin < 1.0:
            log.warning(
                f"[{prediction}] Not enough time to send transaction, "
                f"skipping trim (margin {time_margin:.02f}s < 1.0s."
            )
            return

        worker = ThreadWorker(
            self.apply_correction, correction[1, :], prediction
        )
        worker.exception.connect(
            lambda e: log.exception("Failed to apply trim to LSA.")
        )

        QtCore.QThreadPool.globalInstance().start(worker)

    def apply_correction(
        self, correction: np.ndarray, cycle_data: CycleData
    ) -> None:
        if not self._trim_lock.tryLock():
            log.warning("Already applying trim, skipping.")
            return

        comment = (
            "Hysteresis prediction correction "
            f"{str(cycle_data.cycle_time)[:-7]}"
        )

        if self.selector is None:
            return

        try:
            with time_execution() as t:
                resp_get = self._lsa.get(
                    LsaEndpoint(
                        device_name=DEV_LSA_B,
                        property_name=PROP_LSA_B,
                        setting_part="CORRECTION",
                    ),
                    context=LsaCycleContext(cycle=self.selector),
                )
            log.info(f"Got current trim in {t.duration:.02f}s.")
            if resp_get.exception is not None:
                raise resp_get.exception

            current_currection: DiscreteFunction[np.float64] = resp_get.value[
                "correction"
            ]

            lsa_time_axis: npt.NDArray[np.float64] = current_currection.xs
            current_correction: npt.NDArray[np.float64] = current_currection.ys

            # downsample to match prediction
            downsample = cycle_data.num_samples // len(correction)
            time_axis = np.arange(cycle_data.num_samples)[::downsample].astype(
                np.float64
            )

            # trim only part of beam that is before bema out
            valid_indices = (self._beam_in <= lsa_time_axis) & (
                lsa_time_axis <= self._beam_out
            )
            lsa_time_axis = lsa_time_axis[valid_indices]
            current_correction = current_correction[valid_indices]

            valid_indices_new = (self._beam_in <= time_axis) & (
                time_axis <= self._beam_out
            )
            time_axis = time_axis[valid_indices_new]
            correction = correction[valid_indices_new]

            if lsa_time_axis.size < time_axis.size:
                # upsample LSA trim to match prediction
                current_correction = np.interp(
                    time_axis, lsa_time_axis, current_correction
                )
                lsa_time_axis = time_axis
            elif lsa_time_axis.size > time_axis.size:
                # upsample prediction to match LSA trim
                correction = np.interp(lsa_time_axis, time_axis, correction)

            new_correction = (current_correction + correction).astype(
                np.float64
            )

            log.info(f"[{cycle_data}] Sending trims to LSA.")
            trim_time = datetime.now()

            new_correction = signal.perona_malik_smooth(
                new_correction, 10.0, 5e-2, 2.0
            )
            if (
                TRIM_SOFT_THRESHOLD
                < np.max(np.abs(new_correction))
                < TRIM_THRESHOLD
            ):
                log.info(
                    "Max value in correction {} ".format(
                        np.max(np.abs(new_correction))
                    )
                    + " is greater than {}, but less than {}. ".format(
                        TRIM_SOFT_THRESHOLD, TRIM_THRESHOLD
                    )
                    + "Truncating trim."
                )
                new_correction[
                    new_correction > TRIM_SOFT_THRESHOLD
                ] = TRIM_SOFT_THRESHOLD
                new_correction[
                    new_correction < -TRIM_SOFT_THRESHOLD
                ] = -TRIM_SOFT_THRESHOLD
            elif np.max(np.abs(new_correction)) > TRIM_THRESHOLD:
                log.error(
                    "Max value in correction {} is ".format(
                        np.max(np.max(new_correction))
                    )
                    + f"greater than threshold {TRIM_THRESHOLD}. "
                    "Skipping trim."
                )
                return

            lsa_time_axis = typing.cast(npt.NDArray[np.float64], lsa_time_axis)
            func: DiscreteFunction[np.float64] = DiscreteFunction(
                lsa_time_axis, new_correction
            )
            resp_set = self._lsa.set(
                endpoint=LsaEndpoint(
                    device_name=DEV_LSA_B,
                    property_name=PROP_LSA_B,
                    setting_part="CORRECTION",
                ),
                value={"correction": func},
                context=LsaCycleContext(
                    cycle=self.selector, comment=comment, transient=True
                ),
            )

            if resp_set.exception is not None:
                raise resp_set.exception

            trim_time_diff = (datetime.now() - trim_time).total_seconds()
            log.info(f"Trim applied in {trim_time_diff:.02f}s.")

            print(f"Shape: {time_axis.shape}, {new_correction.shape}")
            self.trimApplied.emit(
                (lsa_time_axis, new_correction), trim_time, comment
            )
        except:  # noqa E722
            log.exception("Failed to apply trim to LSA.")
            raise
        finally:
            self._trim_lock.unlock()

    @property
    def selector(self) -> str | None:
        return self._selector

    @selector.setter
    def selector(self, value: str) -> None:
        if value == self._selector:
            log.debug(f"Selector already set to {value}, skipping.")
            return

        self._beam_in = self._da.get(BEAM_IN, context=value).value["value"]
        self._beam_out = self._da.get(BEAM_OUT, context=value).value["value"]
        self._selector = value

        log.info(f"Setting beam in/out to C{self._beam_in}/C{self._beam_out}.")

    def enable_trim(self) -> None:
        log.debug("Enabling trim.")
        self._trim_enabled = True

    def disable_trim(self) -> None:
        log.debug("Disabling trim.")
        self._trim_enabled = False
