"""
The TrimModel is responsible for the logic of the TrimWidget.
"""

from __future__ import annotations

import logging
import typing
from datetime import datetime

import numpy as np
from op_app_context import context
from pyda import SimpleClient
from pyda.data import DiscreteFunction
from pyda_japc import JapcProvider
from pyda_lsa import LsaCycleContext, LsaEndpoint, LsaProvider
from qtpy import QtCore
from transformertf.utils import signal

from ...data import CycleData
from ...utils import ThreadWorker, time_execution

log = logging.getLogger(__name__)


DEV_LSA_B = "SPSBEAM"
PROP_LSA_B = "BHYS"
DEV_LSA_I = "MBI/IREF"

TRIM_THRESHOLD = 0.1
TRIM_SOFT_THRESHOLD = 0.01

BEAM_IN = "SIX.MC-CTML/ControlValue#controlValue"
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue#controlValue"


class TrimModel(QtCore.QObject):
    trimApplied = QtCore.Signal(tuple[np.ndarray, np.ndarray], datetime, str)
    """ Signal emitted when a trim has been applied. (corr_x, corr_y), time, comment """

    def __init__(self, parent: QtCore.QObject | None = None):
        """
        The TrimModel is responsible for the logic of the TrimWidget.

        The TrimModel works per selector. I.e. the trim can only be applied
        to one cycle at a time. The selector is set by the user in the TrimWidgetView,
        or conversely in the :meth:`TrimModel.selector` property.

        The TrimModel only trims during the time where there is beam in, and will retrieve
        the time boundary values from LSA when a new selector is selected.

        The trims are threaded.

        Parameters
        ----------
        parent : QtCore.QObject | None, optional
            The parent object, by default None
        """
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

    @QtCore.Slot(CycleData, typing.Any, name="on_new_prediction")
    def on_new_prediction(self, prediction: CycleData, *_: typing.Any) -> None:
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

        # calc delta and smooth it
        correction = prediction.field_ref - prediction.field_pred
        correction = signal.perona_malik_smooth(correction, 10.0, 5e-2, 5.0)

        time_margin = (prediction.cycle_time - datetime.now()).total_seconds()
        if time_margin < 1.0:
            log.warning(
                f"[{prediction}] Not enough time to send transaction, "
                f"skipping trim (margin {time_margin:.02f}s < 1.0s."
            )
            return

        worker = ThreadWorker(self.apply_correction, correction, prediction)
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
            current_correction_df = self.get_current_correction()

            current_time_axis = current_correction_df.xs
            current_correction = current_correction_df.ys

            # downsample to match prediction
            time_axis = correction[0, :]
            correction = correction[1, :]

            time_axis, correction = self.calc_new_correction(
                current_time_axis, current_correction, time_axis, correction
            )

            log.debug(f"[{cycle_data}] Sending trims to LSA.")

            with time_execution() as trim_time:
                self.send_trim(time_axis, correction, comment)

            trim_time_diff = trim_time.duration
            log.debug(f"Trim applied in {trim_time_diff:.02f}s.")

            self.trimApplied.emit((time_axis, correction), trim_time, comment)
        except:  # noqa E722
            log.exception("Failed to apply trim to LSA.")
            raise
        finally:
            self._trim_lock.unlock()

    def calc_new_correction(
        self,
        current_time_axis: np.ndarray,
        current_correction: np.ndarray,
        new_time_axis: np.ndarray,
        new_correction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        (
            (
                current_time_axis,
                current_correction,
            ),
            (time_axis, correction),
        ) = match_array_size(
            (current_time_axis, current_correction),
            (new_time_axis, new_correction),
        )

        # calculate correction
        new_correction = (current_correction + correction).astype(np.float64)

        # trim only part of beam that is before beam out
        time_axis, new_correction = self.truncate_beam_in(
            time_axis, new_correction
        )

        # smooth the correction
        new_correction = signal.perona_malik_smooth(
            new_correction, 10.0, 5e-2, 2.0
        )

        new_correction = truncate_correction(
            new_correction, (TRIM_SOFT_THRESHOLD, TRIM_THRESHOLD)
        )

        return time_axis, new_correction

    def truncate_beam_in(
        self, time_axis: np.ndarray, correction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        valid_indices = (self._beam_in <= time_axis) & (
            time_axis <= self._beam_out
        )
        time_axis = time_axis[valid_indices]
        correction = correction[valid_indices]

        return time_axis, correction

    def get_current_correction(self) -> DiscreteFunction:
        with time_execution() as t:
            resp_get = self._lsa.get(
                LsaEndpoint(
                    device_name=DEV_LSA_B,
                    property_name=PROP_LSA_B,
                    setting_part="CORRECTION",
                ),
                context=LsaCycleContext(cycle=self.selector),
            )
        log.debug(f"Got current trim in {t.duration:.02f}s.")
        if resp_get.exception is not None:
            raise resp_get.exception

        current_currection: DiscreteFunction = resp_get.value["correction"]
        return current_currection

    def send_trim(
        self,
        time_axis: np.ndarray,
        correction: np.ndarray,
        comment: str | None = None,
    ) -> None:
        if comment is None:
            comment = (
                "Hysteresis prediction correction " + str(datetime.now())[:-7]
            )

        func: DiscreteFunction = DiscreteFunction(time_axis, correction)
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

    @property
    def selector(self) -> str | None:
        return self._selector

    @selector.setter
    def selector(self, value: str) -> None:
        """
        Set the selector for the trim model. This will also set the beam in/out.

        Parameters
        ----------
        value : str
            The new selector.
        """
        if value == self._selector:
            log.debug(f"Selector already set to {value}, skipping.")
            return

        self._beam_in = self._da.get(BEAM_IN, context=value).value["value"]
        self._beam_out = self._da.get(BEAM_OUT, context=value).value["value"]
        self._selector = value

        log.info(f"Setting beam in/out to C{self._beam_in}/C{self._beam_out}.")

    def enable_trim(self) -> None:
        """
        Enable trim.
        """
        log.debug("Enabling trim.")
        self._trim_enabled = True

    def disable_trim(self) -> None:
        """
        Disable trim.
        """
        log.debug("Disabling trim.")
        self._trim_enabled = False


def match_array_size(
    current_correction: tuple[np.ndarray, np.ndarray],
    new_correction: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if current_correction[0].size < new_correction[0].size:
        # upsample LSA trim to match prediction
        current_correction = (
            new_correction[0],
            np.interp(
                new_correction[0], current_correction[0], current_correction[1]
            ),
        )
    elif current_correction[0].size > new_correction[0].size:
        # upsample prediction to match LSA trim
        new_correction = (
            current_correction[0],
            np.interp(
                current_correction[0], new_correction[0], new_correction[1]
            ),
        )

    return current_correction, new_correction


def truncate_correction(
    correction: np.ndarray,
    thresholds: tuple[float, float] = (TRIM_SOFT_THRESHOLD, TRIM_THRESHOLD),
) -> np.ndarray:
    """
    Truncate the correction to the given thresholds.
    Values exceeding the soft threshold will be set to the soft threshold, whereas
    values exceeding the hard threshold will raise an error.

    Parameters
    ----------
    correction : np.ndarray
        Array to truncate.
    thresholds : tuple[float, float], optional
        The thresholds, by default (TRIM_SOFT_THRESHOLD, TRIM_THRESHOLD). The first value is the soft threshold,
        the second is the hard threshold.

    Returns
    -------
    np.ndarray
        The truncated array.
    """
    if thresholds[0] < np.max(np.abs(correction)) < thresholds[1]:
        log.info(
            "Max value in correction {} ".format(np.max(np.abs(correction)))
            + " is greater than {}, but less than {}. ".format(
                thresholds[0], thresholds[1]
            )
            + "Truncating trim."
        )
        correction[correction > thresholds[0]] = thresholds[0]
        correction[correction < -thresholds[0]] = -thresholds[0]
    elif np.max(np.abs(correction)) > thresholds[1]:
        log.error(
            "Max value in correction {} is ".format(np.max(np.max(correction)))
            + f"greater than threshold {thresholds[1]}. "
            "Skipping trim."
        )
        raise ValueError("Max value in correction is greater than threshold.")

    return correction
