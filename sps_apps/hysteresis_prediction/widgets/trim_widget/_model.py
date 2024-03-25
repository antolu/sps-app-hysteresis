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
    trimApplied = QtCore.Signal(tuple, datetime, str)
    """ Signal emitted when a trim has been applied. (corr_x, corr_y), time, comment """

    beamInRetrieved = QtCore.Signal(int, int)
    """ Signal emitted when beam in/out has been retrieved. (beam_in, beam_out) """

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
        self._reference_fields: dict[str, np.ndarray] = {}

        # settable properties
        self._trim_t_min: int = 0
        self._trim_t_max: int = 100000
        self._gain: float = 1.0
        self._dry_run = False
        self._flatten = False

        # states
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

        if prediction.field_pred is None:
            raise ValueError(f"[{prediction}] No field prediction found.")

        if prediction.cycle not in self._reference_fields:
            log.info(
                f"[{prediction}] No field reference found, skipping "
                "trim but saving prediction as reference."
            )
            self._reference_fields[prediction.cycle] = prediction.field_pred
            return

        field_ref = self._reference_fields[prediction.cycle]

        # calc delta and smooth it
        correction = field_ref[1, :] - prediction.field_pred[1, :]

        time_margin = (prediction.cycle_time - datetime.now()).total_seconds()
        if time_margin < 1.0:
            log.warning(
                f"[{prediction}] Not enough time to send transaction, "
                f"skipping trim (margin {time_margin:.02f}s < 1.0s."
            )
            return

        # time axis in predictions are in timestamps, convert to ms
        time_axis = (
            prediction.field_pred[0, :] - prediction.field_pred[0, 0]
        ) * 1e3

        # trim in a new thread
        worker = ThreadWorker(
            self.apply_correction, time_axis, correction, prediction
        )
        worker.exception.connect(
            lambda e: log.exception("Failed to apply trim to LSA.:\n" + str(e))
        )

        QtCore.QThreadPool.globalInstance().start(worker)

    def apply_correction(
        self,
        correction_t: np.ndarray,
        correction_v: np.ndarray,
        cycle_data: CycleData,
    ) -> None:
        if not self._trim_lock.tryLock():
            log.warning("Already applying trim, skipping.")
            return

        comment = (
            "Hysteresis prediction correction "
            f"{str(cycle_data.cycle_time)[:-7]}"
        )

        if self.selector is None:
            # this should not happen, but just in case
            raise RuntimeError("No selector set, cannot apply trim.")

        try:
            # current_correction_df = self.get_current_correction()
            assert cycle_data.correction is not None, "No correction found."
            current_correction = cycle_data.correction
            current_time_axis, current_correction = current_correction
            # round to ms
            current_time_axis = np.round(current_time_axis, 1)

            # current_time_axis = current_correction_df.xs
            # current_correction = current_correction_df.ys

            correction, delta = self.calc_new_correction(
                current_time_axis,
                current_correction,
                correction_t,
                correction_v,
            )

            time_axis, correction = correction

            log.debug(
                f"[{cycle_data}] Sending trims to LSA with {time_axis.size} points."
            )

            if not self.dry_run:
                with time_execution() as trim_time:
                    trim_time_d = self.send_trim(
                        time_axis, correction, comment
                    )

                trim_time_diff = trim_time.duration
                log.debug(f"Trim applied in {trim_time_diff:.02f}s.")
            else:
                log.debug(
                    f"[{cycle_data}] Dry run is enabled, skipping LSA trim."
                )

                trim_time_d = datetime.now()

            self.trimApplied.emit((delta[0], delta[1]), trim_time_d, comment)
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
            (new_time_axis, new_correction),
        ) = match_array_size(
            (current_time_axis, current_correction),
            (new_time_axis, new_correction),
        )

        delta = new_correction
        delta_t = new_time_axis

        # calculate correction
        new_correction = (current_correction + new_correction).astype(
            np.float64
        )

        # trim only part of beam that is before beam out
        new_time_axis, new_correction = self.truncate_beam_in(
            new_time_axis, new_correction
        )

        # apply gain
        new_correction *= self.gain

        # smooth the correction
        # new_correction = signal.perona_malik_smooth(
        #     new_correction, 10.0, 5e-2, 2.0
        # )

        new_correction = truncate_correction(
            new_correction, (TRIM_SOFT_THRESHOLD, TRIM_THRESHOLD)
        )
        new_correction = np.stack((new_time_axis, new_correction), axis=0)

        # return delta for plotting
        delta = np.stack(self.truncate_beam_in(delta_t, delta), axis=0)
        delta[1, :] *= self.gain

        return new_correction, delta

    def truncate_beam_in(
        self, time_axis: np.ndarray, correction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = float(int(max(self._beam_in, self._trim_t_min)))
        upper = float(int(min(self._beam_out, self._trim_t_max)))

        # add lower and upper bounds to the time axis
        time_axis_new: np.ndarray = np.concatenate(
            ([lower], time_axis, [upper])  # type: ignore[arg-type]
        )
        time_axis_new = np.sort(np.unique(time_axis))
        correction = np.interp(time_axis, time_axis_new, correction)

        valid_indices = (lower <= time_axis_new) & (time_axis_new <= upper)
        time_axis_trunc = time_axis_new[valid_indices]
        correction_trunc = correction[valid_indices]

        return time_axis_trunc, correction_trunc

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
    ) -> datetime:
        now = datetime.now()
        if comment is None:
            comment = "Hysteresis prediction correction " + str(now)[:-7]

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

        return now

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
        self.beamInRetrieved.emit(self._beam_in, self._beam_out)

    @property
    def gain(self) -> float:
        return self._gain

    @gain.setter
    def gain(self, value: float) -> None:
        if value > 1.0:
            log.warning(f"Gain {value:.2f} > 1.0, this may cause instability.")
        elif value < 0.0:
            raise ValueError(f"Gain {value:.2f} < 0.0, this is not allowed.")

        log.debug(f"Setting gain to {value:.2f} for selector {self.selector}.")
        self._gain = value

    def set_gain(self, value: float) -> None:
        """
        Set the gain for the trim model.

        Useful as a Qt slot.
        """
        self.gain = value

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, value: bool) -> None:
        log.debug(f"Setting dry run to {value} for selector {self.selector}.")
        self._dry_run = value

    def set_dry_run(self, value: bool) -> None:
        """
        Set the dry run flag for the trim model.

        Useful as a Qt slot.
        """
        self.dry_run = value

    @QtCore.Slot()
    def reset_reference_fields(self) -> None:
        self._reference_fields.clear()

    def set_trim_t_min(self, value: int) -> None:
        if value < self._beam_in:
            raise ValueError(
                f"Trim t_min {value} < beam in {self._beam_in}, not allowed."
            )
        self._trim_t_min = value

    def set_trim_t_max(self, value: int) -> None:
        if value > self._beam_out:
            raise ValueError(
                f"Trim t_max {value} > beam out {self._beam_out}, not allowed."
            )
        self._trim_t_max = value

    @property
    def flatten(self) -> bool:
        return self._flatten

    @flatten.setter
    def flatten(self, value: bool) -> None:
        log.debug(f"Setting flatten to {value} for selector {self.selector}.")
        self._flatten = value

    def set_flatten(self, value: bool) -> None:
        """
        Set the flatten flag for the trim model.

        Useful as a Qt slot.
        """
        self.flatten = value

    def enable_trim(self) -> None:
        """
        Enable trim.
        """
        log.debug(f"Enabling trim for selector {self.selector}.")
        self._trim_enabled = True

    def disable_trim(self) -> None:
        """
        Disable trim.
        """
        log.debug(f"Disabling trim for selector {self.selector}")
        self._trim_enabled = False


def match_array_size(
    current_correction: tuple[np.ndarray, np.ndarray],
    new_correction: tuple[np.ndarray, np.ndarray],
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if current_correction[0].size < new_correction[0].size:
        # upsample LSA trim to match prediction, keep BP edges
        new_x = np.concatenate((new_correction[0], current_correction[0]))
        new_x = np.sort(np.unique(new_x))
        current_correction = (
            new_x,
            np.interp(
                new_x,
                *current_correction,
            ),
        )

        # upsample prediction to match new LSA trim
        new_correction = (
            new_x,
            np.interp(
                new_x,
                *new_correction,
            ),
        )
    elif current_correction[0].size > new_correction[0].size:
        # upsample prediction to match LSA trim
        new_correction = (
            current_correction[0],
            np.interp(
                current_correction[0],
                *new_correction,
            ),
        )
    else:
        new_x = np.concatenate((new_correction[0], current_correction[0]))
        new_x = np.sort(np.unique(new_x))
        current_correction = (
            new_x,
            np.interp(
                new_x,
                *current_correction,
            ),
        )

        new_correction = (
            new_x,
            np.interp(
                new_x,
                *new_correction,
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
