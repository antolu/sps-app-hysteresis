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

from hystcomp_utils.cycle_data import CycleData
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
    trimApplied = QtCore.Signal(np.ndarray, datetime, str)
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
        token = context.rbac_token
        if token is None:
            import pyrbac

            token = pyrbac.AuthenticationClient().login_location()
        self._lsa = SimpleClient(
            provider=LsaProvider(server=context.lsa_server, rbac_token=token)
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

    @QtCore.Slot(CycleData, name="onNewPrediction")
    def onNewPrediction(self, prediction: CycleData, *_: typing.Any) -> None:
        if not self._trim_enabled:
            log.debug("Trim is disabled, skipping trim.")
            return

        if self.selector is None:
            log.debug("No selector set, skipping trim.")
            return

        if self.selector != prediction.user:
            log.debug(
                f"Selector {self.selector} != {prediction.user}, "
                "skipping trim."
            )
            return

        if prediction.field_pred is None:
            raise ValueError(f"[{prediction}] No field prediction found.")

        delta_t, delta_v = prediction.delta_applied

        max_val = np.max(np.abs(delta_v))
        if max_val < 5e-6:
            msg = f"Max value in delta {max_val:.2e} < 5e-6. Skipping trim on {prediction}"
            log.info(msg)
            return

        time_margin = (prediction.cycle_time - datetime.now()).total_seconds()
        if time_margin < 1.0:
            log.warning(
                f"[{prediction}] Not enough time to send transaction, "
                f"skipping trim (margin {time_margin:.02f}s < 1.0s."
            )
            return
        else:
            log.info(f"[{prediction}] Time margin: {time_margin:.02f}s.")

        # trim in a new thread
        worker = ThreadWorker(self.apply_correction, prediction)
        worker.exception.connect(
            lambda e: log.exception("Failed to apply trim to LSA.:\n" + str(e))
        )

        QtCore.QThreadPool.globalInstance().start(worker)

    def apply_correction(
        self,
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
            assert cycle_data.correction is not None, "No correction found."
            correction_t = cycle_data.correction[0]
            correction_v = cycle_data.correction[1]
            # log shapes
            log.info(
                f"[{cycle_data}] Correction shape: {correction_t.shape}, {correction_v.shape}"
            )

            correction_t, correction_v = self.cut_trim_beyond_time(
                correction_t, correction_v
            )

            assert (
                cycle_data.delta_applied is not None
            ), "No delta applied found."
            delta_t, delta_v = self.cut_trim_beyond_time(
                *cycle_data.delta_applied
            )

            log.debug(
                f"[{cycle_data}] Sending trims to LSA with {correction_t.size} points."
            )

            if not self.dry_run:
                with time_execution() as trim_time:
                    trim_time_d = self.send_trim(
                        correction_t, correction_v, comment
                    )

                trim_time_diff = trim_time.duration
                log.debug(f"Trim applied in {trim_time_diff:.02f}s.")
            else:
                log.debug(
                    f"[{cycle_data}] Dry run is enabled, skipping LSA trim."
                )

                trim_time_d = datetime.now()

            self.trimApplied.emit(
                np.vstack((delta_t, delta_v)), trim_time_d, comment
            )
        except:  # noqa E722
            log.exception("Failed to apply trim to LSA.")
            raise
        finally:
            self._trim_lock.unlock()

    def cut_trim_beyond_time(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        lower: float | None = None,
        upper: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower = lower or self.trim_t_start
        upper = upper or self.trim_t_end

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

    @property
    def trim_t_start(self) -> float:
        return float(int(max(self._beam_in, self._trim_t_min)))

    @property
    def trim_t_end(self) -> float:
        return float(int(min(self._beam_out, self._trim_t_max)))

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
