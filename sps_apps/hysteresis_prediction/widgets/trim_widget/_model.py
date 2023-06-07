from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
from pyda import SimpleClient
from pyda_japc import JapcProvider
from qtpy import QtCore

from ...core.application_context import context
from ...data import SingleCycleData
from ...utils import TrimManager

log = logging.getLogger(__name__)


DEV_LSA_B = "SPSBEAM/B"

BEAM_IN = "SIX.MC-CTML/ControlValue#controlValue"
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue#controlValue"


class TrimModel(QtCore.QObject):
    newPredictedData = QtCore.Signal(SingleCycleData, np.ndarray)

    trimApplied = QtCore.Signal(tuple, datetime, str)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent=parent)

        self._trim_manager = TrimManager(context.lsa)

        self._da = SimpleClient(provider=JapcProvider())

        self._beam_in: int = 0
        self._beam_out: int = 0
        self._selector: str | None = None

        self._trim_enabled = False

        self.newPredictedData.connect(self.on_new_prediction)

    def on_new_prediction(self, prediction: SingleCycleData, *_) -> None:
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

        time_margin = (prediction.cycle_time - datetime.now()).total_seconds()
        if time_margin < 1.0:
            log.warning(
                f"[{prediction}] Not enough time to send transaction, "
                f"skipping trim (margin {time_margin:.02f}s < 1.0s."
            )
            return

        self.apply_correction(correction[1, :], prediction)

    def apply_correction(
        self, correction: np.ndarray, cycle_data: SingleCycleData
    ) -> None:
        comment = (
            "Hysteresis prediction correction "
            f"{str(cycle_data.cycle_time)[:-7]}"
        )

        try:
            self._trim_manager.active_context
        except ValueError:
            log.warning("No active context, skipping trim.")
            return

        current_currection = self._trim_manager.get_current_trim(
            DEV_LSA_B, part="CORRECTION"
        )

        lsa_time_axis = current_currection[0]
        current_value = current_currection[1]

        downsample = cycle_data.num_samples // len(correction)

        time_axis = np.arange(cycle_data.num_samples)[: self._beam_out + 1][
            ::downsample
        ]
        correction = correction[round(self._beam_out / downsample) + 1]

        if lsa_time_axis.size < time_axis.size:
            # upsample LSA trim to match prediction
            current_value = np.interp(time_axis, lsa_time_axis, current_value)
        elif lsa_time_axis.size > time_axis.size:
            # upsample prediction to match LSA trim
            correction = np.interp(lsa_time_axis, time_axis, correction)
            time_axis = np.interp(lsa_time_axis, time_axis, time_axis)

        new_correction = current_value + correction

        log.debug(f"[{cycle_data}] Sending trims to LSA.")
        trim_time = datetime.now()

        if False:
            self._trim_manager.send_trim(
                DEV_LSA_B,
                values=(time_axis, new_correction),
                comment=comment,
                part="CORRECTION",
            )
        else:
            log.info("Debug environment, skipping trim.")

        self.trimApplied.emit((time_axis, new_correction), trim_time, comment)

    @property
    def selector(self) -> str | None:
        return self._selector

    @selector.setter
    def selector(self, value: str) -> None:
        self._trim_manager.active_context = value

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
