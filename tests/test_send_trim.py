from __future__ import annotations

import pytest
import numpy as np
import logging
from op_app_context import context
import datetime


from sps_apps.hysteresis_prediction.widgets.trim_widget import TrimModel
from sps_apps.hysteresis_prediction.data import CycleData

context.lsa_server = "next"

log = logging.getLogger()
log.addHandler(logging.StreamHandler())


SELECTOR = "SPS.USER.MD1"
DOWNSAMPLE = 50


CYCLE_DATA = CycleData(
    cycle="test",
    user=SELECTOR,
    cycle_timestamp=datetime.datetime.now().timestamp() * 1e9,
    current_prog=np.zeros((2, 3600 + 1)),
    field_prog=np.zeros((2, 3600 + 1)),
)


@pytest.mark.uses_virtual_device
def test_send_trim() -> None:
    trim = TrimModel()
    trim.selector = SELECTOR

    xs = np.arange(0, 3600 + 1, 1)[::DOWNSAMPLE].astype(np.float64)
    ys = np.zeros_like(xs) + 1e-5

    trim.apply_correction(
        correction_t=xs, correction_v=ys, cycle_data=CYCLE_DATA
    )


@pytest.mark.uses_virtual_device
def test_send_trim_boundaries() -> None:
    trim = TrimModel()
    trim.selector = SELECTOR

    trim.set_trim_t_max(2000)
    trim.set_trim_t_min(1200)

    xs = np.arange(0, 3600 + 1, 1)[::DOWNSAMPLE].astype(np.float64)
    ys = np.zeros_like(xs) + 1e-5

    trim.apply_correction(
        correction_t=xs, correction_v=ys, cycle_data=CYCLE_DATA
    )
