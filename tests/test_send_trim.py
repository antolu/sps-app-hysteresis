from __future__ import annotations

import copy
import datetime
import logging

import numpy as np
import pytest
from op_app_context import context

from sps_apps.hysteresis_prediction.data import CycleData
from sps_apps.hysteresis_prediction.widgets.trim_widget import TrimModel

context.lsa_server = "next"  # WARNING

log = logging.getLogger()
log.addHandler(logging.StreamHandler())


SELECTOR = "SPS.USER.SFTPRO1"
DOWNSAMPLE = 50


CYCLE_DATA = CycleData(
    cycle="test",
    user=SELECTOR,
    cycle_timestamp=datetime.datetime.now().timestamp() * 1e9,
    current_prog=np.zeros((2, 10800 + 1)),
    field_prog=np.zeros((2, 10800 + 1)),
)


@pytest.mark.uses_virtual_device
def test_send_trim() -> None:
    trim = TrimModel()
    trim.selector = SELECTOR

    xs = np.arange(0, 10800 + 1, 1)[::DOWNSAMPLE].astype(np.float64)
    ys = np.zeros_like(xs) + 1e-5

    cycle_data = copy.deepcopy(CYCLE_DATA)
    current_correction = trim.get_current_correction()
    cycle_data.correction = np.stack(
        (current_correction.xs, current_correction.ys), axis=0
    )

    trim.apply_correction(
        delta_t=xs,
        delta_v=ys,
        cycle_data=cycle_data,
    )


@pytest.mark.uses_virtual_device
def test_send_trim_boundaries(cycle_data_list: list[CycleData]) -> None:
    trim = TrimModel()
    trim.selector = SELECTOR

    trim.set_trim_t_min(200)
    trim.set_trim_t_max(1460)

    cycle_data = copy.deepcopy(cycle_data_list[0])
    trim._reference_fields[cycle_data.cycle] = cycle_data.field_ref  # type: ignore[assignment]
    time, delta = trim.calc_delta(cycle_data)

    trim.apply_correction(
        delta_t=time,
        delta_v=delta,
        cycle_data=cycle_data,
    )


@pytest.mark.uses_virtual_device
def test_send_on_new_prediction(cycle_data_list: list[CycleData]) -> None:
    trim = TrimModel()
    trim.selector = SELECTOR

    trim.set_trim_t_min(200)
    trim.set_trim_t_max(1460)

    cycle_data = copy.deepcopy(cycle_data_list[0])

    trim.on_new_prediction(
        prediction=cycle_data,
    )
