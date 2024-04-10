from __future__ import annotations

import pytest
import numpy as np
import logging
from op_app_context import context
import datetime
import copy
import pyrbac


from sps_apps.hysteresis_prediction.widgets.trim_widget import TrimModel
from sps_apps.hysteresis_prediction.data import CycleData

context.lsa_server = "next"  # WARNING
context.rbac_token = pyrbac.AuthenticationClient().login_location()

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


# @pytest.mark.uses_virtual_device
# def test_send_trim() -> None:
#     trim = TrimModel()
#     trim.selector = SELECTOR
#
#     xs = np.arange(0, 10800 + 1, 1)[::DOWNSAMPLE].astype(np.float64)
#     ys = np.zeros_like(xs) + 1e-5
#
#     cycle_data = copy.deepcopy(CYCLE_DATA)
#     current_correction = trim.get_current_correction()
#     cycle_data.correction = np.stack(
#         (current_correction.xs, current_correction.ys), axis=0
#     )
#
#     trim.apply_correction(
#         correction_t=xs,
#         correction_v=ys,
#         cycle_data=cycle_data,
#     )
#


@pytest.mark.uses_virtual_device
def test_send_trim_boundaries(cycle_data_list: list[CycleData]) -> None:
    trim = TrimModel()
    trim.selector = SELECTOR

    trim.set_trim_t_min(200)
    trim.set_trim_t_max(1460)

    cycle_data = copy.deepcopy(cycle_data_list[0])
    delta = cycle_data.field_ref[1, :] - cycle_data.field_pred[1, :]  # type: ignore[index]
    xs = cycle_data.field_pred[0, :] - cycle_data.field_pred[0, 0]  # type: ignore[index]
    xs = xs * 1e3
    xs = np.round(xs, 1)

    current_correction = trim.get_current_correction()
    cycle_data.correction = np.stack(
        (current_correction.xs, current_correction.ys), axis=0
    )

    trim.apply_correction(
        correction_t=xs,
        correction_v=delta,
        cycle_data=cycle_data,
    )
