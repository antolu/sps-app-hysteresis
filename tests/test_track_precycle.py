from __future__ import annotations

import datetime

import hystcomp_utils.cycle_data
import numpy as np

from sps_app_hysteresis.standalone.track_precycle import (
    TrackPrecycleEventBuilder,
)


def create_cycle_data(user: str) -> hystcomp_utils.cycle_data.CycleData:
    return hystcomp_utils.cycle_data.CycleData(
        cycle=user,
        user=user,
        cycle_timestamp=datetime.datetime.now().timestamp() * 1e9,
        current_prog=np.array([[1.0, 1.0]]),
        field_prog=np.array([[]]),
    )


def test_track_precycle() -> None:
    precycle_sequence = ["SPS.USER.LHCPILOT", "SPS.USER.MD1"]
    event_builder = TrackPrecycleEventBuilder(precycle_sequence)

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.LHCPILOT"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.LHCPILOT"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.LHCPILOT"))
    assert event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert event_builder.precycle_active

    # end precycle

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.SFTPRO1"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert not event_builder.precycle_active

    # next precycle

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.LHCPILOT"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.LHCPILOT"))
    assert not event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.LHCPILOT"))
    assert event_builder.precycle_active

    event_builder.onNewCycleData(create_cycle_data("SPS.USER.MD1"))
    assert event_builder.precycle_active
