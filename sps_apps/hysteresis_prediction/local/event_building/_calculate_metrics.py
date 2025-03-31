"""
Converter to calculate metrics for the hysteresis prediction model when
field measurements are available.

This converter is a copy of `sps_ucap_hystcomp/metrics/_converter.py` with modifications to make it work
offline.
"""

from __future__ import annotations

import enum
import logging
import typing

import hystcomp_utils.cycle_data
import numpy as np
from qtpy import QtCore

from ...trim import cycle_metadata
from ._event_builder_abc import EventBuilderAbc

if typing.TYPE_CHECKING:
    import pyda.access


log = logging.getLogger(__name__)


class Metrics(typing.TypedDict):
    lsaCycleName: str

    allRmse: float
    allMae: float
    allStd: float
    allMaxError: float

    beamInRmse: float
    beamInMae: float
    beamInStd: float
    beamInMaxError: float

    injectionRmse: float
    injectionMae: float
    injectionStd: float
    injectionMaxError: float

    flattopRmse: float
    flattopMae: float
    flattopStd: float
    flattopMaxError: float


class CalculateMetricsConverter(EventBuilderAbc):
    newMetricsAvailable = QtCore.Signal(dict)  # dict[str, Metrics]

    def __init__(self, *, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        # no need to handle acquisition
        msg = f"{self.__class__.__name__} does not subscribe to triggers."
        raise NotImplementedError(msg)

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        if cycle_data.field_ref is None:
            msg = f"{cycle_data}: field_ref is None, cannot calculate metrics."
            raise RuntimeError(msg)

        if cycle_data.user.split(".")[-1] in {"ZERO", "MD1"}:
            log.debug(f"Skipping cycle data for {cycle_data.user}")
            return

        relative_metrics = _calculate_metrics(
            cycle_data,
            beam_in=cycle_metadata.beam_in(cycle_data.cycle),
            beam_out=cycle_metadata.beam_out(cycle_data.cycle),
            injection_end=cycle_metadata.ramp_start(cycle_data.cycle),
            flattop_start=cycle_metadata.flattop_start(cycle_data.cycle),
            metrics_type=MetricsType.RELATIVE,
        )

        absolute_metrics = _calculate_metrics(
            cycle_data,
            beam_in=cycle_metadata.beam_in(cycle_data.cycle),
            beam_out=cycle_metadata.beam_out(cycle_data.cycle),
            injection_end=cycle_metadata.ramp_start(cycle_data.cycle),
            flattop_start=cycle_metadata.flattop_start(cycle_data.cycle),
            metrics_type=MetricsType.ABSOLUTE,
        )

        self.newMetricsAvailable.emit({
            "relative": relative_metrics,
            "absolute": absolute_metrics,
        })


class MetricsType(enum.StrEnum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


def _calculate_metrics(
    cycle_data: hystcomp_utils.cycle_data.CycleData,
    beam_in: float,
    beam_out: float,
    injection_end: float,
    flattop_start: float,
    *,
    metrics_type: MetricsType = MetricsType.RELATIVE,
) -> Metrics:
    """
    Calculate metrics for a cycle data.
    """
    if cycle_data.field_meas is None:
        msg = f"{cycle_data}: field_meas is None"
        raise RuntimeError(msg)
    if cycle_data.field_meas_ref is None:
        msg = f"{cycle_data}: field_meas_ref is None"
        raise RuntimeError(msg)
    if cycle_data.field_pred is None:
        msg = f"{cycle_data}: field_pred is None"
        raise RuntimeError(msg)
    if cycle_data.field_ref is None:
        msg = f"{cycle_data}: field_ref is None"
        raise RuntimeError(msg)

    metrics_fn = (
        _calculate_relative_metrics_with_boundaries
        if metrics_type == MetricsType.RELATIVE
        else _calculate_absolute_metrics_with_boundaries
    )

    metrics = {}

    metrics["lsaCycleName"] = cycle_data.cycle

    def rename_metrics(metrics: dict[str, float], name: str) -> dict[str, float]:
        return {
            f"{name}Rmse": metrics["rmse"],
            f"{name}Mae": metrics["mae"],
            f"{name}Std": metrics["std"],
            f"{name}MaxError": metrics["max_error"],
        }

    all_metrics = metrics_fn(cycle_data)
    metrics |= rename_metrics(all_metrics, "all")

    beam_in_metrics = metrics_fn(cycle_data, lower_t=beam_in, upper_t=beam_out)
    metrics |= rename_metrics(beam_in_metrics, "beamIn")

    injection_metrics = metrics_fn(cycle_data, lower_t=beam_in, upper_t=injection_end)
    metrics |= rename_metrics(injection_metrics, "injection")

    flattop_metrics = metrics_fn(cycle_data, lower_t=flattop_start, upper_t=beam_out)
    metrics |= rename_metrics(flattop_metrics, "flattop")

    return typing.cast(Metrics, metrics)


def _calculate_relative_metrics_with_boundaries(
    cycle_data: hystcomp_utils.cycle_data.CycleData,
    lower_t: float = 0,
    upper_t: float = np.inf,
) -> dict[str, float]:
    if cycle_data.field_pred is None:
        msg = f"{cycle_data}: field_pred is None"
        raise RuntimeError(msg)
    assert cycle_data.field_ref is not None

    if cycle_data.field_meas is None:
        msg = f"{cycle_data}: field_meas is None"
        raise RuntimeError(msg)

    t_pred = cycle_data.field_pred[0] * 1e3  # ms
    t_pred -= t_pred[0]  # start from 0
    t_pred = np.round(t_pred, 1)
    delta_pred = cycle_data.field_pred[1] - cycle_data.field_ref[1]

    t_meas = np.arange(0, len(cycle_data.field_meas.flatten()))  # ms
    delta_meas = (cycle_data.field_meas - cycle_data.field_meas_ref).flatten()

    pred_to_keep = (t_pred >= lower_t) & (t_pred <= upper_t)
    t_pred = t_pred[pred_to_keep]
    delta_pred = delta_pred[pred_to_keep]

    meas_to_keep = (t_meas >= lower_t) & (t_meas <= upper_t)
    t_meas = t_meas[meas_to_keep]  # type: ignore[assignment]
    delta_meas = delta_meas[meas_to_keep]

    delta_meas_interp = np.interp(t_pred, t_meas, delta_meas)

    rmse = np.sqrt(np.mean((delta_meas_interp - delta_pred) ** 2))
    mae = np.mean(np.abs(delta_meas_interp - delta_pred))
    std = np.std(np.abs(delta_meas_interp - delta_pred))
    max_error = np.max(np.abs(delta_meas_interp - delta_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "std": std,
        "max_error": max_error,
    }


def _calculate_absolute_metrics_with_boundaries(
    cycle_data: hystcomp_utils.cycle_data.CycleData,
    lower_t: float = 0,
    upper_t: float = np.inf,
) -> dict[str, float]:
    if cycle_data.field_pred is None:
        msg = f"{cycle_data}: field_pred is None"
        raise RuntimeError(msg)
    assert cycle_data.field_ref is not None

    if cycle_data.field_meas is None:
        msg = f"{cycle_data}: field_meas is None"
        raise RuntimeError(msg)

    t_pred = cycle_data.field_pred[0] * 1e3  # ms
    t_pred -= t_pred[0]  # start from 0
    t_pred = np.round(t_pred, 1)
    field_pred = cycle_data.field_pred[1]

    t_meas = np.arange(0, cycle_data.field_meas.size)  # ms
    field_meas = cycle_data.field_meas.flatten()
    field_meas_interp = np.interp(t_pred, t_meas, field_meas)

    to_keep = (t_pred >= lower_t) & (t_pred <= upper_t)
    t_pred = t_pred[to_keep]
    field_pred = field_pred[to_keep]
    field_meas_interp = field_meas_interp[to_keep]

    rmse = np.sqrt(np.mean((field_meas_interp - field_pred) ** 2))
    mae = np.mean(np.abs(field_meas_interp - field_pred))
    std = np.std(np.abs(field_meas_interp - field_pred))
    max_error = np.max(np.abs(field_meas_interp - field_pred))

    return {
        "rmse": rmse,
        "mae": mae,
        "std": std,
        "max_error": max_error,
    }
