"""Plotting utilities for the hysteresis application."""

from __future__ import annotations

import numpy as np
from hystcomp_actions import TrimSettings
from hystcomp_actions.utils import cycle_metadata
from hystcomp_utils.cycle_data import CycleData


def calculate_delta_for_plotting(
    cycle_data: CycleData,
    trim_settings: TrimSettings,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Calculate delta for plotting purposes.

    This function calculates the delta (correction) that was applied for visualization.
    It should only be used for plotting, not for actual correction calculations.

    Args:
        cycle_data: The cycle data containing delta_applied
        trim_settings: Trim settings for gain values
        start: Start time for trimming (if None, uses beam_in)
        end: End time for trimming (if None, uses beam_out)

    Returns:
        Delta array [2, n_points] where [0] is time and [1] is corrected values

    Raises:
        AssertionError: If delta_applied is not available in cycle_data
    """
    assert cycle_data.delta_applied is not None, "No delta applied found for plotting."

    # Use provided time limits or fall back to beam times
    if start is None:
        start = cycle_metadata.beam_in(cycle_data.cycle)
    if end is None:
        end = cycle_metadata.beam_out(cycle_data.cycle)

    # Extract delta
    delta_t = cycle_data.delta_applied[0]
    delta_v = cycle_data.delta_applied[1]

    # Apply time limits if provided
    if start is not None and end is not None:
        delta_t, delta_v = cut_trim_beyond_time(delta_t, delta_v, start, end)

    # Apply gain
    gain = trim_settings.gain[cycle_data.cycle]
    delta_v = gain * delta_v

    return np.vstack((delta_t, delta_v))


def cut_trim_beyond_time(
    xs: np.ndarray,
    ys: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut and trim arrays to time range [lower, upper].

    This is a copy of the function from StandaloneTrim, moved here for plotting use.
    """
    # Add lower and upper bounds to the time axis
    time_axis_new = np.concatenate(([lower], xs, [upper]))
    time_axis_new = np.sort(np.unique(time_axis_new))

    ys_new = np.interp(time_axis_new, xs, ys)

    valid_indices = (lower <= time_axis_new) & (time_axis_new <= upper)
    time_axis_trunc = time_axis_new[valid_indices]
    correction_trunc = ys_new[valid_indices]

    return time_axis_trunc, correction_trunc
