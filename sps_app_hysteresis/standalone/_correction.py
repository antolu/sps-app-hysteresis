from __future__ import annotations

import datetime
import logging
import time
from collections.abc import Iterator, MutableMapping
from dataclasses import dataclass

import hystcomp_utils.cycle_data
import numpy as np
import numpy.typing as npt
import pyda
import pyda.access
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtCore

from ..contexts import app_context
from ..settings import TrimSettings
from ..utils import cycle_metadata
from ._inference import PredictionMode
from .event_building import EventBuilderAbc

log = logging.getLogger(__name__)


@dataclass
class FieldReference:
    """Container for field reference data and timestamp."""

    data: npt.NDArray[np.float64]  # [2, n_points] array
    timestamp: float


class FieldReferenceStore(MutableMapping[str, npt.NDArray[np.float64]]):
    """Dict-like container for field references with automatic timestamp management."""

    def __init__(self) -> None:
        self._references: dict[str, FieldReference] = {}

    def __getitem__(self, key: str) -> npt.NDArray[np.float64]:
        return self._references[key].data

    def __setitem__(self, key: str, value: npt.NDArray[np.float64]) -> None:
        # For compatibility with MutableMapping, we need to handle both cases
        if isinstance(value, tuple):
            data, timestamp = value
            self._references[key] = FieldReference(data, timestamp)
        else:
            # If only data is provided, use current timestamp
            timestamp = time.time_ns()
            self._references[key] = FieldReference(value, timestamp)

    def __delitem__(self, key: str) -> None:
        del self._references[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._references)

    def __len__(self) -> int:
        return len(self._references)

    def __contains__(self, key: object) -> bool:
        return key in self._references

    def set_with_timestamp(
        self, key: str, data: npt.NDArray[np.float64], timestamp: float
    ) -> None:
        """Set reference with explicit timestamp."""
        self._references[key] = FieldReference(data, timestamp)

    def get_timestamp(self, key: str) -> float:
        """Get the timestamp for a reference."""
        return self._references[key].timestamp

    def clear(self) -> None:
        """Remove all references."""
        self._references.clear()


class CalculateCorrection(EventBuilderAbc):
    newReference = QtCore.Signal(CycleData)

    def _handle_acquisition_impl(
        self, fspv: pyda.access.PropertyRetrievalResponse
    ) -> None:
        pass

    def __init__(
        self, trim_settings: TrimSettings, parent: QtCore.QObject | None = None
    ) -> None:
        super().__init__(parent=parent)

        self._trim_settings = trim_settings

        # Field reference stores with automatic timestamp management
        self._field_ref = FieldReferenceStore()
        self._field_ref_hyst = FieldReferenceStore()
        self._field_ref_eddy = FieldReferenceStore()

        # Prediction mode
        self._prediction_mode: PredictionMode | None = None

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"{cycle}: Calculating correction."
        log.debug(msg)

        self._maybe_save_reference(cycle)

        if cycle.field_pred is None:
            log.error(
                f"{cycle}: Prediction field not set. Cannot calculate correction."
            )
            return

        if cycle.economy_mode is not hystcomp_utils.cycle_data.EconomyMode.NONE:
            log.debug(f"[{cycle}]: Skipping correction calculation for economy cycle.")
            self.cycleDataAvailable.emit(cycle)
            return

        beam_in = cycle_metadata.beam_in(cycle.cycle)
        beam_out = cycle_metadata.beam_out(cycle.cycle)
        log.debug(
            f"{cycle}: Cutting delta with Beam in: {beam_in}, Beam out: {beam_out}"
        )
        delta = calc_delta_field(
            self._field_ref[self._cycle_id(cycle)],
            cycle.field_pred,
            beam_in=beam_in,
            beam_out=beam_out,
        )
        delta[1] = smooth_correction(*delta)[1]
        cycle.delta_applied = delta

        msg = f"{cycle}: Delta calculated."
        log.debug(msg)

        # Check if we need to update eddy current reference
        if self._prediction_mode is not None and self._prediction_mode in [
            PredictionMode.EDDY_CURRENT_ONLY,
            PredictionMode.COMBINED,
        ]:
            self._update_eddy_current_reference_after_correction(
                cycle, self._prediction_mode
            )

        if cycle.correction is not None:  # and not cycle.cycle.endswith("ECO"):
            try:
                current_correction = np.vstack(
                    cut_trim_beyond_time(
                        cycle.correction[0],
                        cycle.correction[1],
                        beam_in,
                        beam_out,
                    )
                )
                correction = calc_new_correction(
                    current_correction, delta, self._trim_settings.gain[cycle.cycle]
                )
            except:  # noqa: E722
                log.exception(f"{cycle}: Could not calculate correction.")
                return

            correction = clip_correction(
                correction,
                cycle.field_prog,
                clip_factor=app_context().TRIM_CLIP_THRESHOLD,
            )

            cycle.correction_applied = correction

            msg = f"{cycle}: New correction calculated."
            log.debug(msg)

        self.cycleDataAvailable.emit(cycle)

    @QtCore.Slot(str)
    def resetReference(self, cycle_name: str | None = None) -> None:
        if cycle_name is None or cycle_name == "all":
            log.info("Resetting all field references.")
            self._field_ref.clear()
            self._field_ref_hyst.clear()
            self._field_ref_eddy.clear()
            return

        # Reset legacy reference
        if cycle_name in self._field_ref:
            log.info(f"{cycle_name}: Resetting field reference.")
            del self._field_ref[cycle_name]

        # Reset separated references
        if cycle_name in self._field_ref_hyst:
            log.info(f"{cycle_name}: Resetting hysteresis field reference.")
            del self._field_ref_hyst[cycle_name]

        if cycle_name in self._field_ref_eddy:
            log.info(f"{cycle_name}: Resetting eddy current field reference.")
            del self._field_ref_eddy[cycle_name]

        if (
            cycle_name not in self._field_ref
            and cycle_name not in self._field_ref_hyst
            and cycle_name not in self._field_ref_eddy
        ):
            log.debug(f"{cycle_name}: No field references set. Nothing to reset.")

    def updateEddyCurrentReference(
        self, cycle_name: str, delta_correction: npt.NDArray[np.float64]
    ) -> None:
        """Update eddy current reference after correction is applied.

        :param cycle_name: Name of the cycle
        :param delta_correction: Correction delta [2, n_points] where [0] is time, [1] is field values
        """
        if cycle_name not in self._field_ref_eddy:
            log.warning(f"{cycle_name}: No eddy current reference to update.")
            return

        # Update reference field values: B_ref_new[1] = B_ref_old[1] + delta_correction[1]
        # Time axis (row 0) remains unchanged
        current_ref = self._field_ref_eddy[cycle_name]

        # Use match_array_size to handle interpolation
        matched_current_ref, matched_delta = match_array_size(
            current_ref, delta_correction
        )

        # Update only the field values (row 1), keep time axis unchanged
        updated_data = np.vstack((
            matched_current_ref[0],  # Keep original time axis
            matched_current_ref[1] + matched_delta[1],  # Update field values
        ))

        # Update the reference with the same timestamp
        current_timestamp = self._field_ref_eddy.get_timestamp(cycle_name)
        self._field_ref_eddy.set_with_timestamp(
            cycle_name, updated_data, current_timestamp
        )

        log.debug(
            f"{cycle_name}: Updated eddy current reference field values after correction application."
        )

    def set_prediction_mode(self, prediction_mode: PredictionMode) -> None:
        """Set the current prediction mode."""
        self._prediction_mode = prediction_mode

    def _update_eddy_current_reference_after_correction(
        self,
        cycle_data: hystcomp_utils.cycle_data.CycleData,
        prediction_mode: PredictionMode,
    ) -> None:
        """Update eddy current reference after correction is calculated."""
        cycle_id = self._cycle_id(cycle_data)

        if cycle_id not in self._field_ref_eddy:
            log.debug(f"{cycle_data}: No eddy current reference to update.")
            return

        if prediction_mode == PredictionMode.EDDY_CURRENT_ONLY:
            # In eddy current only mode, the entire delta is eddy current
            if cycle_data.delta_applied is not None:
                eddy_delta = cycle_data.delta_applied
                self.updateEddyCurrentReference(cycle_data.cycle, eddy_delta)
                log.debug(
                    f"{cycle_data}: Updated eddy current reference (eddy current only mode)"
                )

        elif prediction_mode == PredictionMode.COMBINED:
            # In combined mode, calculate the eddy current portion of the delta
            if (
                cycle_data.field_ref_eddy is not None
                and cycle_data.field_pred_eddy is not None
            ):
                # Calculate eddy current delta: ref - pred
                eddy_ref_t = cycle_data.field_ref_eddy[0]
                eddy_ref_v = cycle_data.field_ref_eddy[1]
                eddy_pred_t = cycle_data.field_pred_eddy[0]
                eddy_pred_v = cycle_data.field_pred_eddy[1]

                # Interpolate prediction to reference time grid
                eddy_pred_interp = np.interp(eddy_ref_t, eddy_pred_t, eddy_pred_v)

                # Calculate delta and apply gain
                eddy_delta_v = eddy_ref_v - eddy_pred_interp
                gain = self._trim_settings.gain[cycle_data.cycle]
                eddy_delta_v = gain * eddy_delta_v

                # Create delta array
                eddy_delta = np.vstack((eddy_ref_t, eddy_delta_v))

                # Apply time limits if needed (same as main correction)
                beam_in = cycle_metadata.beam_in(cycle_data.cycle)
                beam_out = cycle_metadata.beam_out(cycle_data.cycle)

                if beam_in is not None and beam_out is not None:
                    eddy_delta_t, eddy_delta_v = cut_trim_beyond_time(
                        eddy_delta[0], eddy_delta[1], beam_in, beam_out
                    )
                    eddy_delta = np.vstack((eddy_delta_t, eddy_delta_v))

                self.updateEddyCurrentReference(cycle_data.cycle, eddy_delta)
                log.debug(
                    f"{cycle_data}: Updated eddy current reference (combined mode)"
                )
            else:
                log.warning(
                    f"{cycle_data}: Cannot update eddy current reference - missing reference or prediction"
                )

    def _maybe_save_reference(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        if cycle_data.field_pred is None:
            log.error(f"{cycle_data}: Prediction field not set. Cannot save reference.")
            return

        if (
            cycle_data.reference_timestamp is not None
            and np.allclose(cycle_data.cycle_timestamp, cycle_data.reference_timestamp)
            and cycle_data.economy_mode
            is not hystcomp_utils.cycle_data.EconomyMode.NONE
        ):
            cycle_name = cycle_data.cycle

            # compare non-ECO cycle to ECO cycle, if the same, delete the reference
            # because the ECO cycle is the reference
            if self._field_ref.get_timestamp(cycle_name) == cycle_data.cycle_timestamp:
                msg = f"{cycle_data}: Last cycle was ECO, need to delete the last reference."
                log.debug(msg)

                self.resetReference(cycle_name)
            cycle_data.reference_timestamp = None

            # then, save or update the reference with the ECO cycle name
        #
        # save the reference if it has not been set, or reference was removed
        id_ = self._cycle_id(cycle_data)
        if id_ not in self._field_ref:
            log.debug(
                f"{cycle_data}: Saving field reference since it has not "
                f"been set ({cycle_data.field_pred.shape})."
            )

            if cycle_data.field_pred is None:
                log.error(
                    f"{cycle_data}: Prediction field not set. Cannot calculate correction."
                )
                return

            self._field_ref.set_with_timestamp(
                id_, cycle_data.field_pred, cycle_data.cycle_timestamp
            )
            self.newReference.emit(cycle_data)

        # Save separated references for hysteresis and eddy current
        if id_ not in self._field_ref_hyst and cycle_data.field_pred_hyst is not None:
            log.debug(f"{cycle_data}: Saving hysteresis field reference.")
            self._field_ref_hyst.set_with_timestamp(
                id_, cycle_data.field_pred_hyst, cycle_data.cycle_timestamp
            )
            cycle_data.field_ref_hyst = self._field_ref_hyst[id_]

        if id_ not in self._field_ref_eddy and cycle_data.field_pred_eddy is not None:
            log.debug(f"{cycle_data}: Saving eddy current field reference.")
            self._field_ref_eddy.set_with_timestamp(
                id_, cycle_data.field_pred_eddy, cycle_data.cycle_timestamp
            )
            cycle_data.field_ref_eddy = self._field_ref_eddy[id_]
        else:  # set the
            ref_time = datetime.datetime.fromtimestamp(
                self._field_ref.get_timestamp(id_) * 1e-9
            )
            log.debug(
                f"{cycle_data}: Reference already saved from timestamp {ref_time}. "
                f"Adding the reference to the cycle data."
            )

        cycle_data.reference_timestamp = self._field_ref.get_timestamp(id_)
        cycle_data.field_ref = self._field_ref[id_]

        # Set separated references in cycle data if they exist
        if id_ in self._field_ref_hyst:
            cycle_data.field_ref_hyst = self._field_ref_hyst[id_]
        if id_ in self._field_ref_eddy:
            cycle_data.field_ref_eddy = self._field_ref_eddy[id_]

    @staticmethod
    def _cycle_id(cycle_data: hystcomp_utils.cycle_data.CycleData) -> str:
        id_ = cycle_data.cycle
        if cycle_data.economy_mode is not hystcomp_utils.cycle_data.EconomyMode.NONE:
            if cycle_data.economy_mode is hystcomp_utils.cycle_data.EconomyMode.FULL:
                id_ += "_FULLECO"
            elif (
                cycle_data.economy_mode is hystcomp_utils.cycle_data.EconomyMode.DYNAMIC
            ):
                id_ += "_DYNECO"

        return id_


def calc_delta_field(
    field_ref: npt.NDArray[np.float64],
    field_pred: npt.NDArray[np.float64],
    beam_in: float | None = None,
    beam_out: float | None = None,
    *,
    flatten: bool = False,
) -> npt.NDArray[np.float64]:
    # calc delta and smooth it
    ref_t = (field_ref[0, :] - field_ref[0, 0]) * 1e3  # absolute time in s to ms
    ref_t = np.round(ref_t, 1)
    ref_v = field_ref[1, :]

    extra_points = []
    if beam_in is not None:
        extra_points.append(beam_in)
    if beam_out is not None:
        extra_points.append(beam_out)

    ref_t, ref_v = insert_points(ref_t, ref_v, np.array(extra_points))

    pred_t = (field_pred[0, :] - field_pred[0, 0]) * 1e3
    pred_t = np.round(pred_t, 1)
    pred_v = field_pred[1, :]

    pred_t, pred_v = insert_points(pred_t, pred_v, np.array(extra_points))

    ((ref_t, ref_v), (pred_t, pred_v)) = match_array_size(
        np.vstack((ref_t, ref_v)),
        np.vstack((pred_t, pred_v)),
    )

    # only use this when flattening
    if flatten:
        ref_v = np.interp(
            pred_t,
            np.array([beam_in or ref_t[0], beam_out or ref_t[-1]]),
            np.array([pred_v[0], pred_v[0]]),
        )

    # cut trim beyond time limits
    delta_v = ref_v - pred_v
    delta_t = ref_t
    if beam_in is not None and beam_out is not None:
        delta_t, delta_v = cut_trim_beyond_time(delta_t, delta_v, beam_in, beam_out)
    else:
        msg = "Beam in and beam out not set. Not trimming the delta field."
        log.debug(msg)

    return np.vstack((delta_t, delta_v))


def insert_points(
    x: np.ndarray, y: np.ndarray, new_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Insert new points into an existing x, y array.

    Parameters
    ----------
    x : np.ndarray
        The x array.
    y : np.ndarray
        The y array.
    new_points : np.ndarray
        The new points to insert.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The new x, y arrays.
    """
    new_x = np.concatenate((x, new_points))
    new_x = np.sort(np.unique(new_x))
    new_y = np.interp(new_x, x, y)

    return new_x, new_y


def match_array_size(
    array1: npt.NDArray[np.float64],
    array2: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Calculate the union of the time axes
    union_time_axis = np.union1d(array1[0], array2[0])

    # Interpolate the values to the new time axis
    array1_interpolated = np.vstack((
        union_time_axis,
        np.interp(union_time_axis, array1[0], array1[1]),
    ))

    array2_interpolated = np.vstack((
        union_time_axis,
        np.interp(union_time_axis, array2[0], array2[1]),
    ))

    return array1_interpolated, array2_interpolated


def cut_trim_beyond_time(
    xs: npt.NDArray,
    ys: npt.NDArray,
    lower: float,
    upper: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Cuts and trims the time axis and the correction to the time range [lower, upper],
    usually in ms.

    Parameters
    ----------
    xs
    ys
    lower
    upper

    Returns
    -------

    """
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


def calc_new_correction(
    current_correction: npt.NDArray[np.float64],  # [2, n_points]
    delta: npt.NDArray[np.float64],  # [2, n_points]
    gain: float = 1.0,
) -> npt.NDArray[np.float64]:
    log.debug(f"Current correction shape: {current_correction.shape}")
    log.debug(f"Delta shape: {delta.shape}")
    current_correction, delta = match_array_size(
        current_correction,
        delta,
    )

    # calculate correction
    new_correction = (current_correction[1] + gain * delta[1]).astype(np.float64)

    # smooth the correction
    # new_correction = signal.perona_malik_smooth(
    #     new_correction, 5.0, 5e-2, 2.0
    # )

    # return delta for plotting
    return np.vstack((current_correction[0], new_correction))


def clip_correction(
    correction: npt.NDArray[np.float64],
    field_prog: npt.NDArray[np.float64],
    clip_factor: float = 1.0,
) -> npt.NDArray[np.float64]:
    """
    Clip the correction to the a a factor of the reference programmed field.

    Parameters
    ----------
    correction : npt.NDArray[np.float64]
        The correction to clip.
    field_prog : npt.NDArray[np.float64]
        The reference programmed field.
    clip_val : float
        The factor to clip the correction to.
    """
    field_prog_t = field_prog[0]
    field_prog_v = field_prog[1]
    field_prog_v_interp = np.interp(
        correction[0],
        field_prog_t,
        field_prog_v,
    )

    # clip the correction to the field prog
    correction_clipped = np.clip(
        correction[1],
        -(field_prog_v_interp * clip_factor),
        field_prog_v_interp * clip_factor,
    )

    # log max and min values before and after clipping
    log.debug(
        f"Correction min: {np.min(correction[1])}, max: {np.max(correction[1])}, "
        f"Clipped min: {np.min(correction_clipped)}, max: {np.max(correction_clipped)}"
    )

    return np.vstack((correction[0], correction_clipped))


# @numba.njit
def smooth_with_tolerance_interp(y: np.ndarray, atol: float = 1e-5) -> np.ndarray:
    y = y.copy()
    n = len(y)
    i = 1

    while i < n - 1:
        if np.abs(y[i] - y[i - 1]) < atol and np.abs(y[i] - y[i + 1]) < atol:
            start = i - 1
            while i < n - 1 and np.abs(y[start] - y[i]) < atol:
                i += 1
            end = i + 1
            y[start:end] = np.linspace(y[start], y[end - 1], end - start)
        i += 1
    return y


def smooth_correction(
    xs: npt.NDArray[np.float64],
    correction: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # correction = denoise_tv_chambolle(correction, weight=0.1)
    correction = smooth_with_tolerance_interp(correction, atol=5e-6)
    return xs, correction
