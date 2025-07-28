from __future__ import annotations

import dataclasses
import datetime
import logging
import time
from collections.abc import Callable, Iterator, MutableMapping

import hystcomp_utils.cycle_data
import numpy as np
import numpy.typing as npt
import pyda
import pyda.access
from hystcomp_utils.cycle_data import CorrectionMode, CycleData
from qtpy import QtCore

from ..contexts import app_context
from ..settings import TrimSettings
from ..utils import cycle_metadata
from .event_building import EventBuilderAbc

log = logging.getLogger(__name__)


class CorrectionCalculator:
    """Handles correction calculation and clipping operations."""

    def __init__(self, trim_settings: TrimSettings):
        self._trim_settings = trim_settings

    def calculate_correction(
        self,
        current_correction: npt.NDArray[np.float64],
        delta: npt.NDArray[np.float64],
        cycle_name: str,
        field_prog: npt.NDArray[np.float64],
        beam_in: float | None = None,
        beam_out: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Calculate and clip correction using trim window."""
        try:
            # Cut current correction to beam limits
            if beam_in is not None and beam_out is not None:
                current_correction = np.vstack(
                    cut_trim_beyond_time(
                        current_correction[0],
                        current_correction[1],
                        beam_in,
                        beam_out,
                    )
                )

            # Get trim window times
            trim_start = self._trim_settings.trim_start[cycle_name]
            trim_end = self._trim_settings.trim_end[cycle_name]

            # Calculate new correction with trim window
            gain = self._trim_settings.gain[cycle_name]
            correction = calc_new_correction_with_trim_window(
                current_correction, delta, trim_start, trim_end, gain
            )

            # Clip correction
            return clip_correction(
                correction,
                field_prog,
                clip_factor=app_context().TRIM_CLIP_THRESHOLD,
            )

        except Exception:
            log.exception(f"Could not calculate correction for {cycle_name}")
            raise


class CorrectionHelper:
    """Helper class for common correction operations."""

    @staticmethod
    def get_beam_times(cycle_name: str) -> tuple[int, int]:
        """Get beam in and beam out times for a cycle."""
        beam_in = cycle_metadata.beam_in(cycle_name)
        beam_out = cycle_metadata.beam_out(cycle_name)
        return beam_in, beam_out

    @staticmethod
    def apply_running_eddy_adjustment(
        delta_eddy: npt.NDArray[np.float64],
        running_field_ref_eddy: FieldReferenceStore,
        cycle_id: str,
    ) -> npt.NDArray[np.float64]:
        """Apply running eddy current reference adjustment to delta."""
        if cycle_id in running_field_ref_eddy:
            delta_eddy, match_delta = match_array_size(
                delta_eddy, running_field_ref_eddy[cycle_id]
            )
            delta_eddy[1] -= match_delta[1]
        return delta_eddy


class EddyCorrectionManager:
    """Manages eddy current reference updates."""

    def __init__(self, trim_settings: TrimSettings):
        self._trim_settings = trim_settings
        self._correction_calculator = CorrectionCalculator(trim_settings)

    def calculate_eddy_delta(
        self,
        cycle_data: hystcomp_utils.cycle_data.CycleData,
    ) -> npt.NDArray[np.float64]:
        """Calculate eddy current delta from reference and prediction."""
        if cycle_data.field_ref_eddy is None or cycle_data.field_pred_eddy is None:
            msg = "Missing eddy current reference or prediction"
            raise ValueError(msg)

        # Calculate eddy current delta: ref - pred
        eddy_ref_t = cycle_data.field_ref_eddy[0]
        eddy_ref_v = cycle_data.field_ref_eddy[1]
        eddy_pred_t = cycle_data.field_pred_eddy[0]
        eddy_pred_v = cycle_data.field_pred_eddy[1]

        # Interpolate prediction to reference time grid
        eddy_pred_interp = np.interp(eddy_ref_t, eddy_pred_t, eddy_pred_v)

        # Calculate delta and apply gain
        gain = self._trim_settings.gain[cycle_data.cycle]
        eddy_delta_v = gain * (eddy_ref_v - eddy_pred_interp)

        # Create delta array
        return np.vstack((eddy_ref_t, eddy_delta_v))

    def update_reference_after_trim(
        self,
        cycle_data: hystcomp_utils.cycle_data.CycleData,
        correction_mode: CorrectionMode,
        field_ref_eddy_store: FieldReferenceStore,
        running_field_ref_eddy: FieldReferenceStore,
        update_reference_callback: Callable[[str, npt.NDArray[np.float64]], None],
    ) -> None:
        """Update eddy current reference after trim completion."""
        cycle_id = _cycle_id_helper(cycle_data)

        if cycle_id not in field_ref_eddy_store:
            log.debug(f"[{cycle_data}] No eddy current reference to update")
            return

        if correction_mode == CorrectionMode.EDDY_CURRENT_ONLY:  # noqa: SIM102
            # In eddy current only mode, the entire applied correction is eddy current
            if cycle_data.correction_applied is not None:
                update_reference_callback(cycle_id, cycle_data.correction_applied)
                log.debug(
                    f"{cycle_data}: Updated eddy current reference after trim (eddy current only mode)"
                )

        if correction_mode == CorrectionMode.COMBINED:
            # In combined mode, use stored eddy current delta if available
            eddy_delta = cycle_data.delta_eddy

            if eddy_delta is None:
                log.warning(
                    f"[{cycle_data}]: No eddy current delta available for combined mode update"
                )
                return

            # Calculate actual correction to apply
            beam_in, beam_out = CorrectionHelper.get_beam_times(cycle_data.cycle)

            try:
                if cycle_data.correction is not None:
                    # Normally this is not None
                    correction = self._correction_calculator.calculate_correction(
                        cycle_data.correction,
                        eddy_delta,
                        cycle_data.cycle,
                        cycle_data.field_prog,
                        beam_in,
                        beam_out,
                    )

                    update_reference_callback(cycle_data.cycle, correction)
                    log.debug(
                        f"{cycle_data}: Updated eddy current reference after trim (combined mode)"
                    )
            except Exception:
                log.exception(
                    f"[{cycle_data}]: Could not update eddy current reference"
                )
                return

        # hysteresis only


@dataclasses.dataclass
class FieldReference:
    """Container for field reference data and timestamp."""

    data: npt.NDArray[np.float64]  # [2, n_points] array
    timestamp: float


@dataclasses.dataclass
class DeltaComponents:
    """Container for separated delta components."""

    combined: npt.NDArray[np.float64]  # [2, n_points] - combined delta
    hysteresis: npt.NDArray[np.float64] | None = (
        None  # [2, n_points] - hysteresis delta only
    )
    eddy_current: npt.NDArray[np.float64] | None = (
        None  # [2, n_points] - eddy current delta only
    )


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
        self._correction_calculator = CorrectionCalculator(trim_settings)
        self._eddy_correction_manager = EddyCorrectionManager(trim_settings)

        # Field reference stores with automatic timestamp management
        self._field_ref = FieldReferenceStore()
        self._field_ref_hyst = FieldReferenceStore()
        self._field_ref_eddy = FieldReferenceStore()
        self._running_field_ref_eddy = FieldReferenceStore()

        # Prediction mode
        self._prediction_mode: CorrectionMode | None = None

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"{cycle}: Calculating correction."
        log.debug(msg)

        # Set the correction mode from the current prediction mode
        cycle.correction_mode = self._prediction_mode or CorrectionMode.COMBINED

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

        beam_in, beam_out = CorrectionHelper.get_beam_times(cycle.cycle)
        log.debug(
            f"{cycle}: Cutting delta with Beam in: {beam_in}, Beam out: {beam_out}"
        )

        # Use the new separated delta calculation
        delta_components = calc_separated_delta_field(
            cycle,
            self._field_ref,
            self._field_ref_hyst,
            self._field_ref_eddy,
            self._running_field_ref_eddy,
            beam_in=beam_in,
            beam_out=beam_out,
        )

        # Store individual delta components in cycle data for later use
        cycle.delta_hyst = delta_components.hysteresis
        cycle.delta_eddy = delta_components.eddy_current

        # Apply smoothing to the combined delta
        delta = delta_components.combined
        delta[1] = smooth_correction(*delta)[1]
        cycle.delta_applied = delta

        msg = f"{cycle}: Delta calculated."
        log.debug(msg)

        if cycle.correction is not None:  # and not cycle.cycle.endswith("ECO"):
            try:
                correction = self._correction_calculator.calculate_correction(
                    cycle.correction,
                    delta,
                    cycle.cycle,
                    cycle.field_prog,
                    beam_in,
                    beam_out,
                )
                cycle.correction_applied = correction
                msg = f"{cycle}: New correction calculated."
                log.debug(msg)
            except:  # noqa: E722
                log.exception(f"{cycle}: Could not calculate correction.")
                return

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
            log.info(f"[{cycle_name}] Resetting field reference")
            del self._field_ref[cycle_name]

        # Reset separated references
        if cycle_name in self._field_ref_hyst:
            log.info(f"[{cycle_name}] Resetting hysteresis field reference")
            del self._field_ref_hyst[cycle_name]

        if cycle_name in self._field_ref_eddy:
            log.info(f"[{cycle_name}] Resetting eddy current field reference")
            del self._field_ref_eddy[cycle_name]

        if (
            cycle_name not in self._field_ref
            and cycle_name not in self._field_ref_hyst
            and cycle_name not in self._field_ref_eddy
        ):
            log.debug(f"[{cycle_name}] No field references set. Nothing to reset")

    def updateEddyCurrentReference(
        self, cycle_name: str, delta_correction: npt.NDArray[np.float64]
    ) -> None:
        """Update eddy current reference after correction is applied.

        :param cycle_name: Name of the cycle
        :param delta_correction: Correction delta [2, n_points] where [0] is time, [1] is field values
        """
        if cycle_name not in self._field_ref_eddy:
            log.warning(f"[{cycle_name}] No eddy current reference to update")
            return

        # Update reference field values: B_ref_new[1] = B_ref_old[1] + delta_correction[1]
        # Time axis (row 0) remains unchanged
        if cycle_name not in self._running_field_ref_eddy:
            log.debug(f"{cycle_name}: Initializing running eddy current reference")
            self._running_field_ref_eddy[cycle_name] = np.vstack((
                np.array([0.0]),
                np.array([0.0]),
            ))
        current_ref = self._running_field_ref_eddy[cycle_name]

        # Use match_array_size to handle interpolation
        matched_current_ref, matched_delta = match_array_size(
            current_ref, delta_correction
        )

        # Update only the field values (row 1), keep time axis unchanged
        updated_data = np.vstack((
            matched_current_ref[0],  # Keep original time axis
            matched_delta[1],
        ))

        # Update the reference with the same timestamp
        current_timestamp = self._running_field_ref_eddy.get_timestamp(cycle_name)
        self._running_field_ref_eddy.set_with_timestamp(
            cycle_name, updated_data, current_timestamp
        )

        log.debug(
            f"{cycle_name}: Updated eddy current reference field values after correction application."
        )

    def set_prediction_mode(self, prediction_mode: CorrectionMode) -> None:
        """Set the current prediction mode."""
        self._prediction_mode = prediction_mode

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData, object, str)
    def onTrimCompleted(
        self,
        cycle_data: hystcomp_utils.cycle_data.CycleData,
        trim_timestamp: object,
        comment: str,
    ) -> None:
        """Handle trim completion and update eddy current reference if needed."""
        correction_mode = getattr(
            cycle_data, "correction_mode", CorrectionMode.COMBINED
        )

        log.debug(f"[{cycle_data}] Trim completed with mode: {correction_mode.value}")

        # Update eddy current reference only for modes that use eddy current
        if correction_mode in [
            CorrectionMode.EDDY_CURRENT_ONLY,
            CorrectionMode.COMBINED,
        ]:
            self._update_eddy_current_reference_after_trim(cycle_data, correction_mode)

    def _update_eddy_current_reference_after_trim(
        self,
        cycle_data: hystcomp_utils.cycle_data.CycleData,
        correction_mode: CorrectionMode,
    ) -> None:
        """Update eddy current reference after trim is successfully applied."""
        self._eddy_correction_manager.update_reference_after_trim(
            cycle_data,
            correction_mode,
            self._field_ref_eddy,
            self._running_field_ref_eddy,
            self.updateEddyCurrentReference,
        )

    def _maybe_save_reference(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        if cycle_data.field_pred is None:
            log.error(f"[{cycle_data}] Prediction field not set. Cannot save reference")
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
        id_ = _cycle_id_helper(cycle_data)
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
            log.debug(f"[{cycle_data}] Saving hysteresis field reference")
            self._field_ref_hyst.set_with_timestamp(
                id_, cycle_data.field_pred_hyst, cycle_data.cycle_timestamp
            )
            cycle_data.field_ref_hyst = self._field_ref_hyst[id_]

        if id_ not in self._field_ref_eddy and cycle_data.field_pred_eddy is not None:
            log.debug(f"[{cycle_data}] Saving eddy current field reference")
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

    # calculate delta without slicing - keep full time range
    delta_v = ref_v - pred_v
    delta_t = ref_t

    return np.vstack((delta_t, delta_v))


def calc_separated_delta_field(
    cycle_data: hystcomp_utils.cycle_data.CycleData,
    field_ref_store: FieldReferenceStore,
    field_ref_hyst_store: FieldReferenceStore,
    field_ref_eddy_store: FieldReferenceStore,
    running_field_ref_eddy: FieldReferenceStore | None = None,
    beam_in: float | None = None,
    beam_out: float | None = None,
    *,
    flatten: bool = False,
) -> DeltaComponents:
    """Calculate delta field using separated hysteresis and eddy current components.

    This function computes the delta field based on the correction mode:
    - HYSTERESIS_ONLY: Uses only hysteresis reference and prediction
    - EDDY_CURRENT_ONLY: Uses only eddy current reference and prediction
    - COMBINED: Computes both deltas separately, interpolates to common grid, and adds them

    Args:
        cycle_data: The cycle data containing predictions and correction mode
        field_ref_store: Legacy field reference store (for backward compatibility)
        field_ref_hyst_store: Hysteresis field reference store
        field_ref_eddy_store: Eddy current field reference store
        beam_in: Beam in time (ms)
        beam_out: Beam out time (ms)
        flatten: Whether to apply flattening correction

    Returns:
        Combined delta field [2, n_points] where [0] is time, [1] is field values
    """
    cycle_id = _cycle_id_helper(cycle_data)
    correction_mode = getattr(cycle_data, "correction_mode", CorrectionMode.COMBINED)

    log.debug(
        f"[{cycle_data}] Computing separated delta with mode: {correction_mode.value}"
    )

    # Helper function to compute delta for a single component
    def compute_single_delta(
        field_ref: npt.NDArray[np.float64],
        field_pred: npt.NDArray[np.float64],
        component_name: str,
    ) -> npt.NDArray[np.float64]:
        log.debug(f"[{cycle_data}] Computing {component_name} delta")
        return calc_delta_field(
            field_ref, field_pred, beam_in=beam_in, beam_out=beam_out, flatten=flatten
        )

    if correction_mode == CorrectionMode.HYSTERESIS_ONLY:
        # Use only hysteresis components
        if cycle_id in field_ref_hyst_store and cycle_data.field_pred_hyst is not None:
            delta_hyst = compute_single_delta(
                field_ref_hyst_store[cycle_id], cycle_data.field_pred_hyst, "hysteresis"
            )
            return DeltaComponents(combined=delta_hyst, hysteresis=delta_hyst)
        # Fallback to legacy references if separated components not available
        log.warning(
            f"[{cycle_data}] Hysteresis components not available, falling back to legacy"
        )
        if cycle_data.field_pred is not None:
            delta = calc_delta_field(
                field_ref_store[cycle_id],
                cycle_data.field_pred,
                beam_in=beam_in,
                beam_out=beam_out,
                flatten=flatten,
            )
            return DeltaComponents(combined=delta)
        msg = f"[{cycle_data}] No field prediction available"
        raise ValueError(msg)

    if correction_mode == CorrectionMode.EDDY_CURRENT_ONLY:
        # Use only eddy current components
        if cycle_id in field_ref_eddy_store and cycle_data.field_pred_eddy is not None:
            delta_eddy = compute_single_delta(
                field_ref_eddy_store[cycle_id],
                cycle_data.field_pred_eddy,
                "eddy current",
            )
            if running_field_ref_eddy is not None:
                # Update running eddy current reference
                delta_eddy = CorrectionHelper.apply_running_eddy_adjustment(
                    delta_eddy, running_field_ref_eddy, cycle_id
                )
            return DeltaComponents(combined=delta_eddy, eddy_current=delta_eddy)
        # Fallback to legacy references if separated components not available
        log.warning(
            f"[{cycle_data}] Eddy current components not available, falling back to legacy"
        )
        if cycle_data.field_pred is not None:
            delta = calc_delta_field(
                field_ref_store[cycle_id],
                cycle_data.field_pred,
                beam_in=beam_in,
                beam_out=beam_out,
                flatten=flatten,
            )
            return DeltaComponents(combined=delta)
        msg = f"[{cycle_data}] No field prediction available"
        raise ValueError(msg)

    if correction_mode == CorrectionMode.COMBINED:
        # Compute both deltas separately and combine them
        delta_hyst_combined: npt.NDArray[np.float64] | None = None
        delta_eddy_combined: npt.NDArray[np.float64] | None = None

        # Compute hysteresis delta if available
        if cycle_id in field_ref_hyst_store and cycle_data.field_pred_hyst is not None:
            delta_hyst_combined = compute_single_delta(
                field_ref_hyst_store[cycle_id], cycle_data.field_pred_hyst, "hysteresis"
            )

        # Compute eddy current delta if available
        if cycle_id in field_ref_eddy_store and cycle_data.field_pred_eddy is not None:
            delta_eddy_combined = compute_single_delta(
                field_ref_eddy_store[cycle_id],
                cycle_data.field_pred_eddy,
                "eddy current",
            )

            if running_field_ref_eddy is not None:
                # Update running eddy current reference
                delta_eddy_combined = CorrectionHelper.apply_running_eddy_adjustment(
                    delta_eddy_combined, running_field_ref_eddy, cycle_id
                )

        # Combine the deltas
        if delta_hyst_combined is not None and delta_eddy_combined is not None:
            # Interpolate both deltas to common time grid and add them
            log.debug(f"[{cycle_data}] Combining hysteresis and eddy current deltas")
            delta_hyst_interp, delta_eddy_interp = match_array_size(
                delta_hyst_combined, delta_eddy_combined
            )

            combined_delta = np.vstack((
                delta_hyst_interp[0],  # Use common time axis
                delta_hyst_interp[1] + delta_eddy_interp[1],  # Add field values
            ))

            return DeltaComponents(
                combined=combined_delta,
                hysteresis=delta_hyst_combined,
                eddy_current=delta_eddy_combined,
            )

        if delta_hyst_combined is not None:
            # Only hysteresis delta available
            log.debug(
                f"[{cycle_data}] Only hysteresis delta available in combined mode"
            )
            return DeltaComponents(
                combined=delta_hyst_combined, hysteresis=delta_hyst_combined
            )

        if delta_eddy_combined is not None:
            # Only eddy current delta available
            log.debug(
                f"[{cycle_data}] Only eddy current delta available in combined mode"
            )
            return DeltaComponents(
                combined=delta_eddy_combined, eddy_current=delta_eddy_combined
            )

        # Neither component available, fallback to legacy
        log.warning(
            f"[{cycle_data}] No separated components available, falling back to legacy"
        )
        if cycle_data.field_pred is not None:
            delta = calc_delta_field(
                field_ref_store[cycle_id],
                cycle_data.field_pred,
                beam_in=beam_in,
                beam_out=beam_out,
                flatten=flatten,
            )
            return DeltaComponents(combined=delta)
        msg = f"[{cycle_data}] No field prediction available"
        raise ValueError(msg)

    msg = f"Unknown correction mode: {correction_mode}"
    raise ValueError(msg)


def _cycle_id_helper(cycle_data: hystcomp_utils.cycle_data.CycleData) -> str:
    """Helper function to get cycle ID (same logic as _cycle_id static method)."""
    id_ = cycle_data.cycle
    if cycle_data.economy_mode is not hystcomp_utils.cycle_data.EconomyMode.NONE:
        if cycle_data.economy_mode is hystcomp_utils.cycle_data.EconomyMode.FULL:
            id_ += "_FULLECO"
        elif cycle_data.economy_mode is hystcomp_utils.cycle_data.EconomyMode.DYNAMIC:
            id_ += "_DYNECO"
    return id_


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


def calc_new_correction_with_trim_window(
    current_correction: npt.NDArray[np.float64],  # [2, n_points]
    delta: npt.NDArray[np.float64],  # [2, n_points]
    trim_start: float,  # ms
    trim_end: float,  # ms
    gain: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Calculate new correction applying delta only within trim window.

    Args:
        current_correction: Current correction [time, values]
        delta: Delta to apply [time, values]
        trim_start: Start time of trim window (ms)
        trim_end: End time of trim window (ms)
        gain: Gain factor for delta application

    Returns:
        New correction with delta applied only in trim window
    """
    log.debug(f"Current correction shape: {current_correction.shape}")
    log.debug(f"Delta shape: {delta.shape}")
    log.debug(f"Trim window: {trim_start} - {trim_end} ms")

    # Create union time axis and interpolate both arrays
    current_correction_interp, delta_interp = match_array_size(
        current_correction,
        delta,
    )

    # Get the common time axis
    time_axis = current_correction_interp[0]

    # Create mask for trim window (inclusive endpoints)
    trim_mask = (time_axis >= trim_start) & (time_axis <= trim_end)

    # Start with current correction values
    new_correction = current_correction_interp[1].copy()

    # Apply delta * gain only within trim window
    new_correction[trim_mask] += gain * delta_interp[1][trim_mask]

    log.debug(f"Applied correction to {np.sum(trim_mask)} points within trim window")

    return np.vstack((time_axis, new_correction))


def calc_new_correction(
    current_correction: npt.NDArray[np.float64],  # [2, n_points]
    delta: npt.NDArray[np.float64],  # [2, n_points]
    gain: float = 1.0,
) -> npt.NDArray[np.float64]:
    """Legacy function - kept for backward compatibility."""
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
