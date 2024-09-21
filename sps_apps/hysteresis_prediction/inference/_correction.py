from __future__ import annotations

import datetime
import logging

import numpy as np
import numpy.typing as npt
from qtpy import QtCore

import hystcomp_utils.cycle_data

log = logging.getLogger(__name__)


class CalculateCorrection(QtCore.QObject):
    newCorrectionAvailable = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

        self._gain = 1.0
        self._flatten = False

        # map cycle name to field reference [2, n_points]
        self._field_ref: dict[str, npt.NDArray[np.float64]] = {}
        self._field_ref_timestamps: dict[str, float] = {}

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycle(self, cycle: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"{cycle}: Calculating correction."
        log.debug(msg)

        self._maybe_save_reference(cycle)

        if cycle.field_pred is None:
            log.error(
                f"{cycle}: Prediction field not set. Cannot calculate correction."
            )
            return

        delta = calc_delta_field(
            self._field_ref[cycle.cycle], cycle.field_pred
        )
        cycle.delta_applied = delta

        msg = f"{cycle}: Delta calculated."
        log.debug(msg)

        if (
            cycle.correction is not None
        ):  # and not cycle.cycle.endswith("ECO"):
            correction = calc_new_correction(
                cycle.correction, delta, self._gain
            )
            cycle.correction = correction

            msg = f"{cycle}: New correction calculated."
            log.debug(msg)

        self.newCorrectionAvailable.emit(cycle)

    @QtCore.Slot(str)
    def resetReference(self, cycle_name: str | None = None) -> None:
        if cycle_name is None or cycle_name == "all":
            log.debug("Resetting all field references.")
            self._field_ref.clear()
            return

        if cycle_name in self._field_ref:
            log.debug(f"{cycle_name}: Resetting field reference.")
            del self._field_ref[cycle_name]
        else:
            log.debug(
                f"{cycle_name}: Field reference not set. Nothing to reset."
            )
        ...

    def _maybe_save_reference(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        if cycle_data.field_pred is None:
            log.error(
                f"{cycle_data}: Prediction field not set. Cannot save reference."
            )
            return

        if (
            cycle_data.reference_timestamp is not None
            and np.allclose(
                cycle_data.cycle_timestamp, cycle_data.reference_timestamp
            )
            and cycle_data.cycle.endswith("ECO")
        ):
            msg = f"{cycle_data}: Last cycle was ECO, need to delete the last reference."
            log.debug(msg)

            cycle_name = "_".join(cycle_data.cycle.split("_")[:-1])
            self._reset_reference(cycle_name)
            cycle_data.reference_timestamp = None

        if cycle_data.cycle not in self._field_ref:
            log.debug(
                f"{cycle_data}: Saving field reference since it has not "
                f"been set ({cycle_data.field_pred.shape})."
            )
            cycle_data.reference_timestamp = cycle_data.cycle_timestamp
            cycle_data.field_ref = cycle_data.field_pred
            self._field_ref[cycle_data.cycle] = cycle_data.field_pred
            self._field_ref_timestamps[cycle_data.cycle] = (
                cycle_data.reference_timestamp
            )
        else:
            ref_time = datetime.datetime.fromtimestamp(
                self._field_ref_timestamps[cycle_data.cycle] * 1e-9
            )
            log.debug(
                f"{cycle_data}: Reference already saved from timestamp {ref_time}. "
                f"Adding the reference to the cycle data."
            )

            cycle_data.reference_timestamp = self._field_ref_timestamps[
                cycle_data.cycle
            ]
            cycle_data.field_ref = self._field_ref[cycle_data.cycle]


def calc_delta_field(
    field_ref: npt.NDArray[np.float64],
    field_pred: npt.NDArray[np.float64],
    beam_in: float | None = None,
    beam_out: float | None = None,
    *,
    flatten: bool = False,
) -> npt.NDArray[np.float64]:
    # calc delta and smooth it
    ref_t = (
        field_ref[0, :] - field_ref[0, 0]
    ) * 1e3  # absolute time in s to ms
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

    ref_v = np.interp(
        pred_t,
        np.array([beam_in or ref_t[0], beam_out or ref_t[-1]]),
        np.array([pred_v[0], pred_v[0]]),
    )

    # cut trim beyond time limits
    delta_v = ref_v - pred_v
    delta_t = ref_t
    if beam_in is not None and beam_out is not None:
        delta_t, delta_v = cut_trim_beyond_time(
            delta_t, delta_v, beam_in, beam_out
        )
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
    current_correction: npt.NDArray[np.float64],
    new_correction: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if current_correction[0].size < new_correction[0].size:
        # upsample LSA trim to match prediction, keep BP edges
        new_x = np.concatenate((new_correction[0], current_correction[0]))
        new_x = np.sort(np.unique(new_x))
        current_correction = np.interp(
            new_x,
            *current_correction,
        )

        # upsample prediction to match new LSA trim
        new_correction = np.interp(
            new_x,
            *new_correction,
        )
    elif current_correction[0].size > new_correction[0].size:
        # upsample prediction to match LSA trim
        new_correction = np.interp(
            current_correction[0],
            *new_correction,
        )
        new_x = current_correction[0]
    else:
        new_x = np.concatenate((new_correction[0], current_correction[0]))
        new_x = np.sort(np.unique(new_x))
        current_correction = np.interp(
            new_x,
            *current_correction,
        )

        new_correction = np.interp(
            new_x,
            *new_correction,
        )

    return np.vstack((new_x, current_correction)), np.vstack(
        (new_x, new_correction)
    )


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
    new_correction = (current_correction + gain * delta[1]).astype(np.float64)

    # smooth the correction
    # new_correction = signal.perona_malik_smooth(
    #     new_correction, 5.0, 5e-2, 2.0
    # )

    # return delta for plotting
    return np.vstack((current_correction[0], new_correction))
