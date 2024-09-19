from __future__ import annotations

import logging

import numpy as np
from qtpy import QtCore

from ...data import Acquisition
from hystcomp_utils.cycle_data import CycleData
from ._sources import AcquiredDataType, CurrentFieldSource
from transformertf.data import downsample as downsample_tf

log = logging.getLogger(__name__)


class PlotModel(QtCore.QObject):
    def __init__(
        self,
        acquisition: Acquisition,
        parent: QtCore.QObject | None = None,
        downsample: int = 100,
    ) -> None:
        super().__init__(parent=parent)

        self._downsample = downsample
        self._acquisition = acquisition

        self._current_meas_source = CurrentFieldSource(
            AcquiredDataType.MeasuredData, downsample=downsample
        )
        self._field_meas_source = CurrentFieldSource(
            AcquiredDataType.MeasuredData, downsample=downsample
        )
        self._current_prog_source = CurrentFieldSource(
            AcquiredDataType.ProgrammedData, downsample=downsample
        )
        self._field_prog_source = CurrentFieldSource(
            AcquiredDataType.ProgrammedData, downsample=downsample
        )
        self._field_predict_source = CurrentFieldSource(
            AcquiredDataType.PredictedField, downsample=downsample
        )
        self._field_ref_diff_source = CurrentFieldSource(
            AcquiredDataType.PredictedField, downsample=downsample
        )
        self._field_meas_diff_source = CurrentFieldSource(
            AcquiredDataType.PredictedField, downsample=downsample
        )

        self._acquisition.new_measured_data.connect(self._handle_new_measured)
        self._acquisition.sig_new_programmed_cycle.connect(
            self._handle_new_programmed
        )

    def __del__(self) -> None:
        self._acquisition.new_measured_data.disconnect(
            self._handle_new_measured
        )
        self._acquisition.sig_new_programmed_cycle.disconnect(
            self._handle_new_programmed
        )

    def _handle_new_measured(self, cycle_data: CycleData) -> None:
        try:
            assert cycle_data.current_meas is not None
            assert cycle_data.field_meas is not None

            self._current_meas_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.current_meas
            )
            self._field_meas_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.field_meas
            )

            if cycle_data.field_pred is not None:
                field_pred = cycle_data.field_pred
                downsample_factor = (
                    cycle_data.field_meas.size // field_pred.shape[-1]
                )
                delta = (
                    downsample_tf(
                        cycle_data.field_meas, downsample_factor, "interval"
                    )
                    - field_pred[1, :]
                ) * 1e4
                self._field_meas_diff_source.new_value(
                    cycle_data.cycle_timestamp,
                    np.stack(
                        (field_pred[0, :], delta),
                        axis=0,
                    ),
                )

        except Exception:  # noqa: broad-except
            log.exception(
                "An exception occurred while publishing new " "measured data."
            )
            return

    def _handle_new_programmed(self, cycle_data: CycleData) -> None:
        try:
            self._current_prog_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.current_prog
            )
            # self._field_prog_source.new_value(
            #     cycle_data.cycle_timestamp, cycle_data.field_prog
            # )
        except Exception:  # noqa: broad-except
            log.exception(
                "An exception occurred while publishing new "
                "programmed data."
            )
            return

    @QtCore.Slot(CycleData, np.ndarray)
    def onNewPredicted(
        self, cycle_data: CycleData, predicted: np.ndarray
    ) -> None:
        try:
            self._field_predict_source.new_value(
                cycle_data.cycle_timestamp, predicted
            )

            if cycle_data.field_ref is not None:
                log.debug(f"Plotting field diff for cycle {cycle_data.cycle}")
                delta = (cycle_data.field_ref[1, :] - predicted[1, :]) * 1e4
                self._field_ref_diff_source.new_value(
                    cycle_data.cycle_timestamp,
                    np.stack([predicted[0, :], delta], axis=0),
                )
        except Exception:  # noqa: broad-except
            log.exception(
                "An exception occurred while publishing new predicted data."
            )
            return

    @property
    def current_meas_source(self) -> CurrentFieldSource:
        return self._current_meas_source

    @property
    def field_meas_source(self) -> CurrentFieldSource:
        return self._field_meas_source

    @property
    def current_prog_source(self) -> CurrentFieldSource:
        return self._current_prog_source

    @property
    def field_prog_source(self) -> CurrentFieldSource:
        return self._field_prog_source

    @property
    def field_predict_source(self) -> CurrentFieldSource:
        return self._field_predict_source

    @property
    def field_ref_diff_source(self) -> CurrentFieldSource:
        return self._field_ref_diff_source

    @property
    def field_meas_diff_source(self) -> CurrentFieldSource:
        return self._field_meas_diff_source

    @property
    def downsample(self) -> int:
        return self._downsample

    @downsample.setter
    def downsample(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"downsample must be int, not {type(value)}.")
        if value < 1:
            raise ValueError(f"downsample must be >= 1, not {value}.")

        self._downsample = value

        self._current_meas_source.downsample = value
        self._field_meas_source.downsample = value
        self._current_prog_source.downsample = value
        self._field_prog_source.downsample = value
        self._field_predict_source.downsample = value
        self._field_ref_diff_source.downsample = value

    def set_downsample(self, value: int) -> None:
        self.downsample = value
