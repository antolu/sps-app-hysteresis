from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qtpy.QtCore import QObject, Signal

from ...data import Acquisition, SingleCycleData
from ._sources import AcquiredDataType, CurrentFieldSource

log = logging.getLogger(__name__)


class PlotModel(QObject):
    new_predicted_cycle = Signal(SingleCycleData, np.ndarray)

    def __init__(
        self,
        acquisition: Acquisition,
        parent: Optional[QObject] = None,
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
        self._field_ref_discr_source = CurrentFieldSource(
            AcquiredDataType.PredictedField, downsample=downsample
        )

        self._acquisition.new_measured_data.connect(self._handle_new_measured)
        self._acquisition.new_programmed_cycle.connect(
            self._handle_new_programmed
        )
        self.new_predicted_cycle.connect(self._handle_new_predicted)

    def __del__(self) -> None:
        self._acquisition.new_measured_data.disconnect(
            self._handle_new_measured
        )
        self._acquisition.new_programmed_cycle.disconnect(
            self._handle_new_programmed
        )

    def _handle_new_measured(self, cycle_data: SingleCycleData) -> None:
        try:
            assert cycle_data.current_meas is not None
            assert cycle_data.field_meas is not None

            self._current_meas_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.current_meas
            )
            self._field_meas_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.field_meas
            )
        except Exception:  # noqa: broad-except
            log.exception(
                "An exception occurred while publishing new " "measured data."
            )
            return

    def _handle_new_programmed(self, cycle_data: SingleCycleData) -> None:
        try:
            self._current_prog_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.current_prog
            )
            self._field_prog_source.new_value(
                cycle_data.cycle_timestamp, cycle_data.field_prog
            )
        except Exception:  # noqa: broad-except
            log.exception(
                "An exception occurred while publishing new "
                "programmed data."
            )
            return

    def _handle_new_predicted(
        self, cycle_data: SingleCycleData, predicted: np.ndarray
    ) -> None:
        try:
            self._field_predict_source.new_value(
                cycle_data.cycle_timestamp, predicted
            )

            if cycle_data.field_ref is not None:
                log.debug(f"Plotting field diff for cycle {cycle_data.cycle}")
                discr = np.abs(cycle_data.field_ref - predicted)
                self._field_ref_discr_source.new_value(
                    cycle_data.cycle_timestamp, discr
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
    def field_ref_discr_source(self) -> CurrentFieldSource:
        return self._field_ref_discr_source

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
        self._field_ref_discr_source.downsample = value

    def set_downsample(self, value: int) -> None:
        self.downsample = value
