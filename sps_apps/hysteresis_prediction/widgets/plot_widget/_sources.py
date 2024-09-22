from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum

import numpy as np
from accwidgets.graph import CurveData, UpdateSource
from qtpy.QtCore import QTimer

__all__ = ["LocalTimerTimingSource"]

log = logging.getLogger(__name__)


MS = 1e3
NS = 1e9


class AcquiredDataType(Enum):
    MeasuredData = "MEASURED_DATA"
    ProgrammedData = "PROGRAMMED_DATA"
    PredictedField = "PREDICTED_FIELD"


class LocalTimerTimingSource(UpdateSource):
    def __init__(self, offset: float = 0.0):
        """
        Class for sending timing-update signals based on a QTimer instance.

        Args:
            offset: offset of the emitted time to the actual current time
        """
        super().__init__()
        self.timer = QTimer(self)
        self.offset = offset
        self.timer.timeout.connect(self._create_new_value)
        self.timer.start(int(1000 / 30))

    def _create_new_value(self) -> None:
        self.sig_new_timestamp.emit(datetime.now().timestamp() + self.offset)


class CurrentFieldSource(UpdateSource):
    def __init__(
        self, acquired_data_type: AcquiredDataType, downsample: int = 1
    ) -> None:
        super().__init__()

        self.acquired_data_type = acquired_data_type
        self.downsample = downsample

    def new_value(self, cycle_timestamp: float, value: np.ndarray) -> None:
        """
        Pass a new value to the source.

        :param cycle_timestamp: timestamp of the cycle.
        :param value: value of the cycle.
        """
        self._handle_new_value(cycle_timestamp, value)

    def _handle_new_value(self, cycle_timestamp: float, value: np.ndarray) -> None:
        if self.acquired_data_type == AcquiredDataType.MeasuredData:
            self._handle_measured_value(cycle_timestamp, value)
        elif self.acquired_data_type == AcquiredDataType.ProgrammedData:
            self._handle_programmed_value(cycle_timestamp, value)
        elif self.acquired_data_type == AcquiredDataType.PredictedField:
            self._handle_predicted_value(cycle_timestamp, value)
        else:
            log.error(
                f"Acquired data type of type {self.acquired_data_type}"
                "is not of instance AcquiredDataType."
            )
            return

    def _handle_programmed_value(
        self, cycle_timestamp: float, value: np.ndarray
    ) -> None:
        """
        Handle new programmed value. The value is a 2 row array, where
        the first row is the time axis in ms, and the second is the
        value in A or T.

        :param cycle_timestamp: timestamp of the cycle.
        :param value: value of the cycle.
        """
        time, value_ = value
        num_samples_ms = time[-1] - time[0]

        time_axis = time / MS + cycle_timestamp / NS
        time_interp = np.arange(num_samples_ms) / MS + cycle_timestamp / NS

        value_interp = np.interp(time_interp, time_axis, value_)
        data = CurveData(
            x=time_interp[:: self.downsample],
            y=value_interp[:: self.downsample],
        )
        self.send_data(data)

    def _handle_measured_value(self, cycle_timestamp: float, value: np.ndarray) -> None:
        """
        Handle new measured value. The value is a single array with the
        measured values, without time axis. The time axis is built based
        on the length of the array. The sampling rate is assumed to be
        1kHz.
        """
        time_range = np.arange(len(value)) / MS + cycle_timestamp / NS
        value = value.flatten()

        data = CurveData(x=time_range[:: self.downsample], y=value[:: self.downsample])
        self.send_data(data)

    def _handle_predicted_value(
        self, cycle_timestamp: float, value: np.ndarray
    ) -> None:
        """
        Handle new predicted value. The value is a single array with the
        predicted values, without time axis. The value is assumed to be
        with a sampling rate of 1kHz.

        :param cycle_timestamp: timestamp of the cycle.
        :param value: value of the cycle.
        """
        time_range = value[0, :]
        value = value[1, :]

        data = CurveData(time_range, value)
        self.send_data(data)
