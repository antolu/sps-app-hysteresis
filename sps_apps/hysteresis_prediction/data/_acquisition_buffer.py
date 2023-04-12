"""
This module contains the AcquisitionBuffer class, which is used to
buffer and order the data acquired by the Acquisition class.
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime
from enum import Enum, auto

import numpy as np

from ..async_utils import Signal
from ._dataclass import SingleCycleData

__all__ = ["AcquisitionBuffer"]
log = logging.getLogger(__name__)


class BufferData(Enum):
    MEAS_I = auto()
    MEAS_B = auto()
    PROG_I = auto()
    PROG_B = auto()
    REF_B = auto()


class InsufficientDataError(Exception):
    pass


class AcquisitionBuffer:
    def __init__(self, min_buffer_size: int):
        self._buffer_size = min_buffer_size

        """ Reference values for creating samples without measured data """
        self._i_ref: dict[str, np.ndarray] = {}
        self._b_ref: dict[str, np.ndarray] = {}

        self._i_prog: dict[str, np.ndarray] = {}
        self._b_prog: dict[str, np.ndarray] = {}

        self._buffer: deque[SingleCycleData] = deque()

        # maps cycle time(stamp) to data
        self._buffered_cycles: dict[datetime, SingleCycleData] = {}
        self._next_cycles: dict[datetime, SingleCycleData] = {}

        self.new_buffered_data = Signal(list[SingleCycleData])

        self.DISPATCH_MAP = {
            BufferData.MEAS_I: self._new_measured_I,
            BufferData.MEAS_B: self._new_measured_B,
            BufferData.PROG_I: self._new_programmed_I,
            BufferData.PROG_B: self._new_programmed_B,
            BufferData.REF_B: self._new_reference_B,
        }

    def dispatch_data(
        self,
        dest: BufferData,
        cycle: str,
        cycle_timestamp: int,
        data: np.ndarray,
    ) -> None:
        """
        Dispatches the data to the appropriate method. This method is
        intended to be used by the :class:`Acquisition` class, or any other
        external callee that acquires data.

        This method catches any exceptions that may be raised by the
        dispatched methods.

        :param dest: The destination of the data. Represents the data type.
        :param cycle: The LSA cycle corresponding to the data.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        :param data: The data to be dispatched. Must be 2D array where first
            column is time (in CTime [ms]), and second is the data.
        """
        if dest not in self.DISPATCH_MAP:
            log.error(f"Invalid destination: {dest}.")
            return

        try:
            self.DISPATCH_MAP[dest](cycle, cycle_timestamp, data)
        except Exception:
            log.exception(f"Error while dispatching data to {dest}.")
            return

    def current_changed(self, cycle: str) -> None:
        pass

    def collate_samples(self) -> list[SingleCycleData]:
        """
        Collates samples from the buffered data. This requires the minimum
        number of samples to be above the threshold, otherwise an
        :class:`InsufficientDataError` is raised.

        The buffered samples are gathered into a list in ascending order
        by cycle time, and the samples belong to consecutive cycles.

        The collated samples can then be used to perform inference, or
        train models.

        :return: A list of :class:`SingleCycleData` objects.

        :raises InsufficientDataError: If the number of buffered samples is
            less than the minimum buffer size.
        """
        pass

    def _new_measured_I(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        pass

    def _new_measured_B(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        pass

    def _new_programmed_I(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        pass

    def _new_programmed_B(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        pass

    def _new_reference_B(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        pass
