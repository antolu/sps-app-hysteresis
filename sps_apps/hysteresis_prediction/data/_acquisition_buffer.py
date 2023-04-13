"""
This module contains the AcquisitionBuffer class, which is used to
buffer and order the data acquired by the Acquisition class.
"""
from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from threading import Lock

import numpy as np

from ..async_utils import Signal
from ._dataclass import SingleCycleData

__all__ = ["AcquisitionBuffer"]
log = logging.getLogger(__name__)


class BufferData(Enum):
    MEAS_I = "measured_I"
    MEAS_B = "measured_B"
    PROG_I = "programmed_I"
    PROG_B = "programmed_B"
    REF_B = "reference_B"


class InsufficientDataError(Exception):
    pass


class AcquisitionBuffer:
    def __init__(self, min_buffer_size: int):
        """
        :param min_buffer_size: The minimum number of samples required to
            collate samples from the buffer.
        """
        self._buffer_size = min_buffer_size

        """ Reference values for creating samples without measured data """
        self._i_ref: dict[str, np.ndarray] = {}
        self._b_ref: dict[str, np.ndarray] = {}

        self._i_prog: dict[str, np.ndarray] = {}
        self._b_prog: dict[str, np.ndarray] = {}

        self._buffer: deque[SingleCycleData] = deque()
        self._buffer_next: deque[SingleCycleData] = deque()

        # maps cycle time(stamp) to data
        self._buffered_cycles: dict[int, SingleCycleData] = {}
        self._next_cycles: dict[int, SingleCycleData] = {}

        self.new_buffered_data = Signal(list[SingleCycleData])

        self._lock = Lock()

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

    def new_cycle(self, cycle: str, cycle_timestamp: int) -> None:
        """
        Called when a new cycle is started (or is going to start).
        This will create a new :class:`SingleCycleData` object and add
        it to the next cycles buffer with the programmed I and B,
        pending the measured I and B.

        This will also trigger an integrity check of the buffered data.

        :param cycle: The LSA cycle name of the cycle being triggered.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        """

        if cycle not in self._i_prog or self._i_prog[cycle] is None:
            log.debug(
                "Programmed current has not yet been set. "
                "Cannot create new cycle data."
            )
            return

        if cycle not in self._b_prog or self._b_prog[cycle] is None:
            log.debug(
                "Programmed field has not yet been set. "
                "Cannot create new cycle data."
            )
            return

        cycle_data = SingleCycleData(
            cycle, cycle_timestamp, self._i_prog[cycle], self._b_prog[cycle]
        )

        with self._lock:
            self._next_cycles[cycle_timestamp] = cycle_data
            self._buffer_next.append(cycle_data)

        self._check_buffer_integrity()
        self._check_buffer_size()

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

        def buffer_size(buffer: deque[SingleCycleData]) -> int:
            return sum([o.cycle_length for o in buffer])

        with self._lock:
            if (
                buffer_size(self._buffer) + buffer_size(self._buffer_next)
                < self._buffer_size
            ):
                raise InsufficientDataError(
                    f"Buffer size is less than {self._buffer_size}."
                )

            out_buffer = list(self._buffer)
            incomplete_samples = list(self._buffer_next)

            for cycle_data in out_buffer:
                cycle_data.current_input = cycle_data.current_meas

            for cycle_data in incomplete_samples:
                if cycle_data.cycle not in self._i_ref:
                    log.error(
                        f"Missing reference current for {cycle_data.cycle}. "
                        "Cannot build buffer."
                    )
                    return
                cycle_data.current_input = self._i_ref[cycle_data.cycle]

        out_buffer += incomplete_samples

        log.debug(f"Collated {len(out_buffer)} samples.")
        return out_buffer

    def _new_measured_I(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        """
        Adds a new measured current to the buffer. If the current is not
        already saved as a reference it will be saved. This will also
        find the corresponding buffered cycle data and save it to the
        data structure. If the buffered cycle data is complete, i.e.
        has both measured I and B, it will be moved to the buffer queue.

        :param cycle: The LSA cycle corresponding to the data.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        :param value: The new measured current value.
        """
        if cycle not in self._i_ref or self._i_ref[cycle] is None:
            log.debug("Measured current has not yet been set.")
            log.debug(f"Setting new measured current for cycle {cycle}.")

            with self._lock:
                self._i_ref[cycle] = value

            return

        if cycle_timestamp not in self._buffer_next:
            log.debug(f"Next buffered cycle data for cycle {cycle} not found.")
            return

        with self._lock:
            cycle_data = self._buffer_next[cycle_timestamp]
            cycle_data.current_meas = value

        self._check_move_to_buffer(cycle_data)

    def _new_measured_B(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        """
        Adds a new measured magnetic field to the buffer. This will also
        find the corresponding buffered cycle data and save it to the
        data structure. If the buffered cycle data is complete, i.e.
        has both measured I and B, it will be moved to the buffer queue.

        :param cycle: The LSA cycle corresponding to the data.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        :param value: The new measured magnetic field value.
        """
        if cycle not in self._b_ref or self._b_ref[cycle] is None:
            log.debug("Measured magnetic field has not yet been set.")
            log.debug(
                f"Setting new measured magnetic field for cycle {cycle}."
            )

            with self._lock:
                self._b_ref[cycle] = value

            return

        if cycle_timestamp not in self._buffer_next:
            log.debug(f"Next buffered cycle data for cycle {cycle} not found.")
            return

        with self._lock:
            cycle_data = self._buffer_next[cycle_timestamp]
            cycle_data.field_meas = value

        self._check_move_to_buffer(cycle_data)

    def _new_programmed_I(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        """
        Adds a new programmed current to the buffer. Only the one value is
        saved, and is used to determine if the current has changed.

        :param cycle: The LSA cycle corresponding to the data.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        :param value: The new programmed current value.
        """

        def set_new_programmed_current():
            log.debug(f"Setting new programmed current for cycle {cycle}.")

            with self._lock:
                self._i_prog[cycle] = value

            log.debug(f"Removing buffered reference data for cycle {cycle}.")
            if cycle not in self._i_ref:
                log.debug(f"No buffered reference data for cycle {cycle}.")
                return

            with self._lock:
                self._i_ref.pop(cycle)

            return

        if cycle not in self._i_prog or self._i_prog[cycle] is None:
            log.debug("Programmed current has not yet been set.")
            set_new_programmed_current()
        else:
            old_value = self._i_prog[cycle]

            if len(old_value) != len(value):
                log.debug("Programmed current length has changed.")
                set_new_programmed_current()

            elif not np.allclose(old_value, value):
                log.debug("Programmed current values has changed.")
                set_new_programmed_current()

            else:
                log.debug(
                    f"Programmed current has not changed for cycle {cycle}."
                )

    def _new_programmed_B(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        log.warning("Programmed magnetic field not yet implemented.")

    def _new_reference_B(
        self, cycle: str, cycle_timestamp: int, value: np.ndarray
    ) -> None:
        log.warning("Reference magnetic field not yet implemented.")

    def _check_buffer_integrity(self) -> None:
        """
        Check the buffered cycle deques to ensure the consecutiveness
        of the data. I.e. that newly buffered data is not older than
        existing data.
        """
        if len(self._buffer) < 2:
            log.debug(
                "Buffer integrity check not required. Only"
                f"{len(self._buffer)} buffered cycles."
            )
            return

        def check_integrity(buffer: deque) -> list[int]:
            to_remove = []
            with self._lock:
                buffer = buffer.copy()
            for i in range(1, len(buffer)):
                cycle_data = buffer[i]
                prev_cycle_data = buffer[i - 1]

                cycle = cycle_data.cycle
                cycle_time = cycle_data.cycle_time

                prev_cycle = prev_cycle_data.cycle
                prev_cycle_time = prev_cycle_data.cycle_time

                if cycle_time < prev_cycle_time:
                    log.warning(
                        f"Buffered cycle {cycle}@{cycle_time} is older than "
                        f"previous cycle {prev_cycle}@{prev_cycle_time}. "
                        "Buffer integrity may be compromised. Removing "
                        "offending buffered cycle."
                    )
                    to_remove.append(i)

            return to_remove

        log.debug("Checking buffer integrity.")
        to_remove = check_integrity(self._buffer)
        if len(to_remove) > 0:
            for i in reversed(to_remove):
                log.info(f"Removing buffered cycle {self._buffer[i].cycle}.")
                with self._lock:
                    self._buffer.pop(i)
                    self._buffered_cycles.pop(self._buffer[i].cycle_timestamp)

            to_remove = check_integrity(self._buffered_cycles)
            if len(to_remove) > 0:
                log.error(
                    "Buffer integrity compromised. Could not fix "
                    "automatically. Clearing buffer."
                )
                with self._lock:
                    self._buffer.clear()
                    self._buffered_cycles.clear()

        log.debug("Checking NEXT buffer integrity.")
        to_remove = check_integrity(self._next_cycles)
        if len(to_remove) > 0:
            for i in reversed(to_remove):
                log.info(
                    f"Removing buffered cycle {self._next_cycles[i].cycle}."
                )
                with self._lock:
                    self._next_cycles.pop(i)
                    self._buffer_next.pop(self._next_cycles[i].cycle_timestamp)

            to_remove = check_integrity(self._buffer_next)
            if len(to_remove) > 0:
                log.error(
                    "Buffer integrity compromised. Could not fix "
                    "automatically. Clearing buffer."
                )
                with self._lock:
                    self._next_cycles.clear()
                    self._buffer_next.clear()

    def _check_buffer_size(self) -> None:
        """
        Checks the buffer size and removes the oldest data if the buffer
        is too large.

        The total buffer size is calculated as the sum of the number of
        buffered cycles and the number of buffered NEXT cycles, as
        they will be combined into a single buffer when then the
        :meth:`collate_samples` method is called.`
        """
        if len(self._buffer) == 0:
            log.debug("No buffered cycles.")
            return

        if len(self._buffer) == 1:
            log.debug(f"Insufficient buffered cycles ({len(self._buffer)}).")
            return

        if len(self._buffer_next) == 0:
            log.debug("No buffered NEXT cycles.")
            return

        def buffer_size(buffer: deque) -> int:
            return sum([o.sample_length for o in buffer])

        def buffer_too_large(buffer: deque, buffer_next: deque) -> bool:
            num_samples_buffer = [o.sample_length for o in buffer]
            num_samples_next = [o.sample_length for o in buffer_next]

            return (
                sum(num_samples_buffer) + sum(num_samples_next)
                > self._buffer_size
            )

        if not buffer_too_large(self._buffer, self._next_cycles):
            log.debug(
                "Buffer size is within limits. "
                f"({buffer_size(self._buffer) + buffer_size(self._buffer_next)}/"  # noqa E501
                f"{self._buffer_size})"
            )
            return
        else:
            while buffer_too_large(
                self._buffer, self._next_cycles
            ) and not buffer_too_large(self._buffer[1:], self._next_cycles):
                log.debug(
                    "Buffer size is too large. Removing oldest buffered cycle. "  # noqa E501
                    f"({buffer_size(self._buffer) + buffer_size(self._buffer_next)}/"  # noqa E501
                    f"{self._buffer_size})"
                )
                with self._lock:
                    cycle_data = self._buffer.popleft()
                    log.debug(f"Removing buffered cycle {cycle_data.cycle}.")
                    self._buffered_cycles.pop(cycle_data.cycle_timestamp)

    def _check_move_to_buffer(self, cycle_data: SingleCycleData) -> None:
        """
        Checks if the buffered cycle data is complete, and if so moves it
        to the buffer queue.

        :param cycle_data: The buffered cycle data to check.
        """
        cycle = cycle_data.cycle
        cycle_timestamp = cycle_data.cycle_timestamp

        if cycle_timestamp not in self._buffer_next:
            log.debug(f"Next buffered cycle data for cycle {cycle} not found.")
            return
        elif cycle_timestamp in self._buffered_cycles:
            log.debug(f"Buffered cycle data for cycle {cycle} already exists.")
            return

        if (
            cycle_data.current_meas is not None
            and cycle_data.field_meas is not None
        ):
            log.debug(
                f"Moving buffered cycle data for cycle {cycle} to buffer "
                "queue."
            )
            if self._next_cycles[0] is not cycle_data:
                log.error("Buffered cycle data is not the next cycle data.")
                return

            with self._lock:
                self._next_cycles.popleft()
                self._buffer_next.pop(cycle_timestamp)

                self._buffer_queue.append(cycle_data)
                self._buffered_cycles[cycle_timestamp] = cycle_data

            self.new_buffered_data.emit(cycle_data)
