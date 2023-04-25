"""
This module contains the AcquisitionBuffer class, which is used to
buffer and order the data acquired by the Acquisition class.
"""
from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from threading import Lock
from typing import Iterable, Optional, Union

import numpy as np

from ..async_utils import Signal
from ..utils import from_timestamp
from ._dataclass import SingleCycleData

__all__ = [
    "AcquisitionBuffer",
    "BufferData",
    "BufferSignal",
    "InsufficientDataError",
]
log = logging.getLogger(__name__)


def log_cycle(
    msg: str, cycle: str, timestamp: Optional[Union[int, float]] = None
) -> None:
    if timestamp is None:
        timestamp_s = ""
    else:
        timestamp_s = "@" + str(
            from_timestamp(timestamp, from_utc=True, unit="ns")
        )

    log.debug(f"[{cycle}{timestamp_s}] " + msg)


def _cycle_buffer_str(buffer: Iterable[SingleCycleData]) -> str:
    return ", ".join(
        [cycle.cycle + "@" + str(cycle.cycle_time) for cycle in buffer]
    )


class BufferData(Enum):
    MEAS_I = "measured_I"
    MEAS_B = "measured_B"
    PROG_I = "programmed_I"
    PROG_B = "programmed_B"
    REF_B = "reference_B"


class BufferSignal(Enum):
    CYCLE_START = "cycle_start"
    DYNECO = "DYNECO"
    FOREWARNING = "forewarning"


class InsufficientDataError(Exception):
    pass


class AcquisitionBuffer:
    def __init__(self, min_buffer_size: int) -> None:
        """
        :param min_buffer_size: The minimum number of samples required to
            collate samples from the buffer.
        """
        self._buffer_size = min_buffer_size

        """ Reference values for creating samples without measured data """
        self._i_ref: dict[str, np.ndarray] = {}
        self._b_meas: dict[str, np.ndarray] = {}
        self._b_ref: dict[str, np.ndarray] = {}

        self._i_prog: dict[str, np.ndarray] = {}
        self._b_prog: dict[str, np.ndarray] = {}

        self._buffer: deque[SingleCycleData] = deque()
        self._buffer_next: deque[SingleCycleData] = deque()

        # maps cycle time(stamp) to data
        self._cycles_index: dict[Union[int, float], SingleCycleData] = {}
        self._cycles_next_index: dict[Union[int, float], SingleCycleData] = {}

        self.new_buffered_data = Signal(list[SingleCycleData])
        self.new_measured_data = Signal(SingleCycleData)
        self.new_programmed_cycle = Signal(SingleCycleData)
        self.buffer_size_changed = Signal(int)

        self._unknown_cycles: dict[int, SingleCycleData] = {}

        self._lock = Lock()

        self.DISPATCH_MAP = {
            BufferData.MEAS_I: self._new_measured_I,
            BufferData.MEAS_B: self._new_measured_B,
            BufferData.PROG_I: self._new_programmed_I,
            BufferData.PROG_B: self._new_programmed_B,
            BufferData.REF_B: self._new_reference_B,
        }

        self._SIGNAL_MAP = {
            BufferSignal.CYCLE_START: self.on_start_cycle,
            BufferSignal.DYNECO: self._handle_dyneco,
            BufferSignal.FOREWARNING: self.new_cycle,
        }

    def dispatch_signal(
        self, dest: BufferSignal, cycle: str, cycle_timestamp: float
    ) -> None:
        """
        Dispatches the signal to the appropriate method. This method is
        intended to be used by the :class:`Acquisition` class, or any
        other external callee that acquires data.`

        This method catches any exceptions that may be raised by the
        dispatched methods.

        :param dest: The destination of the signal. Represents the signal type.
        :param cycle: The LSA cycle corresponding to the signal.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        """
        if dest not in self._SIGNAL_MAP:
            log.error(f"Invalid signal: {dest}.")
            return

        try:
            self._SIGNAL_MAP[dest](cycle, cycle_timestamp)
        except Exception:
            log.exception(f"Error while dispatching signal to {dest}.")
            return

    def dispatch_data(
        self,
        dest: BufferData,
        cycle: str,
        cycle_timestamp: Union[int, float],
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

    def new_cycle(
        self, cycle: str, cycle_timestamp: Union[int, float]
    ) -> None:
        """
        Called when a new cycle is started (or is going to start).
        This will create a new :class:`SingleCycleData` object and add
        it to the next cycles buffer with the programmed I and B,
        pending the measured I and B.

        This will also trigger an integrity check of the buffered data.

        :param cycle: The LSA cycle name of the cycle being triggered.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        """

        log_cycle("Triggered by new cycle arrival.", cycle, cycle_timestamp)
        if cycle not in self._i_prog or self._i_prog[cycle] is None:
            log_cycle(
                "Programmed current has not yet been set. "
                "Cannot create new cycle data.",
                cycle,
                cycle_timestamp,
            )
            return

        if cycle not in self._b_prog or self._b_prog[cycle] is None:
            log_cycle(
                "Programmed field has not yet been set. "
                "Cannot create new cycle data.",
                cycle,
                cycle_timestamp,
            )
            return

        cycle_data = SingleCycleData(
            cycle, cycle_timestamp, self._i_prog[cycle], self._b_prog[cycle]
        )
        if cycle in self._b_ref:
            log_cycle("Adding reference field.", cycle, cycle_timestamp)
            cycle_data.field_ref = self._b_ref[cycle]
        else:
            log_cycle(
                "No reference field available, not adding it to "
                "buffered cycle data.",
                cycle,
                cycle_timestamp,
            )

        if cycle_timestamp in self._cycles_next_index:
            log.error(
                f"[{cycle}@{from_timestamp(cycle_timestamp)}] "
                "Cycle data already exists in NEXT buffer. "
                f"Shifting new cycle by "
                f"{self._buffer_next[-1].num_samples * 1e6} ms."
            )

            # return
            # below is only a fix for FCY cycle timestamp being incorrect
            if len(self._unknown_cycles) > 1:
                log.warning(
                    "There exists more than one unknown cycle. "
                    "Don't know what to do."
                )
                return

            prev_cycle = self._buffer_next[-1]
            cycle_timestamp = (
                prev_cycle.cycle_timestamp + prev_cycle.num_samples * int(1e6)
            )
            with self._lock:
                self._unknown_cycles[cycle_timestamp] = cycle_data

        log_cycle(
            "Creating new cycle data in NEXT buffer.", cycle, cycle_timestamp
        )
        with self._lock:
            self._cycles_next_index[cycle_timestamp] = cycle_data
            self._buffer_next.append(cycle_data)

        self._check_buffer_integrity()
        self._check_buffer_size()

        self.new_programmed_cycle.emit(cycle_data)

    def on_start_cycle(
        self, cycle: str, cycle_timestamp: Union[int, float]
    ) -> None:
        """
        This method is implemented only to cope with the timing cycle
        timestamp being incorrect for two following cycles.

        This method checks if the latest added cycle in NEXT buffer is
        correct, and corrects the cycle timestamp if required.

        :param cycle The LSA cycle name.
        :param cycle_timestamp The unique (and correct) timestamp of the
        cycle.
        """
        log_cycle("Start cycle event received.", cycle, cycle_timestamp)

        if len(self._buffer_next) == 0:
            log.debug("No cycles in NEXT buffer. No need to change timestamp.")
            return

        last_cycle = self._buffer_next[-1]

        if last_cycle.cycle_timestamp == cycle_timestamp:
            log_cycle(
                "Cycle timestamp is identical to last data in NEXT "
                "buffer. No need to correct.",
                cycle,
                cycle_timestamp,
            )
            return

        time_descr = abs(last_cycle.cycle_timestamp - cycle_timestamp) / 1e6

        if time_descr < 5:  # diff less than 5 ms
            log_cycle(
                "Diff in cycle timestamp between last cycle in NEXT buffer"
                "and started cycle is not the same, but less than 5 ms. "
                f"Setting the cycle timestamp to {cycle_timestamp}.",
                cycle,
                cycle_timestamp,
            )

            with self._lock:
                self._cycles_next_index.pop(last_cycle.cycle_timestamp)
                last_cycle.cycle_timestamp = cycle_timestamp
                last_cycle.__post_init__()
                self._cycles_next_index[cycle_timestamp] = last_cycle

            return
        else:
            log.warning(
                f"Time discrepancy {time_descr} ms between last cycle in "
                "NEXT buffer and started cycle is not the same, and "
                f"greater than 5 ms: {last_cycle} -> "
                f"{cycle}@{from_timestamp(cycle_timestamp)}."
            )
            return

        log.warning("Don't know what to do here.")

    def _handle_dyneco(
        self, cycle: str, cycle_timestamp: Union[int, float]
    ) -> None:
        """
        This method handles the dynamic / partial economy timing information.
        The cycle associated with the cycle timestamp will have its LSA cycle
        name appended with _DYNECO. This effectively creates a new cycle
        in the buffer.

        Cycles running in dyneco will therefore require require its own set
        of measured data for prediction, and not reference current as normal
        cycles.
        """
        log_cycle("Triggered by DYNECO signal.", cycle, cycle_timestamp)

        if len(self._buffer_next) == 0:
            log_cycle(
                "NEXT buffer is empty. No need to handle DYNECO signal.",
                cycle,
                cycle_timestamp,
            )
            return

        if cycle_timestamp not in self._cycles_next_index:
            log_cycle(
                "Could not find cycle in NEXT index. "
                "Skipping DYNECO signal.",
                cycle,
                cycle_timestamp,
            )
            return

        cycle_data = self._cycles_next_index[cycle_timestamp]

        dyneco_cycle = cycle_data.cycle + "_DYNECO"
        log_cycle(
            f"Changing cycle name to {dyneco_cycle}.", cycle, cycle_timestamp
        )
        with self._lock:
            cycle_data.cycle = dyneco_cycle

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
        log.debug("Buffer asked to collate samples.")

        with self._lock:
            total_samples = buffer_size(self._buffer) + buffer_size(
                self._buffer_next
            )
            if total_samples < self._buffer_size:
                raise InsufficientDataError(
                    f"Buffer size is less than {self._buffer_size} "
                    f"({total_samples})."
                )

            log.debug(
                "Sufficient number samples exist for a collating "
                f"samples: {total_samples} (/{self._buffer_size}). "
                "Building buffer."
            )

            out_buffer = list(self._buffer)
            incomplete_samples = list(self._buffer_next)

            # TODO: log cycle data parsing
            for cycle_data in out_buffer:
                cycle_data.current_input = cycle_data.current_meas

            for cycle_data in incomplete_samples:
                if cycle_data.cycle not in self._i_ref:
                    msg = (
                        f"Missing reference current for {cycle_data.cycle}. "
                        "Cannot build buffer."
                    )
                    raise InsufficientDataError(msg)
                cycle_data.current_input = self._i_ref[cycle_data.cycle]

        out_buffer += incomplete_samples

        log.debug(f"Collected {len(out_buffer)} cycle samples.")
        return out_buffer

    def _new_measured_I(
        self, cycle: str, cycle_timestamp: Union[int, float], value: np.ndarray
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
        log_cycle("Buffer received new measured I", cycle, cycle_timestamp)

        if cycle_timestamp not in self._cycles_next_index:
            log_cycle(
                "NEXT buffered cycle data not found.", cycle, cycle_timestamp
            )
            return

        if cycle not in self._i_ref or self._i_ref[cycle] is None:
            log_cycle("Measured current has not yet been set.", cycle)
            log_cycle("Setting new measured current for cycle.", cycle)

            cycle_data = self._cycles_next_index[cycle_timestamp]
            if cycle_data.cycle != cycle:
                if not cycle_data.cycle.endswith("_DYNECO"):
                    log_cycle(
                        "Cycle name does not match. "
                        f"Expected {cycle_data.cycle}, got {cycle}.",
                        cycle,
                        cycle_timestamp,
                    )
                    return
                else:
                    log_cycle(
                        "Cycle is DYNECO. Saving current to cycle "
                        f"{cycle_data.cycle}.",
                        cycle,
                        cycle_timestamp,
                    )

            with self._lock:
                self._i_ref[cycle_data.cycle] = value

        if cycle_timestamp not in self._cycles_next_index:
            log_cycle(
                "NEXT buffered cycle data not found. "
                f"Only {_cycle_buffer_str(self._buffer_next)} available.",
                cycle,
                cycle_timestamp,
            )
            return

        value = value.flatten()

        log_cycle(
            "Setting measured I for cycle data in NEXT buffer.",
            cycle,
            cycle_timestamp,
        )
        with self._lock:
            cycle_data = self._cycles_next_index[cycle_timestamp]
            cycle_data.current_meas = value

        self._check_move_to_buffer(cycle_data)

    def _new_measured_B(
        self, cycle: str, cycle_timestamp: Union[int, float], value: np.ndarray
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
        log_cycle("Buffer received new measured B.", cycle, cycle_timestamp)

        if cycle_timestamp not in self._cycles_next_index:
            log_cycle(
                "NEXT buffered cycle data not found.", cycle, cycle_timestamp
            )
            return

        if cycle not in self._b_meas or self._b_meas[cycle] is None:
            log_cycle("Measured magnetic field has not yet been set.", cycle)
            log_cycle("Setting new measured magnetic field.", cycle)

            cycle_data = self._cycles_next_index[cycle_timestamp]
            if cycle_data.cycle != cycle:
                if not cycle_data.cycle.endswith("_DYNECO"):
                    log_cycle(
                        "Cycle name does not match. "
                        f"Expected {cycle_data.cycle}, got {cycle}.",
                        cycle,
                        cycle_timestamp,
                    )
                    return
                else:
                    log_cycle(
                        "Cycle is DYNECO. Saving current to cycle "
                        f"{cycle_data.cycle}.",
                        cycle,
                        cycle_timestamp,
                    )

            with self._lock:
                self._b_meas[cycle_data.cycle] = value

        if cycle_timestamp not in self._cycles_next_index:
            log_cycle(
                "NEXT buffered cycle data not found. "
                f"Only {_cycle_buffer_str(self._buffer_next)} available.",
                cycle,
                cycle_timestamp,
            )
            return

        value = value.flatten() / 1e4  # G to T

        log_cycle(
            "Setting measured B for cycle data in NEXT buffer.",
            cycle,
            cycle_timestamp,
        )
        with self._lock:
            cycle_data = self._cycles_next_index[cycle_timestamp]
            cycle_data.field_meas = value

        self._check_move_to_buffer(cycle_data)

    def _new_programmed_I(
        self,
        cycle: str,
        cycle_timestamp: Union[int, float, None],
        value: np.ndarray,
    ) -> None:
        """
        Adds a new programmed current to the buffer. Only the one value is
        saved, and is used to determine if the current has changed.

        :param cycle: The LSA cycle corresponding to the data.
        :param cycle_timestamp: The timestamp of the cycle (in UTC, and ns).
        :param value: The new programmed current value.
        """
        log_cycle("Buffer received new programmed I.", cycle)
        if len(value) != 2:
            log.error(
                "Received programmed I that is not composed of "
                "time and I subarrays."
            )
            return

        value[0] *= 1e3  # s to ms

        def set_new_programmed_current() -> None:
            log_cycle("Setting new programmed current.", cycle)

            with self._lock:
                self._i_prog[cycle] = value

            log_cycle("Removing buffered reference data.", cycle)
            if cycle not in self._i_ref:
                log_cycle("No buffered reference data.", cycle)
                return

            with self._lock:
                self._i_ref.pop(cycle)

            return

        if cycle not in self._i_prog or self._i_prog[cycle] is None:
            log_cycle("Programmed current has not yet been set.", cycle)
            set_new_programmed_current()
        else:
            old_value = self._i_prog[cycle]

            if len(old_value) != len(value):
                log_cycle("Programmed current length has changed.", cycle)
                set_new_programmed_current()

            elif not np.allclose(old_value, value):
                log_cycle("Programmed current values has changed.", cycle)
                set_new_programmed_current()

            else:
                log_cycle("Programmed current has not changed.", cycle)

    def _new_programmed_B(
        self,
        cycle: str,
        cycle_timestamp: Union[int, float, None],
        value: np.ndarray,
    ) -> None:
        log_cycle("Buffer received new programmed B.", cycle)
        if len(value) != 2:
            log.error(
                "Received programmed B that is not composed of "
                "time and B subarrays."
            )
            return

        value[0] *= 1e3  # s to ms

        def set_new_programmed_field() -> None:
            log_cycle("Setting new programmed field.", cycle)
            with self._lock:
                self._b_prog[cycle] = value

            log_cycle("Removing buffered reference data.", cycle)
            if cycle not in self._b_ref:
                log_cycle("No buffered reference data.", cycle)
                return

            with self._lock:
                self._b_ref.pop(cycle)

            return

        if cycle not in self._b_prog or self._b_prog[cycle] is None:
            log_cycle("Programmed field has not yet been set.", cycle)
            set_new_programmed_field()
        else:
            old_value = self._b_prog[cycle]

            if len(old_value) != len(value):
                log_cycle("Programmed field length has changed.", cycle)
                set_new_programmed_field()

            elif not np.allclose(old_value, value):
                log_cycle("Programmed field values has changed.", cycle)
                set_new_programmed_field()

            else:
                log_cycle("Programmed field has not changed.", cycle)

    def _new_reference_B(
        self, cycle: str, cycle_timestamp: Union[int, float], value: np.ndarray
    ) -> None:
        log_cycle("Buffer received new reference B.", cycle)

        if cycle in self._b_ref:
            log_cycle("Reference field already set. Not updating it.", cycle)
            return

        log_cycle("Setting new reference field.", cycle)
        with self._lock:
            self._b_ref[cycle] = value

    def _check_buffer_integrity(self) -> None:
        """
        Check the buffered cycle eques to ensure the consecutiveness
        of the data. I.e. that newly buffered data is not older than
        existing data.
        """
        if len(self._buffer) < 2:
            log.debug(
                "Buffer integrity check not required. Only "
                f"{len(self._buffer)} buffered cycles."
            )
            return

        def check_integrity(buffer: deque[SingleCycleData]) -> list[int]:
            to_remove: list[int] = []
            with self._lock:
                buffer = buffer.copy()

            if len(buffer) < 2:
                return to_remove

            for i in range(1, len(buffer)):
                cycle_data = buffer[i]
                prev_cycle_data = buffer[i - 1]

                cycle = cycle_data.cycle
                cycle_time = cycle_data.cycle_time
                cycle_timestamp = cycle_data.cycle_timestamp

                prev_cycle = prev_cycle_data.cycle
                prev_cycle_time = prev_cycle_data.cycle_time
                prev_cycle_timestamp = prev_cycle_data.cycle_timestamp
                prev_cycle_len = prev_cycle_data.num_samples

                time_desc = abs(
                    (prev_cycle_timestamp / 1e6 + prev_cycle_len)
                    - cycle_timestamp / 1e6
                )

                if cycle_time < prev_cycle_time:
                    log.warning(
                        f"Buffered cycle {cycle}@{cycle_time} is older than "
                        f"previous cycle {prev_cycle}@{prev_cycle_time}. "
                        "Buffer integrity may be compromised. Removing "
                        "offending buffered cycle."
                    )
                    to_remove.append(i)
                elif time_desc > 5:  # 5 ms
                    msg = (
                        f"Time discrepancy {time_desc} between cycles "
                        f"{prev_cycle_data} -> {cycle_data} is greater than "
                        f"cycle length {prev_cycle_len} ms. Removing preceding"
                        " cycles."
                    )
                    log.warning(msg)

                    last_index = to_remove[-1] if len(to_remove) > 0 else 0
                    to_remove.extend(list(range(last_index, i)))

            if len(to_remove) > 0:
                log.warning(
                    "Found offending cycles at index "
                    f"{', '.join(map(str, to_remove))}."
                )
            return to_remove

        log.debug("Checking buffer integrity.")
        to_remove = check_integrity(self._buffer)
        if len(to_remove) > 0:
            for i in reversed(to_remove):
                log.info(
                    f"Removing buffered cycle {self._buffer[i]} at index {i}."
                )
                with self._lock:
                    self._cycles_index.pop(self._buffer[i].cycle_timestamp)
                    del self._buffer[i]

            to_remove = check_integrity(self._buffer)
            if len(to_remove) > 0:
                log.error(
                    "Buffer integrity compromised. Could not fix "
                    "automatically. Clearing buffer."
                )
                with self._lock:
                    self._buffer.clear()
                    self._cycles_index.clear()
        else:
            log.debug("Buffer is OK!")

        log.debug("Checking NEXT buffer integrity.")
        to_remove = check_integrity(self._buffer_next)
        if len(to_remove) > 0:
            for i in reversed(to_remove):
                log.info(
                    f"Removing buffered cycle {self._buffer_next[i].cycle}."
                )
                with self._lock:
                    self._cycles_next_index.pop(
                        self._buffer_next[i].cycle_timestamp
                    )
                    del self._buffer_next[i]

            to_remove = check_integrity(self._buffer_next)
            if len(to_remove) > 0:
                log.error(
                    "Buffer integrity compromised. Could not fix "
                    "automatically. Clearing buffer."
                )
                with self._lock:
                    self._cycles_next_index.clear()
                    self._buffer_next.clear()
        else:
            log.debug("NEXT buffer is OK!")

    def _check_buffer_size(self) -> None:
        """
        Checks the buffer size and removes the oldest data if the buffer
        is too large.

        The total buffer size is calculated as the sum of the number of
        buffered cycles and the number of buffered NEXT cycles, as
        they will be combined into a single buffer when then the
        :meth:`collate_samples` method is called.`
        """
        log.debug("Checking buffer size...")
        if len(self._buffer) == 0:
            log.debug("No buffered cycles.")
            return

        if len(self._buffer) == 1:
            log.debug(f"Insufficient buffered cycles ({len(self._buffer)}).")
            return

        if len(self._buffer_next) == 0:
            log.debug("No buffered NEXT cycles.")
            return

        def buffer_too_large(
            buffer: Iterable[SingleCycleData],
            buffer_next: Iterable[SingleCycleData],
        ) -> bool:
            num_samples_buffer = [o.num_samples for o in buffer]
            num_samples_next = [o.num_samples for o in buffer_next]

            return (
                sum(num_samples_buffer) + sum(num_samples_next)
                > self._buffer_size
            )

        def logger_msg(
            buffer: Iterable[SingleCycleData],
            buffer_next: Iterable[SingleCycleData],
        ) -> str:
            buf_size = buffer_size(buffer)
            buf_next_size = buffer_size(buffer_next)
            msg = (
                "Buffer size is too large. Removing oldest buffered cycle. "  # noqa E501
                f"[{buf_size} + {buf_next_size} = {buf_size + buf_next_size}] "  # noqa E501
                f"(/{self._buffer_size})."
            )

            return msg

        if not buffer_too_large(self._buffer, self._buffer_next):
            log.debug(logger_msg(self._buffer, self._buffer_next))
        else:
            while buffer_too_large(
                self._buffer, self._buffer_next
            ) and buffer_too_large(list(self._buffer)[1:], self._buffer_next):
                log.debug(logger_msg(self._buffer, self._buffer_next))
                with self._lock:
                    cycle_data = self._buffer.popleft()
                    log.debug(f"Removing buffered cycle {cycle_data.cycle}.")
                    self._cycles_index.pop(cycle_data.cycle_timestamp)

        self.buffer_size_changed.emit(len(self))

    def _check_move_to_buffer(self, cycle_data: SingleCycleData) -> None:
        """
        Checks if the buffered cycle data is complete, and if so moves it
        to the buffer queue.

        :param cycle_data: The buffered cycle data to check.
        """
        cycle = cycle_data.cycle
        cycle_time = cycle_data.cycle_time
        cycle_timestamp = cycle_data.cycle_timestamp

        if cycle_timestamp not in self._cycles_next_index:
            log_cycle(
                "NEXT buffered cycle data not found. "
                f"Only {_cycle_buffer_str(self._buffer_next)}.",
                cycle,
                cycle_timestamp,
            )
            return
        elif cycle_timestamp in self._cycles_index:
            log_cycle(
                "Buffered cycle data already exists.", cycle, cycle_timestamp
            )
            return

        if (
            cycle_data.current_meas is not None
            and cycle_data.field_meas is not None
        ):
            if self._buffer_next[0] is not cycle_data:
                if cycle_timestamp not in self._cycles_next_index:
                    log.error(
                        f"[{cycle}@{cycle_time}] cycle data not in NEXT "
                        f"buffer: {_cycle_buffer_str(self._buffer_next)}."
                    )
                    return

                # else
                idx = list(self._buffer_next).index(cycle_data)

                log.warning(
                    f"[{cycle}@{cycle_time}] Cycle data is not the "
                    f"next. It is at index {idx}. Preceding "
                    "entries will be removed. Existing cycles are "
                    + _cycle_buffer_str(self._buffer_next)
                )

                with self._lock:
                    for i in range(idx - 1):
                        to_remove = self._buffer_next.popleft()
                        self._cycles_next_index.pop(to_remove.cycle_timestamp)

            log_cycle(
                "Moving buffered cycle data to buffer queue.",
                cycle,
                cycle_timestamp,
            )
            with self._lock:
                self._buffer_next.popleft()
                self._cycles_next_index.pop(cycle_timestamp)

                self._buffer.append(cycle_data)
                self._cycles_index[cycle_timestamp] = cycle_data

            log_cycle(
                "Notifying new measured cycle available.",
                cycle,
                cycle_timestamp,
            )
            self.new_measured_data.emit(cycle_data)

    def __len__(self) -> int:
        return buffer_size(self._buffer) + buffer_size(self._buffer_next)


def buffer_size(buffer: Iterable[SingleCycleData]) -> int:
    return sum([o.num_samples for o in buffer])
