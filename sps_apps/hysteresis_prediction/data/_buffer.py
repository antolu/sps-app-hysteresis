"""
This module contains the AcquisitionBuffer class, which is used to
buffer and order the data acquired by the Acquisition class.
"""

from __future__ import annotations

import copy
import logging
import typing

from hystcomp_utils.cycle_data import CycleData
from hystcomp_utils.ring_buffer import CycleStampRingBuffer

from ..utils import from_timestamp

__all__ = [
    "AcquisitionBuffer",
    "InsufficientDataError",
]
log = logging.getLogger(__name__)


def log_cycle(
    msg: str,
    cycle: str,
    timestamp: float | None = None,
    log_level: int = logging.DEBUG,
    stacklevel: int = 2,
) -> None:
    if timestamp is None:
        timestamp_s = ""
    else:
        timestamp_s = "@" + str(from_timestamp(timestamp, from_utc=True, unit="ns"))

    log.log(log_level, f"[{cycle}{timestamp_s}] " + msg, stacklevel=stacklevel)


def _cycle_buffer_str(buffer: typing.Iterable[CycleData]) -> str:
    return ", ".join([cycle.cycle + "@" + str(cycle.cycle_time) for cycle in buffer])


class InsufficientDataError(Exception):
    pass


class AcquisitionBuffer:
    def __init__(
        self, min_buffer_size: int, *, buffer_only_measured: bool = False
    ) -> None:
        """
        :param min_buffer_size: The minimum number of samples required to
            collate samples from the buffer.
        """
        self._buffer_size = min_buffer_size

        self._buffer: CycleStampRingBuffer[CycleData] = CycleStampRingBuffer(
            buffer_size=int(min_buffer_size // 1000)
        )

    def new_cycle(
        self,
        cycle_data: CycleData,
    ) -> None:
        """
        Called when a new cycle is started (or is going to start).
        This will create a new :class:`CycleData` object and add
        it to the next cycles buffer with the programmed I and B,
        pending the measured I and B.

        This will also trigger an integrity check of the buffered data.

        Parameters
        ----------
        user
        """

        cycle = cycle_data.cycle
        cycle_timestamp = cycle_data.cycle_timestamp

        log_cycle("Triggered by new cycle arrival.", cycle, cycle_timestamp)
        cycle_timestamp = int(round(cycle_timestamp, -6))
        log_cycle(f"Timestamp is {cycle_timestamp}.", cycle, cycle_timestamp)

        log_cycle("Adding new cycle data to NEXT buffer.", cycle, cycle_timestamp)

        self._buffer.append(cycle_data)

        try:
            self._check_buffer_integrity()
            self._check_buffer_size()
        except KeyError:
            log.exception("Buffer check failed. Resetting buffer.")
            self.reset_buffer()

    def update_cycle(self, cycle_data: CycleData) -> None:
        """
        Called when a cycle in the buffer should be updated.

        """
        cycle = cycle_data.cycle
        cycle_timestamp = cycle_data.cycle_timestamp

        log_cycle("Triggered by cycle data update.", cycle, cycle_timestamp)
        cycle_timestamp = int(round(cycle_timestamp, -6))
        log_cycle(f"Timestamp is {cycle_timestamp}.", cycle, cycle_timestamp)

        if cycle_timestamp in self._buffer:
            log_cycle("Updating cycle data in buffer.", cycle, cycle_timestamp)
            self._buffer[cycle_timestamp] = cycle_data
            return

        msg = f"Cycle data not found in buffers: {cycle_data}"
        log.error(msg)
        raise KeyError(msg)

    def collate_samples(self) -> list[CycleData]:
        """
        Collates samples from the buffered data. This requires the minimum
        number of samples to be above the threshold, otherwise an
        :class:`InsufficientDataError` is raised.

        The buffered samples are gathered into a list in ascending order
        by cycle time, and the samples belong to consecutive cycles.

        The collated samples can then be used to perform inference, or
        train models.

        :return: A list of :class:`CycleData` objects.

        :raises InsufficientDataError: If the number of buffered samples is
            less than the minimum buffer size.
        """
        log.debug("Buffer asked to collate samples.")

        total_samples = len(self)
        if total_samples < self._buffer_size:
            msg = f"Buffer size is less than {self._buffer_size} " f"({total_samples})."
            raise InsufficientDataError(msg)

        log.debug(
            "Sufficient number samples exist for a collating "
            f"samples: {total_samples} (/{self._buffer_size}). "
            "Building buffer."
        )

        buffer = copy.deepcopy(list(self._buffer))

        log.debug(f"Collected {len(buffer)} cycle samples.")
        return buffer

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

        def check_integrity(buffer: CycleStampRingBuffer[CycleData]) -> list[int]:
            to_remove: list[int] = []
            buffer = list(buffer)

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
            log.debug(
                "Current buffer state:\nNEXT"
                + debug_msg(self._buffer)
                + "\n\nBUFFER\n"
                + debug_msg(self._buffer)
            )
            for i in reversed(to_remove):
                log.info(f"Removing buffered cycle {self._buffer[i]} at index {i}.")
                self._buffer.popleft()

            to_remove = check_integrity(self._buffer)
            if len(to_remove) > 0:
                log.debug(f"BUFFER: {debug_msg(self._buffer)}")
                log.error(
                    "Buffer integrity compromised. Could not fix "
                    "automatically. Clearing buffer."
                )
                self._buffer.clear()
        else:
            log.debug("Buffer is OK!")

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
            log.warning("Insufficient buffered cycles " f"({len(self._buffer)}).")
            return

        # if len(self._buffer_next) == 0:
        #     log.warning("No buffered NEXT cycles.")
        #     return

        def buffer_too_large(
            buffer: typing.Iterable[CycleData],
            buffer_next: typing.Iterable[CycleData],
        ) -> bool:
            num_samples_buffer = [o.num_samples for o in buffer]
            num_samples_next = [o.num_samples for o in buffer_next]

            return sum(num_samples_buffer) + sum(num_samples_next) > self._buffer_size

        def logger_msg(
            buffer: typing.Iterable[CycleData],
            buffer_next: typing.Iterable[CycleData],
        ) -> str:
            buf_size = buffer_size(buffer)
            buf_next_size = buffer_size(buffer_next)
            return (
                "Buffer size is too large. Removing oldest buffered cycle. "  # E501
                f"[{buf_size} + {buf_next_size} = {buf_size + buf_next_size}] "  # E501
                f"(/{self._buffer_size})."
            )

        if not buffer_size(self._buffer) > self._buffer_size:
            log.debug(
                "Buffer size is within bounds:"
                f"{buffer_size(self._buffer)} / {self._buffer_size}"
            )
        else:
            while (
                buffer_size(self._buffer) > self._buffer_size
                and buffer_size(list(self._buffer)[1:]) > self._buffer_size
            ):
                log.debug(
                    "Buffer size is too large. Removing oldest buffered " "cycle."
                )

                cycle_data = self._buffer.popleft()
                log.debug(f"Removing buffered cycle {cycle_data.cycle}.")

    def reset_buffer(self) -> None:
        log.debug("Resetting buffer...")

        self._buffer.clear()

    def _reset_except_last(self) -> None:
        """
        Reset buffer but keep last cycle in the NEXT buffer, i.e. the
        cycle that is currently being measured.
        """
        log.debug("Resetting buffer except last cycle...")
        if len(self._buffer) == 0:
            log.debug("Buffer is empty.")
            return

        last = self._buffer[-1]
        self.reset_buffer()
        self._buffer.append(last)

    def __len__(self) -> int:
        return buffer_size(self._buffer)


def buffer_size(buffer: typing.Iterable[CycleData]) -> int:
    return sum(o.num_samples for o in buffer)


def debug_msg(buffer: typing.Iterable[CycleData]) -> str:
    # write the same with list comprehension
    msg = [f"{cycle.cycle}@{str(cycle.cycle_time)[:-4]}" for cycle in buffer]

    return "\n".join(msg)
