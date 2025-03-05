"""
Time utilities for handling acquired data
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Literal, Self

__all__ = ["from_timestamp"]

UNIT_TO_SCALE = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}


class time_execution:  # noqa: N801
    """
    Convenience class for timing execution. Used simply as
    >>> with time_execution() as t:
    >>>     # some code to time
    >>> print(t.duration)
    """

    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0
        self.duration = 0.0

    def __enter__(self) -> Self:
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):  # type: ignore
        self.end = time.time()
        self.duration = self.end - self.start


def from_timestamp(
    timestamp: float,
    unit: Literal["s", "ms", "us", "ns"] = "ns",
    *,
    from_utc: bool = True,
) -> datetime:
    """
    Converts a timestamp to a datetime object.

    :param timestamp: The timestamp to convert.
    :param from_utc: Whether the timestamp is in UTC. Setting to True will
        convert to the local timezone.
    :param unit: The unit of the timestamp. Can be "s", "ms", "us", or "ns".
    """
    if unit not in UNIT_TO_SCALE:
        msg = f"Invalid unit: {unit}."
        raise ValueError(msg)

    scale = UNIT_TO_SCALE[unit]

    dt = datetime.fromtimestamp(timestamp / scale)

    if from_utc:
        dt = dt.astimezone().astimezone(UTC).replace(tzinfo=None)

    return dt
