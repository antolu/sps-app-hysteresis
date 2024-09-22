"""
Time utilities for handling acquired data
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Literal, Union

__all__ = ["from_timestamp"]

UNIT_TO_SCALE = {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}


class time_execution:
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

    def __enter__(self) -> time_execution:
        self.start = time.time()

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):  # type: ignore
        self.end = time.time()
        self.duration = self.end - self.start


def from_timestamp(
    timestamp: Union[float, int],
    from_utc: bool = True,
    unit: Literal["s", "ms", "us", "ns"] = "ns",
) -> datetime:
    """
    Converts a timestamp to a datetime object.

    :param timestamp: The timestamp to convert.
    :param from_utc: Whether the timestamp is in UTC. Setting to True will
        convert to the local timezone.
    :param unit: The unit of the timestamp. Can be "s", "ms", "us", or "ns".
    """
    if unit not in UNIT_TO_SCALE:
        raise ValueError(f"Invalid unit: {unit}.")

    scale = UNIT_TO_SCALE[unit]

    dt = datetime.fromtimestamp(timestamp / scale)

    if from_utc:
        dt = dt.astimezone().astimezone(timezone.utc).replace(tzinfo=None)

    return dt
