from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np

from ..utils import from_timestamp

__all__ = ["SingleCycleData"]

log = logging.getLogger(__name__)


@dataclass
class SingleCycleData:
    """LSA cycle name"""

    cycle: str

    """The cycle timestamp converted to datetime in localtime. """
    cycle_time: datetime = field(init=False)

    """ The cycle timestamp in UTC, and ns. """
    cycle_timestamp: int

    """ Cycle length, in ms. This corresponds to number of samples. """
    cycle_length: int = field(init=False)

    """ Programmed current and field """
    current_prog: np.ndarray
    field_prog: np.ndarray

    current_input: np.ndarray = field(init=False)  # input current to NN

    """ The reference data to compare against, set externally """
    field_ref: np.ndarray = field(init=False)  # reference field

    """ The data for these fields arrives after cycle is played """
    current_meas: Optional[np.ndarray] = None
    field_meas: Optional[np.ndarray] = None

    num_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.cycle_time = from_timestamp(
            self.cycle_timestamp, from_utc=True, unit="ns"
        )
        self.cycle_length = int(
            self.current_prog[0][-1]
        )  # last time marker in ms
        self.num_samples = self.cycle_length

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, SingleCycleData):
            return False

        return (
            self.cycle is other.cycle
            and self.cycle_timestamp is other.cycle_timestamp
            and self.current_prog is other.current_prog
            and self.field_prog is other.field_prog
        )

    def __str__(self) -> str:
        return f"{self.cycle}@{self.cycle_time}"
