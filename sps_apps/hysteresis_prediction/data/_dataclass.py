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
    cycle: str
    """LSA cycle name"""

    user: str
    """ Timing user """

    cycle_time: datetime = field(init=False)
    """The cycle timestamp converted to datetime in localtime. """

    cycle_timestamp: float
    """ The cycle timestamp in UTC, and ns. """

    cycle_length: int = field(init=False)
    """ Cycle length, in ms. This corresponds to number of samples. """

    current_prog: np.ndarray
    field_prog: np.ndarray
    """ Programmed current and field """

    current_input: np.ndarray = field(init=False)  # input current to NN

    field_ref: Optional[np.ndarray] = None  # reference field
    """ The reference data to compare against, set externally """

    field_pred: Optional[np.ndarray] = None
    """ The predicted field """

    current_meas: Optional[np.ndarray] = None
    field_meas: Optional[np.ndarray] = None
    """ The data for these fields arrives after cycle is played """

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

    @property
    def dp_p(self) -> np.ndarray:
        if self.field_meas is None:
            raise ValueError("Reference field is not set")
        if self.field_pred is None:
            raise ValueError("Predicted field is not set")

        return (self.field_meas - self.field_pred) / self.field_meas

    def __str__(self) -> str:
        return f"{self.cycle}@{self.cycle_time}"
