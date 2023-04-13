from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

from ..utils import from_timestamp

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

    def __post_init__(self):
        self.cycle_time = from_timestamp(
            self.cycle_timestamp, from_utc=True, unit="ms"
        )
        self.cycle_length = len(self.current_prog)
