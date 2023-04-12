from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from dateteime import datetime

log = logging.getLogger(__name__)


@dataclass
class SingleCycleData:
    """The cycle timestamp converted to datetime."""

    cycle_time: datetime

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

    def is_complete(self) -> bool:
        return self.current_meas is not None and self.field_meas is not None
