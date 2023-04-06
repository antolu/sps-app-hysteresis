import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from __future import annotations
from pyda import CallbackClient, SimpleClient
from pyda_japc import JapcProvider

DEV_LSA_B = "SPSBEAM/B"
DEV_LSA_BDOT = "SPSBEAM/BDOT"
DEV_LSA_I = "MBI/IREF"

DEV_MEAS_I = "MBI/LOG.I#MEAS"
DEV_MEAS_B = "SR.BMEAS-SP-B-SD/CycleSamples#samples"


log = logging.getLogger(__name__)


@dataclass
class SingleCycleData:
    current_meas: np.ndarray
    field_meas: np.ndarray

    current_ref: np.ndarray
    field_ref: Optional[np.ndarray] = None

    num_samples: int = field(init=False)

    def __post_init__(self) -> None:
        if len(self.current_meas) != len(self.field_meas):
            raise ValueError(
                "Measured current and field must be of the same length."
            )

        self.num_samples = len(self.current_meas)


class Acquisition:
    def __init__(self):
        self._i_ref: dict[str, np.ndarray]
        self._b_ref: dict[str, np.ndarray]
        pass
