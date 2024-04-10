from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..utils import from_timestamp

__all__ = ["CycleData"]

log = logging.getLogger(__name__)


@dataclass
class CycleData:
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

    correction: Optional[np.ndarray] = None
    """ The correction at the time the cycle is played """

    num_samples: int = field(init=False)

    def __post_init__(self) -> None:
        self.cycle_time = from_timestamp(
            self.cycle_timestamp, from_utc=False, unit="ns"
        )
        self.cycle_length = int(
            self.current_prog[0][-1]
        )  # last time marker in ms
        if str(self.cycle_length).endswith("9"):
            self.cycle_length += 1
        elif str(self.cycle_length).endswith("1"):
            self.cycle_length -= 1
        self.num_samples = self.cycle_length

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CycleData):
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

        downsample = self.field_meas.size // self.field_pred.shape[-1]
        return (
            self.field_meas[::downsample] - self.field_pred[1, :]
        ) / self.field_meas[::downsample]

    def to_dict(
        self,
    ) -> dict[str, np.ndarray | int | float | str | datetime | None]:
        return {
            "cycle": self.cycle,
            "user": self.user,
            "cycle_time": self.cycle_time,
            "cycle_timestamp": self.cycle_timestamp,
            "cycle_length": self.cycle_length,
            "current_prog": self.current_prog.flatten(),
            "field_prog": self.field_prog.flatten(),
            "current_input": (
                self.current_input if hasattr(self, "current_input") else None
            ),
            "field_ref": (
                self.field_ref.flatten()
                if self.field_ref is not None
                else None
            ),
            "field_pred": (
                self.field_pred.flatten()
                if self.field_pred is not None
                else None
            ),
            "current_meas": self.current_meas,
            "field_meas": self.field_meas,
            "num_samples": self.num_samples,
            "correction": (
                self.correction.flatten()
                if self.correction is not None
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> CycleData:  # type: ignore
        current_prog = d["current_prog"]
        field_prog = d["field_prog"]

        current_prog = from_1d_array(current_prog)
        field_prog = from_1d_array(field_prog)
        item = cls(
            d["cycle"],
            d["user"],
            d["cycle_timestamp"],
            current_prog,
            field_prog,
        )

        item.current_input = d["current_input"]
        item.field_pred = (
            from_1d_array(d["field_pred"])
            if d["field_pred"] is not None
            else None
        )
        item.field_ref = (
            from_1d_array(d["field_ref"])
            if d["field_ref"] is not None
            else None
        )
        item.current_meas = d["current_meas"]
        item.field_meas = d["field_meas"]
        item.correction = (
            from_1d_array(d["correction"])
            if d["correction"] is not None
            else None
        )

        return item

    def to_pandas(self) -> pd.DataFrame:
        """
        Export cycle data to a Pandas DataFrame.
        """
        return pd.DataFrame.from_dict(
            {k: [v] for k, v in self.to_dict().items()}
        )

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> CycleData:
        """
        Load predictions from a Pandas DataFrame.
        """
        if len(df) != 1:
            raise ValueError("DataFrame must have only one row")

        return cls.from_dict({k: v[0] for k, v in df.to_dict().items()})

    def __str__(self) -> str:
        return f"{self.cycle}@{self.cycle_time}"


def from_1d_array(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(2, arr.size // 2)
