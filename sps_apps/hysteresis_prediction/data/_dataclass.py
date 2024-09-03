from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from hystcomp_utils.cycle_data import CycleData as CycleDataBase

import numpy as np
import pandas as pd


__all__ = ["CycleData"]

log = logging.getLogger(__name__)


@dataclass
class CycleData(CycleDataBase):
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
