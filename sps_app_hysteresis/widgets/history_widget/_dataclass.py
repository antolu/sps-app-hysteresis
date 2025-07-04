from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pyqtgraph as pg
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtGui


class Plot(Enum):
    Diff = auto()
    Pred = auto()
    MeasI = auto()
    MeasB = auto()
    Delta = auto()
    RefDiff = auto()


@dataclass
class PlotItem:
    """
    A PredictionItem exists for every cycle that is acquired,
    and whether it is actually plotted or not is determined by
    the `shown` attribute.
    """

    cycle_data: CycleData

    raw_current_plt: pg.PlotCurveItem | None = None
    raw_meas_plt: pg.PlotCurveItem | None = None
    raw_pred_plt: pg.PlotCurveItem | None = None
    ref_meas_plt: pg.PlotCurveItem | None = None
    ref_pred_plt: pg.PlotCurveItem | None = None
    delta_plt: pg.PlotCurveItem | None = None

    color: QtGui.QColor | None = None
    is_shown: bool = False

    def __hash__(self) -> int:
        return hash(self.cycle_data.cycle_time)
