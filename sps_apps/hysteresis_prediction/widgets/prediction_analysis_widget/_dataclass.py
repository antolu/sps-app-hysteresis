from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pyqtgraph as pg
from hystcomp_utils.cycle_data import CycleData
from qtpy import QtGui


class DiffPlotMode(Enum):
    PredVsPred = auto()
    PredVsMeas = auto()
    PredVsRef = auto()


class MeasPlotMode(Enum):
    RawMeas = auto()
    DownsampledMeas = auto()


class Plot(Enum):
    Diff = auto()
    Pred = auto()
    MeasI = auto()
    MeasB = auto()
    Delta = auto()
    RefDiff = auto()


@dataclass
class PredictionItem:
    """
    A PredictionItem exists for every cycle that is acquired,
    and whether it is actually plotted or not is determined by
    the `shown` attribute.
    """

    cycle_data: CycleData
    diff_plot_item: pg.PlotCurveItem | None = None
    pred_plot_item: pg.PlotCurveItem | None = None
    meas_i_plot_item: pg.PlotCurveItem | None = None
    meas_b_plot_item: pg.PlotCurveItem | None = None
    delta_plot_item: pg.PlotCurveItem | None = None
    ref_diff_plot_item: pg.PlotCurveItem | None = None
    color: QtGui.QColor | None = None
    is_shown: bool = False

    def __hash__(self) -> int:
        return hash(self.cycle_data.cycle_time)
