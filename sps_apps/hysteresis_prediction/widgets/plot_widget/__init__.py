from ._model import PlotModel
from ._sources import LocalTimerTimingSource
from ._view import PlotWidget

for _mod in (PlotWidget, PlotModel, LocalTimerTimingSource):
    _mod.__module__ = __name__

__all__ = ["LocalTimerTimingSource", "PlotModel", "PlotWidget"]
