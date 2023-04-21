from ._model import PlotModel  # noqa: F401
from ._sources import LocalTimerTimingSource  # noqa: F401
from ._view import PlotWidget  # noqa: F401

for _mod in (PlotWidget, PlotModel, LocalTimerTimingSource):
    _mod.__module__ = __name__

__all__ = ["PlotWidget", "PlotModel", "LocalTimerTimingSource"]
