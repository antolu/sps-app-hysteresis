from ..._mod_replace import replace_modname
from ._cycle_model import CycleListModel
from ._plot_adapter import PlotDataAdapter, PlotType
from ._plot_model import BasePlotModel, PredictionPlotModel, UnifiedPlotModel
from ._view import HistoryPlotWidget, UnifiedHistoryPlotWidget
from ._widget import HistoryWidget

for _mod in (
    HistoryWidget,
    CycleListModel,
    PlotDataAdapter,
    PlotType,
    BasePlotModel,
    PredictionPlotModel,
    UnifiedPlotModel,
    HistoryPlotWidget,
    UnifiedHistoryPlotWidget,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod
