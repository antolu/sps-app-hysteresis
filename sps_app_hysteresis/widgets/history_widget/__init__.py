from ..._mod_replace import replace_modname
from ._model import PredictionListModel
from ._plot_adapter import PlotDataAdapter, PlotType
from ._plot_model import BasePlotModel, PredictionPlotModel, UnifiedPlotModel
from ._unified_model import CycleListModel
from ._view import UnifiedHistoryPlotWidget
from ._widget import HistoryWidget

for _mod in (
    HistoryWidget,
    PredictionListModel,
    CycleListModel,
    PlotDataAdapter,
    PlotType,
    BasePlotModel,
    PredictionPlotModel,
    UnifiedPlotModel,
    UnifiedHistoryPlotWidget,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod
