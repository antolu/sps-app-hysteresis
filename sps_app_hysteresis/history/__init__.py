from .._mod_replace import replace_modname
from ..widgets.history_widget._unified_model import CycleListModel
from ._history import PredictionHistory
from ._reference_cycles import ReferenceCycles

for _mod in (PredictionHistory, CycleListModel, ReferenceCycles):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
