from .._mod_replace import replace_modname
from ._history import PredictionHistory
from ._list_model import HistoryListModel
from ._reference_cycles import ReferenceCycles

for _mod in (PredictionHistory, HistoryListModel, ReferenceCycles):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
