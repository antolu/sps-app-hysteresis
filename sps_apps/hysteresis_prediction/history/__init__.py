from ._history import PredictionHistory
from ._list_model import HistoryListModel


from .._mod_replace import replace_modname

for _mod in (PredictionHistory, HistoryListModel):
    replace_modname(_mod, __name__)


del replace_modname
del _mod
