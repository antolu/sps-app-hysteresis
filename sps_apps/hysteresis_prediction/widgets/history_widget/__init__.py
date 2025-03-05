from ..._mod_replace import replace_modname
from ._model import PredictionListModel
from ._widget import HistoryWidget

for _mod in (HistoryWidget, PredictionListModel):
    replace_modname(_mod, __name__)

del replace_modname
del _mod
