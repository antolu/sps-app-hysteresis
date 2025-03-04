from ._widget import HistoryWidget
from ..._mod_replace import replace_modname

for _mod in (HistoryWidget,):
    replace_modname(_mod, __name__)

del replace_modname
del _mod
