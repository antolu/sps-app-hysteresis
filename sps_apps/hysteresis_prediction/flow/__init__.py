from __future__ import annotations

from .._mod_replace import replace_modname


from ._data_flow import DataFlow, FlowWorker
from ._ucap import UcapFlowWorker, UcapDataFlow
from ._local import LocalFlowWorker, LocalDataFlow


for _mod in (DataFlow, FlowWorker, UcapFlowWorker, UcapDataFlow, LocalFlowWorker, LocalDataFlow):
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "DataFlow",
    "FlowWorker",
    "UcapFlowWorker",
    "UcapDataFlow",
    "LocalFlowWorker",
    "LocalDataFlow",
]