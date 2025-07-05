from __future__ import annotations

from .._mod_replace import replace_modname
from ._pipeline import Pipeline
from ._remote import RemotePipeline
from ._standalone import StandalonePipeline

for _mod in (
    Pipeline,
    RemotePipeline,
    StandalonePipeline,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "Pipeline",
    "RemotePipeline",
    "StandalonePipeline",
]
