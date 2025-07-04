from __future__ import annotations

from .._mod_replace import replace_modname
from ._pipeline import Pipeline, PipelineWorker
from ._remote import RemotePipeline, RemotePipelineWorker
from ._standalone import StandalonePipeline, StandalonePipelineWorker

for _mod in (
    Pipeline,
    PipelineWorker,
    RemotePipelineWorker,
    RemotePipeline,
    StandalonePipelineWorker,
    StandalonePipeline,
):
    replace_modname(_mod, __name__)

del replace_modname
del _mod


__all__ = [
    "Pipeline",
    "PipelineWorker",
    "RemotePipeline",
    "RemotePipelineWorker",
    "StandalonePipeline",
    "StandalonePipelineWorker",
]
