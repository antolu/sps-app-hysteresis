from .._mod_replace import replace_modname
from ._correction import CalculateCorrection
from ._correction_refactored import create_correction
from ._inference import Inference
from ._inference_refactored import create_inference
from ._local_trim import StandaloneTrim
from ._local_trim_refactored import create_standalone_trim

for _mod in (StandaloneTrim, CalculateCorrection, Inference):
    replace_modname(_mod, __name__)

__all__ = [
    "CalculateCorrection",
    "Inference",
    "StandaloneTrim",
    "create_correction",
    "create_inference",
    "create_standalone_trim",
]
