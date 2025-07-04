from .._mod_replace import replace_modname
from ._correction import CalculateCorrection
from ._inference import Inference
from ._local_trim import StandaloneTrim

for _mod in (StandaloneTrim, CalculateCorrection, Inference):
    replace_modname(_mod, __name__)

__all__ = ["CalculateCorrection", "Inference", "StandaloneTrim"]
