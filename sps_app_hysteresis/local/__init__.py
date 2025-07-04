from .._mod_replace import replace_modname
from ._correction import CalculateCorrection
from ._inference import Inference
from ._local_trim import LocalTrim

for _mod in (LocalTrim, CalculateCorrection, Inference):
    replace_modname(_mod, __name__)

__all__ = ["CalculateCorrection", "Inference", "LocalTrim"]
