from ._correction import CalculateCorrection
from ._inference import Inference

Inference.__module__ = __name__
CalculateCorrection.__module__ = __name__

__all__ = ["CalculateCorrection", "Inference"]
