from ._inference import Inference  # noqa: F401
from ._correction import CalculateCorrection

Inference.__module__ = __name__
CalculateCorrection.__module__ = __name__

__all__ = ["Inference", "CalculateCorrection"]
