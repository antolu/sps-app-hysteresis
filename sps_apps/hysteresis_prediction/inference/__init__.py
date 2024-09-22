from ._correction import CalculateCorrection
from ._inference import Inference  # noqa: F401

Inference.__module__ = __name__
CalculateCorrection.__module__ = __name__

__all__ = ["Inference", "CalculateCorrection"]
