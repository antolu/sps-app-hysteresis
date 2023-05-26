from ._model import PredictionAnalysisModel  # noqa: F401
from ._view import PredictionAnalysisWidget  # noqa: F401

PredictionAnalysisWidget.__module__ = __name__
PredictionAnalysisModel.__module__ = __name__


__all__ = ["PredictionAnalysisWidget", "PredictionAnalysisModel"]
