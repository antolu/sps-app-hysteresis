from ._model import TrimModel  # noqa: F401
from ._view import TrimWidgetView  # noqa: F401

TrimWidgetView.__module__ = __name__
TrimModel.__module__ = __name__


__all__ = ["TrimWidgetView", "TrimModel"]
