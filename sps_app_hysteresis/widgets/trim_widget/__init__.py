"""
The TrimWidget is a widget that enables trimming for the feedforward field corrections.

The widget is separated into the TrimModel and TrimWidgetView.
The TrimModel is responsible for the logic and the TrimWidgetView is responsible for the GUI.
"""

from ._model import TrimModel
from ._view import TrimWidgetView

TrimWidgetView.__module__ = __name__
TrimModel.__module__ = __name__


__all__ = ["TrimModel", "TrimWidgetView"]
