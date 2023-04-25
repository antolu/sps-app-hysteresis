"""
This module defines status messages.
"""
from __future__ import annotations

import logging
from enum import Enum, auto

log = logging.getLogger(__name__)


__all__ = ["AppStatus", "LOG_MESSAGES"]


class AppStatus(Enum):
    NO_MODEL = auto()
    BUFFER_WAITING = auto()
    INFERENCE_DISABLED = auto()
    INFERENCE_IDLE = auto()
    INFERENCE_RUNNING = auto()


LOG_MESSAGES = {
    AppStatus.NO_MODEL: "Load a model to start.",
    AppStatus.BUFFER_WAITING: "Waiting for buffer to fill up...",
    AppStatus.INFERENCE_DISABLED: "Predictions not enabled.",
    AppStatus.INFERENCE_IDLE: "Ready. Waiting for next cycle.",
    AppStatus.INFERENCE_RUNNING: "Predicting...",
}
