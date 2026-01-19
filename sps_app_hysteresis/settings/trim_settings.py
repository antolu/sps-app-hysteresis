from __future__ import annotations

import logging
import typing

import pyda
from hystcomp_actions import (
    OnlineTrimSettings as HystcompOnlineTrimSettings,
)
from hystcomp_actions import QtTrimSettings as HystcompQtTrimSettings
from hystcomp_actions import TrimSettings as HystcompTrimSettings
from hystcomp_actions.utils import cycle_metadata
from op_app_context import context
from op_app_context import settings as app_settings
from qtpy import QtCore

log = logging.getLogger(__package__)


class TrimSettings(HystcompTrimSettings, QtCore.QObject):
    trimEnabledChanged = QtCore.Signal(str, bool)
    """ Emitted when the trim is enabled or disabled for a cycle """

    trimDryRunChanged = QtCore.Signal(str, bool)
    """ Emitted when the trim is a dry run or not for a cycle """

    trimStartChanged = QtCore.Signal(str, float)
    """ Emitted when the trim start time is changed for a cycle """

    trimEndChanged = QtCore.Signal(str, float)
    """ Emitted when the trim end time is changed for a cycle """

    trimGainChanged = QtCore.Signal(str, float)
    """ Emitted when the trim gain is changed for a cycle """

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)

    @property
    def trim_enabled(self) -> typing.MutableMapping[str, bool]:
        """Whether the trim is enabled or not."""
        raise NotImplementedError

    @property
    def trim_start(self) -> typing.MutableMapping[str, float]:
        """CTime of the start of the trim. Normally should not exceed beam injection time."""
        raise NotImplementedError

    @property
    def trim_end(self) -> typing.MutableMapping[str, float]:
        """CTime of the end of the trim. Normally should not exceed beam injection time."""
        raise NotImplementedError

    @property
    def gain(self) -> typing.MutableMapping[str, float]:
        """Gain of the trim."""
        raise NotImplementedError


class QtTrimSettings(HystcompQtTrimSettings):
    """Wrapper around hystcomp_actions.QtTrimSettings with local defaults."""

    def __init__(self, parent: QtCore.QObject | None = None, *, prefix: str):
        super().__init__(
            parent=parent,
            prefix=prefix,
            settings=app_settings.settings,
            cycle_metadata=cycle_metadata,
        )


class OnlineTrimSettings(HystcompOnlineTrimSettings):
    """Wrapper around hystcomp_actions.OnlineTrimSettings with local defaults."""

    def __init__(
        self,
        parent: QtCore.QObject | None = None,
        *,
        device: str,
    ):
        super().__init__(
            parent=parent,
            device=device,
            da=pyda.SimpleClient(provider=context.japc_provider),
        )
