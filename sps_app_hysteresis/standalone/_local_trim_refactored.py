from __future__ import annotations

import typing

from hystcomp_actions import Trim
from hystcomp_actions.qt import QtTrimAdapter
from op_app_context import context
from qtpy import QtCore

from ..contexts import app_context
from ..settings import StandaloneTrimSettings


class StandaloneTrimAdapter(QtTrimAdapter):
    """Extended Qt adapter that exposes additional methods for compatibility."""

    def __init__(
        self,
        trim: Trim,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(trim, parent)
        self._trim_core = trim

    def set_correction_system(self, correction_system: typing.Any) -> None:
        """Set reference to correction system for eddy current reference updates."""
        self._trim_core.set_correction_system(correction_system)


def create_standalone_trim(
    param_b_corr: str,
    settings: StandaloneTrimSettings,
    *,
    trim_threshold: float | None = None,
    parent: QtCore.QObject | None = None,
) -> QtTrimAdapter:
    """Factory function for creating Qt-wrapped Trim.

    Args:
        param_b_corr: LSA parameter for correction (e.g., "SPSBEAM/BHYS")
        settings: Trim settings instance
        trim_threshold: Minimum delta magnitude to apply trim (default from app_context)
        parent: Qt parent object

    Returns:
        Qt adapter wrapping the core Trim instance
    """
    trim_threshold = trim_threshold or app_context().TRIM_MIN_THRESHOLD

    core_trim = Trim(
        param_b_corr=param_b_corr,
        settings=settings,
        lsa_provider=context.lsa_provider,
        trim_threshold=trim_threshold,
        dry_run=False,
    )

    return StandaloneTrimAdapter(core_trim, parent=parent)
