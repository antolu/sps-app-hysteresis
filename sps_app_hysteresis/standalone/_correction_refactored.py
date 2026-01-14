from __future__ import annotations

from hystcomp_actions import Correction
from hystcomp_actions.qt import QtCorrectionAdapter
from hystcomp_utils.cycle_data import CorrectionMode
from qtpy import QtCore

from ..settings import StandaloneTrimSettings


class StandaloneCorrectionAdapter(QtCorrectionAdapter):
    """Extended Qt adapter that exposes additional methods for compatibility."""

    def __init__(
        self,
        correction: Correction,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(correction, parent)
        self._correction_core = correction

    def set_prediction_mode(self, mode: CorrectionMode) -> None:
        """Set prediction mode."""
        self._correction_core.set_prediction_mode(mode)

    @property
    def newReference(self):  # type: ignore[no-untyped-def]
        """Get new reference signal."""
        return super().newReference


def create_correction(
    trim_settings: StandaloneTrimSettings,
    parent: QtCore.QObject | None = None,
) -> tuple[StandaloneCorrectionAdapter, Correction]:
    """Factory function for creating Qt-wrapped Correction.

    Args:
        trim_settings: Trim settings instance
        parent: Qt parent object

    Returns:
        Tuple of (adapter, core) - adapter for Qt signals, core for direct method access
    """
    core_correction = Correction(trim_settings=trim_settings)
    adapter = StandaloneCorrectionAdapter(core_correction, parent=parent)

    return adapter, core_correction
