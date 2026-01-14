from __future__ import annotations

from hystcomp_actions import Inference
from hystcomp_actions.qt import QtInferenceAdapter
from hystcomp_utils.cycle_data import CorrectionMode
from qtpy import QtCore

from ..contexts import app_context


class StandaloneInferenceAdapter(QtInferenceAdapter):
    """Extended Qt adapter that exposes additional methods for compatibility."""

    def __init__(
        self,
        inference: Inference,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(inference, parent)
        self._inference_core = inference

    def load_eddy_current_model(self, model_name: str, model_version: str) -> None:
        """Load eddy current model."""
        self._inference_core.load_eddy_current_model(model_name, model_version)

    def load_measurement_eddy_current_model(
        self, model_name: str, model_version: str
    ) -> None:
        """Load measurement eddy current model."""
        self._inference_core.load_measurement_eddy_current_model(
            model_name, model_version
        )

    def set_prediction_mode(self, mode: CorrectionMode) -> None:
        """Set prediction mode."""
        self._inference_core.set_prediction_mode(mode)

    @property
    def model_loaded(self):  # type: ignore[no-untyped-def]
        """Get model loaded signal."""
        return self.modelLoaded

    @property
    def reset_state(self):  # type: ignore[no-untyped-def]
        """Get reset state slot."""
        return self.resetState


def create_inference(
    device: str = "cpu",
    parent: QtCore.QObject | None = None,
) -> QtInferenceAdapter:
    """Factory function for creating Qt-wrapped Inference.

    Args:
        device: Device for inference ("cpu" or "cuda")
        parent: Qt parent object

    Returns:
        Qt adapter wrapping the core Inference instance
    """
    core_inference = Inference(device=device)

    # Load eddy current model from app context
    core_inference.load_eddy_current_model(
        app_context().EDDY_CURRENT_MODEL.NAME,
        app_context().EDDY_CURRENT_MODEL.VERSION,
    )

    return StandaloneInferenceAdapter(core_inference, parent=parent)
