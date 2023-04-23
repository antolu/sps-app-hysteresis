from __future__ import annotations

import logging
from typing import Optional

import lightning as L
from qtpy.QtCore import QObject, Signal
from sps_projects.hysteresis_compensation.models import PhyLSTMModule

log = logging.getLogger(__name__)


class Inference(QObject):
    load_model = Signal(str, str)  # ckpt path, device
    model_loaded = Signal()
    do_inference = Signal()

    def __init__(
        self, device: str = "cpu", parent: Optional[QObject] = None
    ) -> None:
        super().__init__(parent=parent)

        # self.do_inference.connect(self._do_inference)

        self._device = device
        self._trainer = L.Trainer(accelerator=str(device))
        self._ckpt_path: Optional[str] = None
        self._module: Optional[L.LightningModule] = None

        self.load_model.connect(self.on_load_model)

    def on_load_model(self, ckpt_path: str, device: str = "cpu") -> None:
        try:
            self._load_model(ckpt_path)
            self.device = device
        except:  # noqa: broad-except
            log.exception("Failed to load model.")
            return

        self.model_loaded.emit()

    def _load_model(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

        self._module = PhyLSTMModule.load_from_checkpoint(ckpt_path)

        dataloader = ...

        self._trainer.predict(self._module, dataloader)

    def _get_device(self) -> str:
        return self._device

    def _set_device(self, device: str) -> None:
        self._device = device
        if device != self._device:
            log.info(f"Switching inference device to {device}.")
            self._trainer = L.Trainer(accelerator=str(device))

    device = property(_get_device, _set_device)

    @property
    def model_is_loaded(self) -> bool:
        return self._module is not None
