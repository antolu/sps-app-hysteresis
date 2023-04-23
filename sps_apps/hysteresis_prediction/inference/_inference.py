from __future__ import annotations

import logging
from typing import Optional

import lightning as L
import numpy as np
from qtpy.QtCore import QObject, Signal
from sps_projects.hysteresis_compensation.data import PhyLSTMDataModule
from sps_projects.hysteresis_compensation.models import PhyLSTMModule
from sps_projects.hysteresis_compensation.utils import (
    PhyLSTMOutput,
    PhyLSTMOutputCompiled,
)

from ..data import SingleCycleData

MS = int(1e3)
NS = int(1e9)

log = logging.getLogger(__name__)


class Inference(QObject):
    load_model = Signal(str, str)  # ckpt path, device
    cycle_predicted = Signal(SingleCycleData, np.ndarray)
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
        self._module: Optional[PhyLSTMModule] = None
        self._data_module: Optional[PhyLSTMDataModule] = None

        self.load_model.connect(self.on_load_model)

    def on_load_model(self, ckpt_path: str, device: str = "cpu") -> None:
        try:
            self._load_model(ckpt_path)
            self.device = device
        except:  # noqa: broad-except
            log.exception("Failed to load model.")
            return

        self.model_loaded.emit()

    def predict_last_cycle(
        self, cycle_data: list[SingleCycleData]
    ) -> np.ndarray:
        # first check if all data has current set
        for data in cycle_data:
            if data.current_input is None:
                raise ValueError("Not all data has input current set.")

        current_input = np.concatenate(
            [data.current_input for data in cycle_data]
        )

        last_cycle = cycle_data[-1]

        predictions = self.predict(current_input, last_cycle.num_samples)
        downsampling_factor = int(last_cycle.num_samples / len(predictions))

        # uppsample predictions to match the number of samples
        time_axis = (
            np.arange(last_cycle.num_samples) / MS
            + last_cycle.cycle_timestamp / NS
        )
        predictions_upsampled = np.interp(
            time_axis, time_axis[::downsampling_factor], predictions
        )

        self.cycle_predicted.emit(last_cycle, predictions_upsampled)
        return predictions_upsampled

    def predict(
        self, input_current: np.ndarray, last_n_samples: Optional[int] = None
    ) -> np.ndarray:
        if self._module is None or self._data_module is None:
            raise RuntimeError("No model loaded.")

        dataloader = self._data_module.make_dataloader(input_current)

        predictions = self._trainer.predict(self._module, dataloader)
        if predictions is None:
            raise RuntimeError(
                "Inference failed. No predictions were returned."
            )

        if isinstance(
            predictions[0], PhyLSTMOutputCompiled
        ) and not isinstance(predictions[0], PhyLSTMOutput):
            predictions = [PhyLSTMOutput(*pred) for pred in predictions]

        predictions = PhyLSTMOutput.concatenate(*predictions)  # type: ignore

        if last_n_samples is not None:
            predictions = predictions[-last_n_samples:]

        return predictions.to_cpu().z.numpy()

    def _load_model(self, ckpt_path: str) -> None:
        self._ckpt_path = ckpt_path

        self._module = PhyLSTMModule.load_from_checkpoint(ckpt_path)
        self._data_module = PhyLSTMDataModule.load_from_checkpoint(ckpt_path)

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
