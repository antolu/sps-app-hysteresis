from __future__ import annotations

import logging
import time
import warnings
from threading import Lock, Thread
from typing import Optional

import lightning as L
import numpy as np
import torch
from qtpy import QtCore
from sps_projects.hysteresis_compensation.data import PhyLSTMDataModule
from sps_projects.hysteresis_compensation.models import PhyLSTM1, PhyLSTMModule
from sps_projects.hysteresis_compensation.utils import PhyLSTM1Output, ops

from ..data import SingleCycleData
from ..utils import ThreadWorker, load_cursor, run_in_thread

MS = int(1e3)
NS = int(1e9)

log = logging.getLogger(__name__)


_thread: QtCore.QThread | None = None


def inference_thread() -> QtCore.QThread:
    global _thread
    if _thread is None:
        _thread = QtCore.QThread()
    return _thread


class Inference(QtCore.QObject):
    load_model = QtCore.Signal(str, str)  # ckpt path, device
    cycle_predicted = QtCore.Signal(SingleCycleData, np.ndarray)
    model_loaded = QtCore.Signal()

    started = QtCore.Signal()
    completed = QtCore.Signal()

    def __init__(
        self, device: str = "cpu", parent: Optional[QtCore.QObject] = None
    ) -> None:
        super().__init__(parent=parent)

        # self.do_inference.connect(self._do_inference)

        self._device = device
        self._fabric = L.Fabric(accelerator=str(device))
        self._trainer = L.Trainer(accelerator=str(device))
        self._ckpt_path: Optional[str] = None
        self._module: Optional[PhyLSTMModule] = None
        self._data_module: Optional[PhyLSTMDataModule] = None
        self._model: Optional[PhyLSTM1] = None

        self._lock = Lock()
        self._do_inference = False
        self._doing_inference = False

        self.load_model.connect(self.on_load_model)

        self.started.connect(lambda: self._set_doing_inference(True))
        self.completed.connect(lambda: self._set_doing_inference(False))

    def on_load_model(self, ckpt_path: str, device: str = "cpu") -> None:
        worker = ThreadWorker(self._on_load_model, ckpt_path, device)

        worker.exception.connect(lambda: log.exception("Error loading model"))

        QtCore.QThreadPool.globalInstance().start(worker)

    def _on_load_model(self, ckpt_path: str, device: str = "cpu") -> None:
        with load_cursor():
            data_module, module = self._load_model(ckpt_path)

            with self._lock:
                self.device = device

                self._ckpt_path = ckpt_path
                self._data_module = data_module
                self._module = module
                self._model = module.model

                self.configure_model()

        self.model_loaded.emit()

    @run_in_thread(inference_thread)
    def predict_last_cycle(
        self, cycle_data: list[SingleCycleData]
    ) -> Optional[Thread]:
        if not self._do_inference:
            log.debug("Inference is disabled. Not predicting.")
            return None

        if self._doing_inference:
            log.warning(
                "Inference is already underway. " "Cannot do more in parallel."
            )
            return None

        self.started.emit()
        # first check if all data has current set
        assert self._data_module is not None
        try:
            predictions = self._predict_last_cycle(cycle_data)
            last_cycle = cycle_data[-1]

            last_cycle.field_pred = predictions
            self.cycle_predicted.emit(last_cycle, predictions)
        except:  # noqa: broad-except
            log.exception("Inference failed.")
        finally:
            self.completed.emit()

    def predict(
        self, input_current: np.ndarray, last_n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Predict the field for the given input current.
        Returns the predicted field for the last n samples.

        :param input_current: The input current to predict the field for.
        :param last_n_samples: Number of predicted samples to return. If None,
            all predictions are returned.

        :return: The predicted field.

        :raises RuntimeError: If no model is loaded.
        """
        if (
            self._module is None
            or self._data_module is None
            or self._model is None
        ):
            raise RuntimeError("No model loaded.")

        dataloader = self._data_module.make_dataloader(
            input_current, num_workers=0, pin_memory=False, predict=True
        )

        assert self._fabric is not None
        dataloader_f = self._fabric.setup_dataloaders(dataloader)

        predictions_raw: list[PhyLSTM1Output] = []
        hidden = None
        for batch in dataloader_f:
            model_output, hidden = self._model(
                batch["input"], hidden_state=hidden, return_states=True
            )
            predictions_raw.append(model_output)

        if len(predictions_raw) == 0:
            log.warning("No predictions were made.")
            return np.array([])

        predictions_detached = ops.to_cpu(ops.detach(predictions_raw))
        predictions: PhyLSTM1Output = ops.concatenate(
            *ops.squeeze(predictions_detached)
        )

        log.debug(f"Raw predictions shape: {predictions['z'].shape}")
        log.debug("Running prediction postprocessing.")
        current = torch.cat(
            [batch["input"].squeeze() for batch in dataloader], dim=0
        ).numpy()
        pred_field = predictions["z"].numpy()[..., 0]

        current, pred_field = dataloader.dataset.truncate_arrays(
            current, pred_field
        )

        if last_n_samples is not None:
            last_n_samples //= self._data_module.hparams["downsample"]
            pred_field = (
                pred_field[-last_n_samples:] if last_n_samples else pred_field
            )
            log.debug(f"Truncated predictions to {len(pred_field)} samples.")

        return np.array(pred_field)

    def _predict_last_cycle(self, cycle_data: list[SingleCycleData]):
        """
        Predict the field for the last cycle in the given data.

        :param cycle_data: The data to predict the field for.

        :return: The predicted field of the last cycle.

        :raises ValueError: If not all data has input current set.
        """
        for data in cycle_data:
            if data.current_input is None:
                raise ValueError("Not all data has input current set.")

        current_input = np.concatenate(
            [data.current_input for data in cycle_data]
        )

        last_cycle = cycle_data[-1]
        log.debug(f"Running prediction on {len(current_input)} samples.")

        assert self._data_module is not None
        start = time.time()
        log.debug("Running inference.")
        predictions = self.predict(current_input, last_cycle.num_samples)
        stop = time.time()
        log.info("Inference took: %f s", stop - start)

        time_axis = (
            np.arange(last_cycle.num_samples) / MS
            + last_cycle.cycle_timestamp / NS
        )
        time_axis = time_axis[:: self._data_module.hparams["downsample"]]

        return np.stack((time_axis, predictions), axis=0)

    def _load_model(
        self, ckpt_path: str
    ) -> tuple[PhyLSTMDataModule, PhyLSTMModule]:
        log.debug(f"Loading model and datamodule from {ckpt_path}.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = PhyLSTMModule.load_from_checkpoint(
                ckpt_path, phylstm=1, strict=False
            )
            data_module = PhyLSTMDataModule.load_from_checkpoint(ckpt_path)

        model = module.model
        module.eval()
        model.eval()
        assert model is not None
        log.debug("Compiling model.")
        # self._model = torch.compile(model)  # type: ignore

        data_module.hparams["batch_size"] = 1

        log.info("Model loaded.")

        return data_module, module

    def _get_device(self) -> str:
        return self._device

    def _set_device(self, device: str) -> None:
        if device != self._device:
            log.info(f"Switching inference device to {device}.")
            self._trainer = L.Trainer(accelerator=str(device))
            self._fabric = L.Fabric(accelerator=str(device))
        self._device = device

    device = property(_get_device, _set_device)

    @property
    def model_is_loaded(self) -> bool:
        return self._module is not None

    def configure_model(self) -> None:
        if self._model is None:
            log.error("No model loaded. Cannot configure.")
            return

        self._model = self._fabric.setup(self._model)

    def set_do_inference(self, state: bool) -> None:
        with self._lock:
            self._do_inference = state

    def _set_doing_inference(self, state: bool) -> None:
        with self._lock:
            log.debug(f"Setting doing inference to {state}.")
            self._doing_inference = state
