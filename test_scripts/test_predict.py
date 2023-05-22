from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from signal import SIGINT, SIGTERM, signal
from threading import Thread
from traceback import print_exc

import torch
from qtpy.QtCore import QCoreApplication

from sps_apps.hysteresis_prediction.data import SingleCycleData
from sps_apps.hysteresis_prediction.inference import Inference

log = logging.getLogger()

torch.set_float32_matmul_precision("high")


INPUT_PATH = Path(__file__).parent / "output"
CKPT_PATH: str = "model_every_epoch=3000_val_loss=193.24742126464844.pt"


def load_buffers() -> list[list[SingleCycleData]]:
    """
    Load all pickle files in the input directory.
    """
    buffers = []

    for file in INPUT_PATH.glob("*.pickle"):
        with open(file, "rb") as f:
            buffers.append(pickle.load(f))

    print(f"Loaded {len(buffers)} buffers.")

    return buffers


def setup_logging() -> None:
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    log.addHandler(ch)


class ThreadWrapper:
    def __init__(
        self,
        inference: Inference,
        buffers: list[list[SingleCycleData]],
        app: QCoreApplication,
    ) -> None:
        self._inference = inference
        self._buffers = buffers
        self._app = app

    def run(self) -> None:
        time.sleep(5)  # wait for event loop to start

        try:
            for buffer in self._buffers:
                start = time.time()
                th = self._inference.predict_last_cycle(buffer)
                if th is None:
                    log.error("FATAL: Thread item is None.")
                    return
                th.join()
                stop = time.time()
                print(f"Prediction took {stop - start:.2f} seconds.")
        except:  # noqa
            print("Exception caught.")
            print_exc()
        finally:
            self._app.quit()


def main() -> None:
    application = QCoreApplication([])
    signal(SIGINT, lambda *_: application.quit())
    signal(SIGTERM, lambda *_: application.quit())

    inference = Inference()
    inference.set_do_inference(True)
    inference.on_load_model(CKPT_PATH, device="cuda")

    wrapper = ThreadWrapper(inference, load_buffers(), application)
    th = Thread(target=wrapper.run)
    th.start()

    application.exec()
    th.join()


if __name__ == "__main__":
    main()
