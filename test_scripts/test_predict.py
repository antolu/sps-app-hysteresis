from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from rich.logging import RichHandler

from sps_apps.hysteresis_prediction.data import SingleCycleData
from sps_apps.hysteresis_prediction.inference import Inference

log = logging.getLogger()

torch.set_float32_matmul_precision("high")


INPUT_PATH = Path(__file__).parent / "output"
OUTPUT_PATH = Path(__file__).parent / "output_pred"
CKPT_PATH: str = "phylstm_checkpoint.ckpt"


def load_buffers(pth: Path = INPUT_PATH) -> list[list[SingleCycleData]]:
    """
    Load all pickle files in the input directory.
    """
    buffers = []

    for file in pth.glob("*.pickle"):
        with open(file, "rb") as f:
            buffers.append(pickle.load(f))

    print(f"Loaded {len(buffers)} buffers.")

    return buffers


def save_buffers(buffers: list[list[SingleCycleData]]) -> None:
    """
    Save all buffers to pickle files in the output directory.
    """
    for i, buffer in enumerate(buffers):
        with open(OUTPUT_PATH / f"buffer_{i}.pickle", "wb") as f:
            pickle.dump(buffer, f)


def setup_logging() -> None:
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    ch = RichHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    log.addHandler(ch)


def plot_predictions(
    cycle_data: SingleCycleData, predictions: torch.Tensor
) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

    ax1.plot(cycle_data.current_input, label="True", c="k")
    ax11 = ax1.twinx()
    ax11.plot(cycle_data.field_meas, label="True", c="b")
    ax11.plot(predictions, label="Predicted", c="r")

    dpp = (cycle_data.field_meas - predictions) / cycle_data.field_meas * 1e4
    ax2.plot(dpp, c="k")

    ax1.set_title("Predictions")

    plt.show()


def main() -> None:
    buffers = load_buffers()

    inference = Inference()
    inference.set_do_inference(True)
    inference._load_model(CKPT_PATH)
    inference.device = "cpu"

    predictions_ = []

    for b in buffers:
        for c in b:
            assert c.current_meas is not None
            c.current_input = c.current_meas

        predictions = inference._predict_last_cycle(b)
        print(predictions)

        b[-1].field_pred = np.array(predictions)

        predictions_.append(b)

    save_buffers(predictions_)

    return


if __name__ == "__main__":
    main()
