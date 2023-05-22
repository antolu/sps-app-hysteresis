from __future__ import annotations

import logging
import pickle
from argparse import ArgumentParser
from pathlib import Path
from signal import SIGINT, SIGTERM, signal

from sps_apps.hysteresis_prediction.data import Acquisition, SingleCycleData

log = logging.getLogger()


OUTPUT_PATH = Path(__file__).parent / "output"
save = False


def buffer_handler(buffer: list[SingleCycleData], save: bool = False) -> None:
    msg = "\n".join(
        [f"{cycle} [-> {cycle.num_samples} ms]" for cycle in buffer]
    )
    log.info(f"Buffer handler called with {len(buffer)} elements.\n" + msg)

    if save:
        fmt = "%Y%m%d-%H%M%S"
        filename = (
            f"{buffer[0].cycle_time.strftime(fmt)}-"
            f"--{buffer[-1].cycle_time.strftime(fmt)}.pickle"
        )
        with open(OUTPUT_PATH / filename, "wb") as f:
            pickle.dump(buffer, f)


def new_measured_handler(cycle_data: SingleCycleData) -> None:
    log.info(
        f"Received new measured data {cycle_data.cycle}"
        f"@{cycle_data.cycle_time}."
    )


def setup_logging() -> None:
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    log.addHandler(ch)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "-s", "--save", action="store_true", help="Save buffers to file."
    )
    parser.add_argument(
        "-m",
        "--measured",
        action="store_true",
        help="Buffer only measured data.",
    )
    args = parser.parse_args()
    global save
    save = args.save

    setup_logging()
    OUTPUT_PATH.mkdir(exist_ok=True)

    acq = Acquisition(
        min_buffer_size=150000, buffer_only_measured=args.measured
    )
    acq.new_buffer_data.connect(lambda x: buffer_handler(x, save=True))
    acq.new_measured_data.connect(new_measured_handler)

    th = acq.run()

    signal(SIGINT, lambda x, y: acq.stop())  # noqa
    signal(SIGTERM, lambda x, y: acq.stop())  # noqa

    th.join()


if __name__ == "__main__":
    main()
