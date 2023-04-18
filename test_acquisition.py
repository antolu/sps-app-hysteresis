from __future__ import annotations

import logging
import threading
from signal import SIGINT, SIGTERM, signal

from sps_apps.hysteresis_prediction.data import Acquisition, SingleCycleData

log = logging.getLogger()


def buffer_handler(buffer: list) -> None:
    log.info(f"Buffer handler called with {len(buffer)} elements.")


def new_measured_handler(cycle_data: SingleCycleData) -> None:
    log.info(
        f"Received new measured data {cycle_data.cycle}"
        "@{cycle_data.cycle_time}."
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
    setup_logging()

    acq = Acquisition(min_buffer_size=150000)
    acq.new_buffer_data.connect(buffer_handler)
    acq.new_measured_data.connect(new_measured_handler)

    th = acq.run()

    signal(SIGINT, lambda x, y: acq.stop())  # noqa
    signal(SIGTERM, lambda x, y: acq.stop())  # noqa

    th.join()
    print(f"Running threads: {threading.enumerate()}")
    print("All tasks finished")


if __name__ == "__main__":
    main()
