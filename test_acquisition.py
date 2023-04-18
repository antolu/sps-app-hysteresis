from __future__ import annotations

import asyncio
import logging
import threading
from signal import SIGINT, SIGTERM, signal
from threading import Event, Thread, current_thread
from typing import Optional

from sps_apps.hysteresis_prediction.async_utils import Signal
from sps_apps.hysteresis_prediction.data import Acquisition, SingleCycleData

log = logging.getLogger()


class Main:
    def __init__(self) -> None:
        self.acq: Optional[Acquisition] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.main_task: Optional[asyncio.Task] = None

        self.ready = Event()

    def start(self) -> None:
        """This should run in a separate thread."""
        log.info(f"Starting acquisition thread in {current_thread()}.")

        asyncio.run(self.run())

        Signal.cancel_all()

    async def run(self) -> None:
        self.acq = Acquisition(min_buffer_size=150000)
        self.acq.new_buffer_data.connect(self.buffer_handler)
        self.acq.new_measured_data.connect(self.new_measured_handler)

        loop = asyncio.get_running_loop()
        if loop is None:
            return
        self.main_task = loop.create_task(self.acq.run())

        self.ready.set()

        await self.main_task

    @staticmethod
    def buffer_handler(buffer: list) -> None:
        log.info(f"Buffer handler called with {len(buffer)} elements.")

    @staticmethod
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

    m = Main()

    t = Thread(target=m.start)
    t.start()

    m.ready.wait()
    signal(SIGINT, lambda x, y: m.main_task.cancel())  # noqa
    signal(SIGTERM, lambda x, y: m.main_task.cancel())  # noqa

    t.join()
    print(f"Running threads: {threading.enumerate()}")
    print("All tasks finished")


if __name__ == "__main__":
    main()
