from __future__ import annotations

import asyncio
import logging
from signal import SIGINT, SIGTERM, signal
from threading import Event, Thread, current_thread
from typing import Optional

from sps_apps.hysteresis_prediction.async_utils import Signal
from sps_apps.hysteresis_prediction.data import Acquisition

log = logging.getLogger()


class Main:
    def __init__(self):
        self.acq: Optional[Acquisition] = None
        self.loop = None
        self.main_task = None

        self.ready = Event()

    def start(self) -> None:
        """This should run in a separate thread."""
        log.info(f"Starting acquisition thread in {current_thread()}.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop

        self.acq = Acquisition(min_buffer_size=150000)
        self.acq.new_buffer_data.connect(self.buffer_handler)

        self.main_task = loop.create_task(self.acq.run())

        self.ready.set()

        try:
            loop.run_until_complete(self.main_task)
        except asyncio.CancelledError:
            print("Task cancelled.")
        finally:
            Signal.cancel_all()

    @staticmethod
    def buffer_handler(buffer: list) -> None:
        log.info(f"Buffer handler called with {len(buffer)} elements.")


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

    import sys
    import threading

    m = Main()

    t = Thread(target=m.start)
    t.start()

    m.ready.wait()
    signal(SIGINT, lambda x, y: m.main_task.cancel())
    signal(SIGTERM, lambda x, y: m.main_task.cancel())

    t.join()
    print(f"Running threads: {threading.enumerate()}")
    print("All tasks finished")
    sys.exit(0)


if __name__ == "__main__":
    main()
