from __future__ import annotations

import asyncio
import logging
from signal import SIGINT, SIGTERM
from threading import Thread, current_thread
from typing import Optional

from sps_apps.hysteresis_prediction.data import Acquisition

log = logging.getLogger()


class Main:
    def __init__(self):
        self.acq: Optional[Acquisition] = None

    def start(self) -> None:
        """This should run in a separate thread."""
        log.info(f"Starting acquisition thread {current_thread()}.")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        self.acq = Acquisition()
        # self.acq.new_buffer_data.connect(self.buffer_handler)

        task = asyncio.ensure_future(self.acq.run(), loop=loop)

        try:
            loop.run_until_complete(task)
        finally:
            loop.close()

    @staticmethod
    def buffer_handler(buffer: list) -> None:
        log.info(f"Buffer handler called with {len(buffer)} elements.")


def setup_logging() -> None:
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    log.addHandler(ch)


def main() -> None:
    setup_logging()

    m = Main()

    t = Thread(target=m.start)
    t.start()

    try:
        t.join()
    except KeyboardInterrupt:
        log.info("Registered KeyboardInterrupt")
        m.acq.join()
        t.join()


if __name__ == "__main__":
    main()
