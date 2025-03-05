from __future__ import annotations

import datetime
import logging
from pathlib import Path

from hystcomp_utils.cycle_data import CycleData
from op_app_context import settings

log = logging.getLogger(__name__)


class IO:
    def __init__(self) -> None:
        self._enabled = False

    def enable(self) -> IO:
        if not self._enabled:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dirname = Path(".") / f"app_hysteresis_{timestamp}"
            settings["save_dir"] = str(dirname)
            log.info(f"Saving data to {dirname}")

        self._enabled = True
        return self

    def disable(self) -> IO:
        self._enabled = False
        return self

    def set_enabled(self, *, enabled: bool = True) -> IO:
        log.debug(f"Setting IO enabled to {enabled}")
        self._enabled = enabled
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save_data(self, cycle_data: CycleData) -> None:
        if not self.enabled:
            log.debug(f"IO disabled, skipping save for {cycle_data!s}")
            return

        out_dir = Path(settings["save_dir", "."])
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        user = cycle_data.user.split(".")[-1]
        cycle = cycle_data.cycle
        cycle_time = cycle_data.cycle_time.strftime("%Y%m%d_%H%M%S")
        filename = f"{cycle_time}_{user}_{cycle}.parquet"

        log.debug(f"Saving {cycle_data!s} data to {out_dir / filename}")
        cycle_data.to_pandas().to_parquet(out_dir / filename)
