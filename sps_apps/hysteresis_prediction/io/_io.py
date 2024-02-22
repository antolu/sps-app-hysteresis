from __future__ import annotations

import logging
from pathlib import Path

from op_app_context import settings

from ..data import CycleData

log = logging.getLogger(__name__)


class IO:
    def __init__(self) -> None:
        self._enabled = False

    def enable(self) -> IO:
        self._enabled = True
        return self

    def disable(self) -> IO:
        self._enabled = False
        return self

    def set_enabled(self, enabled: bool) -> IO:
        self._enabled = enabled
        return self

    @property
    def enabled(self) -> bool:
        return self._enabled

    def save_data(self, cycle_data: CycleData) -> None:
        if not self.enabled:
            log.debug(
                "IO disabled, skipping save for {}".format(str(cycle_data))
            )
            return

        out_dir = Path(settings["save_dir", "."])
        if not out_dir.exists():
            out_dir.mkdir(parents=True)

        user = cycle_data.user.split(".")[-1]
        cycle = cycle_data.cycle
        cycle_time = cycle_data.cycle_time.strftime("%Y%m%d_%H%M%S")
        filename = "{}_{}_{}.parquet".format(cycle_time, user, cycle)

        log.debug(
            "Saving {} data to {}".format(str(cycle_data), out_dir / filename)
        )
        cycle_data.to_pandas().to_parquet(out_dir / filename)
