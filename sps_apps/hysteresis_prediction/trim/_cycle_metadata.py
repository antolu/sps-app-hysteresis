from __future__ import annotations

import pyda
import pyda.context
import pyda_japc
from op_app_context import context

__all__ = ["cycle_metadata"]


BEAM_IN = "SIX.MC-CTML/ControlValue"  # #controlValue
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue"  # #controlValue
RAMP_START = "XTIM.SX.S-RAMP-CT/Acquisition"  # #acqC
FT_START = "XTIM.SX.SFLATTOP-CT/Acquisition"  # # acqQ


class CycleMetadata:
    def __init__(self, *, provider: pyda_japc.JapcProvider | None = None):
        self._provider = provider
        self._da = pyda.SimpleClient(provider=provider or context.japc_provider)

        self._beam_in_cache: dict[str, int] = {}
        self._beam_out_cache: dict[str, int] = {}
        self._ramp_start_cache: dict[str, int] = {}
        self._flattop_start_cache: dict[str, int] = {}

    def beam_in(self, cycle: str) -> int:
        if cycle not in self._beam_in_cache:
            self._beam_in_cache[cycle] = int(self._get(cycle, BEAM_IN, "controlValue"))

        return self._beam_in_cache[cycle]

    def beam_out(self, cycle: str) -> int:
        if cycle not in self._beam_out_cache:
            self._beam_out_cache[cycle] = int(
                self._get(cycle, BEAM_OUT, "controlValue")
            )

        return self._beam_out_cache[cycle]

    def ramp_start(self, cycle: str) -> int:
        if cycle not in self._ramp_start_cache:
            self._ramp_start_cache[cycle] = int(self._get(cycle, RAMP_START, "acqC"))

        return self._ramp_start_cache[cycle]

    def flattop_start(self, cycle: str) -> int:
        if cycle not in self._flattop_start_cache:
            self._flattop_start_cache[cycle] = int(self._get(cycle, FT_START, "acqQ"))

        return self._flattop_start_cache[cycle]

    def _get(self, cycle: str, device_property: str, field: str) -> int:
        ctxt = pyda.context.TimingSelector(f"SPS.CYCLE.{cycle}")
        return self._da.get(endpoint=device_property, context=ctxt).data[field]


cycle_metadata = CycleMetadata()
