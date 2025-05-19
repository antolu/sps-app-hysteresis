from __future__ import annotations

import pyda
import pyda.context
import pyda_lsa
from op_app_context import context

__all__ = ["cycle_metadata"]


BEAM_IN = "SIX.MC-CTML/ControlValue"  # #controlValue
BEAM_OUT = "SX.BEAM-OUT-CTML/ControlValue"  # #controlValue
RAMP_START = "SX.ST-RAMP-CTML/ControlValue"  # #controlValue
FT_START = "SX.S-FTOP-CTML/ControlValue"  # # acqQ


class CycleMetadata:
    def __init__(self, *, provider: pyda_lsa.LsaProvider | None = None):
        self._provider = provider
        self._da: pyda.SimpleClient | None = None

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
            self._ramp_start_cache[cycle] = int(
                self._get(cycle, RAMP_START, "controlValue")
            )

        return self._ramp_start_cache[cycle]

    def flattop_start(self, cycle: str) -> int:
        if cycle not in self._flattop_start_cache:
            self._flattop_start_cache[cycle] = int(
                self._get(cycle, FT_START, "controlValue")
            )

        return self._flattop_start_cache[cycle]

    def _get(self, cycle: str, device_property: str, field: str) -> int:
        self._init_da()
        assert self._da is not None
        return self._da.get(
            endpoint=pyda_lsa.LsaEndpoint.from_str(f"{device_property}#{field}"),
            context=pyda_lsa.LsaCycleContext(cycle=cycle),
        ).data["value"]

    def _init_da(self) -> None:
        # lazy initialization of the DA client
        if self._da is None:
            self._da = pyda.SimpleClient(
                provider=self._provider or context.lsa_provider
            )


cycle_metadata = CycleMetadata()
