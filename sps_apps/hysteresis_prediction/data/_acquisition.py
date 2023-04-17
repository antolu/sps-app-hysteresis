"""
This module implements the acquisition of data, both measured
and reference, and publishes it for external use.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import partial
from typing import Callable, Iterable, Optional

from pyda import AsyncIOClient, SimpleClient
from pyda.clients.asyncio import AsyncIOSubscription
from pyda.data import PropertyRetrievalResponse, StandardEndpoint
from pyda_japc import JapcProvider
from pyrbac import AuthenticationClient

from ..async_utils import Signal
from ..utils import from_timestamp
from ._acquisition_buffer import (
    AcquisitionBuffer,
    BufferData,
    InsufficientDataError,
)
from ._dataclass import SingleCycleData

__all__ = ["Acquisition"]

# DEV_LSA_B = "rmi://virtual_sps/SPSBEAM/B"
DEV_LSA_B = "MBI/LOG.I.REF#values"  # placeholder until we can get B
DEV_LSA_BDOT = "rm://virtual_sps/SPSBEAM/BDOT"
DEV_LSA_I = "MBI/LOG.I.REF"

DEV_MEAS_I = "MBI/LOG.I.MEAS"
DEV_MEAS_B = "SR.BMEAS-SP-B-SD/CycleSamples#samples"

TRIGGER_EVENT = "XTIM.SX.FCY2500-CT/Acquisition"
START_SUPERCYCLE = "SX.CZERO-CTML/SuperCycle"


ENDPOINT2BF = {
    DEV_MEAS_I: BufferData.MEAS_I,
    DEV_MEAS_B: BufferData.MEAS_B,
    DEV_LSA_I: BufferData.PROG_I,
    DEV_LSA_B: BufferData.PROG_B,
}

ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)  # noqa: E501


log = logging.getLogger(__name__)

from_utc_ns: Callable[[int], datetime] = partial(
    from_timestamp, from_utc=True, unit="ns"
)


class Acquisition:
    def __init__(
        self,
        min_buffer_size: int = 300000,
        japc_provider: Optional[JapcProvider] = None,
    ):
        """
        This class handles the acquisition of data from the online magnets
        after the cycle has been played, and sends a notification when
        sufficient data has been acquired to perform inference.

        The class also keeps track of mapping of timing users to LSA cycles,
        and resets internal buffers automatically when required.

        :param min_buffer_size: Minimum number of samples required to before
            notification is sent. The buffer only records number of cycles
            that are sufficient to reach this number, the rest are discarded.
            The oldest data is discarded first.
        :param japc_provider: Optional JAPC provider to use for subscribing to
            data. The provider must have an RBAC role for reading measured
            and reference currents.
        """
        self._min_buffer_size = min_buffer_size
        self._buffer = AcquisitionBuffer(min_buffer_size)

        self._pls_to_lsa: dict[str, str] = {}
        self._lsa_to_pls: dict[str, str] = {}

        self._async_handles: dict[str, AsyncIOSubscription] = {}

        if japc_provider is None:
            rbac_client = AuthenticationClient.create()
            log.info(
                "JapcProvider not provided, logging into RBAC by location."
            )
            token = rbac_client.login_location()
            log.info(
                "RBAC login successful. "
                f"Identified as {token.get_user_name()}."
            )
            japc_provider = JapcProvider(rbac_token=token)

        self._stop_execution = False

        self._japc_simple = SimpleClient(provider=japc_provider)
        self._japc_async = AsyncIOClient(provider=japc_provider)

        self.new_buffer_data = Signal(list[SingleCycleData])
        self.new_measured_data = Signal(SingleCycleData)
        self.cycle_mapping_changed = Signal(str)  # LSA cycle name

    async def run(self) -> None:
        """
        Starts the acquisition process in a separate thread.
        """
        # TODO: perform synchronous GETS to initialize the buffer
        try:
            handles = self._setup_subscriptions()
        except:  # noqa
            log.exception("Error while setting up subscriptions.")
            return

        async for response in AsyncIOClient.merge_subscriptions(*handles):
            if self._stop_execution:
                break

            try:
                self._handle_acquisition(response)
            except Exception as e:
                log.exception("Error handling acquisition event.", e)

    @property
    def buffer(self) -> AcquisitionBuffer:
        return self._buffer

    def _setup_subscriptions(self) -> list[AsyncIOSubscription]:
        subscriptions = [
            ("StartSuperCycle", START_SUPERCYCLE, ""),
            ("ProgrammedCurrent", DEV_LSA_I, "SPS.USER.ALL"),
            ("MeasuredCurrent", DEV_MEAS_I, "SPS.USER.ALL"),
            ("MeasuredBField", DEV_MEAS_B, "SPS.USER.ALL"),
            ("Forewarning", TRIGGER_EVENT, "SPS.USER.ALL"),
            ("ProgrammedBField", DEV_LSA_B, "SPS.USER.ALL"),
        ]

        log.debug("Setting up subscriptions.")
        log.debug("Performing GETs to initialize the buffer.")

        self._handle_acquisition(
            self._japc_simple.get(START_SUPERCYCLE, context="")
        )

        def make_endpoint(endpoint: str) -> StandardEndpoint:
            m = ENDPOINT_RE.match(endpoint)

            if not m:
                raise ValueError(f"Not a valid endpoint: {endpoint}.")

            return StandardEndpoint(
                device_name=m.group("device"),
                property_name=m.group("property"),
            )

        for _, endpoint, _ in subscriptions[1:]:
            for selector in self._pls_to_lsa.keys():
                log.debug(f"GET-ting values for {endpoint}@{selector}.")
                self._handle_acquisition(
                    self._japc_simple.get(
                        make_endpoint(endpoint), context=selector
                    )
                )

        log.debug("Subscribing to events.")
        for name, endpoint, selector in subscriptions:
            log.debug(f"Subscribing to {endpoint} with selector {selector}.")
            handle = self._japc_async.subscribe(
                make_endpoint(endpoint), context=selector
            )

            self._async_handles[name] = handle

        return list(self._async_handles.values())

    def _handle_acquisition(self, response: PropertyRetrievalResponse) -> None:
        """
        Ensures that the acquisition does not contain any errors,
        and sends it through the :attr:`data_acquired` signal.
        """
        log.debug(f"Received acquisition event: {response.query.endpoint}.")
        if response.exception is not None:
            log.error(
                "An error occurred trying to access value of event: "
                f"{response.query.endpoint}:"
                f"\n{str(response.exception)}"
            )
            return

        if response.notification_type == "FIRST_UPDATE":
            msg = (
                "Received first update for "
                f"{response.query.endpoint}@{response.value.header.selector}. "
            )
            if str(response.query.endpoint) != DEV_LSA_I:
                msg += "Discarding it."
                log.debug(msg)
                return
            else:
                log.debug(msg)

        value = response.value
        cycle_timestamp = value.header.cycle_timestamp
        cycle_time = (
            from_utc_ns(cycle_timestamp)
            if cycle_timestamp is not None
            else "N/A"
        )
        log.debug(
            "Event received at "
            f"{from_utc_ns(value.header.acquisition_timestamp)} "
            "with cycle stamp "
            f"{cycle_time}."
        )

        endpoint = str(response.query.endpoint)
        if endpoint == START_SUPERCYCLE:
            self._on_start_supercycle(response)
            return
        elif endpoint == TRIGGER_EVENT:
            self._on_forewarning(response)
            return
        elif endpoint in ENDPOINT2BF:
            pass
        else:
            log.error(f"Received event from unknown endpoint " f"{endpoint}.")
            return

        cycle = self._pls_to_lsa.get(value.header.selector)
        if cycle is None:
            log.error(
                "Received event from unknown timing user "
                f"{value.header.selector}."
            )
            return

        data = response.value.get("value")
        if data is None:
            log.error(f"Received event with no data from {endpoint}.")
            return

        cycle_timestamp = value.header.cycle_timestamp

        self._buffer.dispatch_data(
            ENDPOINT2BF[endpoint], cycle, cycle_timestamp, data
        )

    def _on_forewarning(self, response: PropertyRetrievalResponse) -> None:
        """
        Handler for the forewarning event. This is triggered 2500ms before
        the cycle plays.

        This function notify the buffer that a new cycle is about to start,
        and then retrieve the latest buffered data if it is available.
        """
        cycle = response.value.get("lsaCycleName")
        cycle_timestamp = response.value.header.cycle_timestamp
        log.debug(
            f"Cycle {cycle} at {from_utc_ns(cycle_timestamp)} is about to "
            "start. Notifying buffer."
        )

        self.buffer.new_cycle(cycle, cycle_timestamp)

        log.debug("Query buffer for latest buffered data.")

        try:
            buffer = self._buffer.collate_samples()
        except InsufficientDataError as e:
            log.debug(str(e))
            return

        log.debug("Buffered data available. Sending it to listeners.")

        self.new_buffer_data.emit(buffer)

    def _on_start_supercycle(
        self, response: PropertyRetrievalResponse
    ) -> None:
        """
        Callback function for the supercycle start event.
        Maps timing/PLS users to LSA cycle names, and sends a notification
        if the mapped user changes

        :param response: PropertyRetrievalResponse
            The response container on retrieval.
        """
        value = response.value
        dt = from_timestamp(
            value.header.acquisition_timestamp, from_utc=True, unit="ns"
        )
        log.debug(f"Start supercycle triggered at {dt}.")

        mappings_changed = []

        def parse_mappings(
            users: Iterable[str], cycles: Iterable[str]
        ) -> tuple[dict[str, str], dict[str, str]]:
            lsa_to_pls = {}
            pls_to_lsa = {}

            for user, cycle in zip(users, cycles):
                user = "SPS.USER." + user

                if self._pls_to_lsa.get(user) != cycle:
                    mappings_changed.append(user)

                pls_to_lsa[user] = cycle
                lsa_to_pls[cycle] = user

                log.debug(f"Mapping {user} to {cycle}")

            return lsa_to_pls, pls_to_lsa

        normal_lsa_to_pls, normal_pls_to_lsa = parse_mappings(
            value.get("normalUsers"), value.get("normalLsaCycleNames")
        )

        spare_lsa_to_pls, spare_pls_to_lsa = parse_mappings(
            value.get("spareUsers"), value.get("spareLsaCycleNames")
        )

        def log_line(text: str):
            return "# " + text + (" " * (79 - len(text) - 4)) + " #"

        log_lines = [
            log_line(msg)
            for msg in (
                "=" * 75,
                "",
                25 * " " + "START SUPERCYCLE",
                "",
                25 * "" + "Normal users",
                *[
                    "* " + user + " : " + cycle
                    for user, cycle in normal_pls_to_lsa.items()
                ],
                "",
                25 * "" + "Spare users",
                *[
                    "* " + user + " : " + cycle
                    for user, cycle in spare_pls_to_lsa.items()
                ],
                "",
                "=" * 75,
            )
        ]

        log.debug("Updating internal mappings.\n" + "\n".join(log_lines))
        new_lsa_to_pls = {**normal_lsa_to_pls, **spare_pls_to_lsa}
        new_pls_to_lsa = {**normal_pls_to_lsa, **spare_pls_to_lsa}
        self._pls_to_lsa.update(new_pls_to_lsa)
        self._lsa_to_pls.update(new_lsa_to_pls)

        if len(mappings_changed) > 0:
            log.info(
                "LSA cycle mappings changed for users {}. "
                "Notifying listeners.".format(", ".join(mappings_changed))
            )
            for user in mappings_changed:
                self.cycle_mapping_changed.emit(new_pls_to_lsa[user])
