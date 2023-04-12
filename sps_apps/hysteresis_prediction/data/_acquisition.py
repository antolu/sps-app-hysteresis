"""
This module implements the acquisition of data, both measured
and reference, and publishes it for external use.
"""
from __future__ import annotations

import logging
from datetime import datetime
from functools import partial
from typing import Callable, Optional

from pyda import AsyncIOClient, CallbackClient, SimpleClient
from pyda.clients.asyncio import AsyncIOSubscription
from pyda.data import PropertyRetrievalResponse
from pyda_japc import JapcProvider
from pyrbac import AuthenticationClient

from ..async_utils import Signal
from ..utils import from_timestamp
from ._acquisition_buffer import AcquisitionBuffer, BufferData

DEV_LSA_B = "SPSBEAM/B"
DEV_LSA_BDOT = "SPSBEAM/BDOT"
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


log = logging.getLogger(__name__)

from_utc_ns: Callable[[int], datetime] = partial(
    from_timestamp, from_utc=True, unit="ns"
)


class Acquisition:
    def __init__(
        self,
        min_buffer_size: int,
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
                "Identified as {token.get_user_name()}."
            )
            japc_provider = JapcProvider(rbac_token=token)

        self._japc_callback = CallbackClient(provider=japc_provider)
        self._japc_simple = SimpleClient(provider=japc_provider)
        self._japc_async = AsyncIOClient(provider=japc_provider)

        self.data_acquired = Signal(
            str, PropertyRetrievalResponse
        )  # LSA cycle name, acquisition
        self.cycle_mapping_changed = Signal(str)  # LSA cycle name

    async def run(self) -> None:
        """
        Starts the acquisition process in a separate thread.
        """
        # TODO: perform synchronous GETS to initialize the buffer
        handles = self._setup_subscriptions()

        async for response in AsyncIOClient.merge_subscriptions(*handles):
            self.data_acquired.emit(response)

    def _handle_acquisition(self, response: PropertyRetrievalResponse) -> None:
        """
        Ensures that the acquisition does not contain any errors,
        and sends it through the :attr:`data_acquired` signal.
        """
        log.debug(f"Received acquisition event: {response.query.endpoint}.")
        if response.exception is not None:
            log.error(
                "An error occurred trying to access value of event: "
                f"{response.query.endpoint}@{response.value.header.selector}:"
                f"\n{str(response.exception)}"
            )
            return

        value = response.value
        log.debug(
            "Event received at "
            f"{from_utc_ns(value.header.acquisition_timestamp)} "
            "with cycle stamp "
            f"{from_utc_ns((value.header.cycle_timestamp))}."
        )

        endpoint = response.query.endpoint
        if endpoint == START_SUPERCYCLE:
            self._on_start_supercycle(response)
            return
        elif endpoint == TRIGGER_EVENT:
            self._on_forewarning(response)
            return
        elif endpoint not in ENDPOINT2BF:
            log.error(f"Received event from unknown endpoint " f"{endpoint}.")
            return
        else:
            log.error(f"Unknown endpoint {endpoint}.")

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

        self.data_acquired.emit(
            ENDPOINT2BF[endpoint], cycle, cycle_timestamp, data
        )

    def _setup_subscriptions(self) -> list[AsyncIOSubscription]:
        name = ("StartSuperCycle",)
        endpoints = (START_SUPERCYCLE,)
        selectors = ("",)
        callbacks = (self._on_start_supercycle,)

        for name, endpoint, selector, callback in zip(
            name, endpoints, selectors, callbacks
        ):
            log.debug(f"Subscribing to {endpoint} with selector {selector}.")
            handle = self._japc_async.subscribe(endpoint, context=selector)

            self._async_handles[name] = handle

        return list(self._async_handles.values())

    def _on_forewarning(self, response: PropertyRetrievalResponse) -> None:
        pass

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
        new_lsa_to_pls = {}
        new_pls_to_lsa = {}
        for user, cycle in zip(
            value.get("normalUsers"), value.get("normalLsaCycleNames")
        ):
            user = "SPS.USER." + user
            if self._pls_to_lsa.get(user) != cycle:
                mappings_changed.append(user)
            new_pls_to_lsa[user] = cycle
            new_lsa_to_pls[cycle] = user
            log.debug(f"Mapping {user} to {cycle}")

        with self._lock:
            log.debug("Updating internal mappings.")
            self._pls_to_lsa = new_pls_to_lsa
            self._lsa_to_pls = new_lsa_to_pls

        if len(mappings_changed) > 0:
            log.info(
                "LSA cycle mappings changed for users {}. "
                "Notifying listeners.".format(", ".join(mappings_changed))
            )
            for user in mappings_changed:
                self.cycle_mapping_changed.emit(new_pls_to_lsa[user])
