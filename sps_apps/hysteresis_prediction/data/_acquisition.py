"""
This module implements the acquisition of data, both measured
and reference, and publishes it for external use.
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime
from functools import partial
from signal import SIGINT, SIGTERM, signal
from threading import Thread
from typing import Callable, Iterable, Optional, Union

from pyda import AsyncIOClient, SimpleClient
from pyda.clients.asyncio import AsyncIOSubscription
from pyda.data import PropertyRetrievalResponse

try:
    from pyda_japc import JapcProvider
    from pyjapc import PyJapc
    from pyrbac import AuthenticationClient  # noqa: import-error
except ImportError:
    # for my macos laptop
    AuthenticationClient = None  # type: ignore

from ..async_utils import Signal
from ..utils import from_timestamp
from ._acquisition_buffer import (
    AcquisitionBuffer,
    BufferData,
    BufferSignal,
    InsufficientDataError,
)
from ._cycle_to_tgm import LSAContexts
from ._dataclass import SingleCycleData
from ._pyjapc import PyJapc2Pyda, PyJapcEndpoint, SubscriptionCallback

__all__ = ["Acquisition"]

DEV_LSA_B = "rmi://virtual_sps/SPSBEAM/B"
DEV_LSA_BDOT = "rmi://virtual_sps/SPSBEAM/BDOT"
DEV_LSA_I = "MBI/REF.TABLE.FUNC.VALUE"

DEV_MEAS_I = "MBI/LOG.I.MEAS"
DEV_MEAS_B = "SR.BMEAS-SP-B-SD/CycleSamples#samples"

TRIGGER_EVENT = "XTIM.SX.FCY2500-CT/Acquisition"
TRIGGER_DYNECO = "XTIM.SX.APECO-CT/Acquisition"
START_CYCLE = "XTIM.SX.SCY-CT/Acquisition"
START_SUPERCYCLE = "SX.CZERO-CTML/SuperCycle"


ENDPOINT2BF = {
    DEV_MEAS_I: BufferData.MEAS_I,
    DEV_MEAS_B: BufferData.MEAS_B,
    DEV_LSA_I: BufferData.PROG_I,
    DEV_LSA_B: BufferData.PROG_B,
}

ENDPOINT2SIG = {
    TRIGGER_EVENT: BufferSignal.FOREWARNING,
    START_CYCLE: BufferSignal.CYCLE_START,
    TRIGGER_DYNECO: BufferSignal.DYNECO,
}


ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)  # noqa: E501


log = logging.getLogger(__name__)

from_utc_ns: Callable[[Union[int, float]], datetime] = partial(
    from_timestamp, from_utc=True, unit="ns"
)


class Acquisition:
    data_acquired: Signal  # PropertyRetrievalResponse
    new_buffer_data: Signal  # list[SingleCycleData]
    cycle_mapping_changed: Signal  # str
    cycle_started: Signal  # str, str, int
    supercycle_started: Signal  # int, str
    supercycle_changed: Signal  # no argument

    new_measured_data: Signal  # SingleCycleData
    new_programmed_cycle: Signal  # SingleCycleData

    def __init__(
        self,
        min_buffer_size: int = 300000,
        buffer_only_measured: bool = False,
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
        self._buffer = AcquisitionBuffer(
            min_buffer_size, buffer_only_measured=buffer_only_measured
        )

        self._async_handles: dict[str, AsyncIOSubscription] = {}
        self._subscribe_handles: dict[str, SubscriptionCallback] = {}
        self._main_task: Optional[asyncio.Task] = None

        if japc_provider is None:
            assert AuthenticationClient is not None
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

        # Get mapped contextd
        self._pls_to_lsa: dict[str, str]
        self._lsa_to_pls: dict[str, str]
        self._current_supercycle: int | None = None

        self._lsa_contexts = LSAContexts(machine="SPS")
        self._lsa_contexts.update()

        self._pls_to_lsa = self._lsa_contexts.pls_to_lsa
        self._lsa_to_pls = self._lsa_contexts.lsa_to_pls

        self._stop_execution = False

        # set up PyDA and PyJApc
        self._japc_simple = SimpleClient(provider=japc_provider)
        self._japc_async = AsyncIOClient(provider=japc_provider)
        japc = PyJapc(incaAcceleratorName="SPS", selector="SPS.USER.ALL")
        japc.rbacLogin()
        self._japc = PyJapc2Pyda(japc)

        # Initialize signals, but don't start them
        self.data_acquired = Signal(PropertyRetrievalResponse)
        self.new_buffer_data = self._buffer.new_buffered_data
        self.cycle_mapping_changed = Signal(str)  # LSA cycle name
        self.cycle_started = Signal(str, str, float)  # PLS, LSA, timestamp

        self.supercycle_started = Signal(int, str)  # ID, name
        self.supercycle_changed = Signal()

        # add buffer signals
        self.new_measured_data = self._buffer.new_measured_data
        self.new_programmed_cycle = self._buffer.new_programmed_cycle
        self.data_acquired.connect(self._handle_acquisition)

        # This must be executed in the main thread (I think)
        signal(SIGINT, lambda *_: self.stop())
        signal(SIGTERM, lambda *_: self.stop())  # noqa

    def run(self) -> Thread:
        def wrapper() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            self._main_task = loop.create_task(self._run())

            loop.run_until_complete(self._main_task)

        th = Thread(target=wrapper)
        log.debug("Starting acquisition in new thread: " + th.name)
        th.start()

        return th

    def stop(self) -> None:
        """
        Stops the acquisition task and consequently the thread.
        """
        if self._main_task is None:
            log.warning(
                "Acquisition event loop has not yet been started. "
                "Nothing to cancel."
            )
            return

        log.debug("Acquisition received stop signal.")

        self._main_task.cancel()
        Signal.cancel_all()

    async def _run(self) -> None:
        """
        Starts the acquisition process in a separate thread.
        """
        # TODO: perform synchronous GETS to initialize the buffer
        Signal.start_all(asyncio.get_running_loop())
        try:
            handles = self._setup_subscriptions()
        except:  # noqa
            log.exception("Error while setting up subscriptions.")
            return

        for name, sub in self._subscribe_handles.items():
            log.debug(f"Starting subscription {name}.")
            sub.handle.startMonitoring()

        try:
            async for response in AsyncIOClient.merge_subscriptions(*handles):
                if self._stop_execution:
                    break

                try:
                    self._handle_acquisition(response)
                except Exception as e:
                    log.exception("Error handling acquisition event." + str(e))
        except asyncio.CancelledError:
            log.debug("Acquisition loop received cancel event.")

    @property
    def buffer(self) -> AcquisitionBuffer:
        return self._buffer

    def _setup_subscriptions(self) -> list[AsyncIOSubscription]:
        pyda_subscriptions = [
            ("StartSuperCycle", START_SUPERCYCLE, ""),
            ("StartCycle", START_CYCLE, "SPS.USER.ALL"),
            ("Forewarning", TRIGGER_EVENT, "SPS.USER.ALL"),
            ("PartialEconomy", TRIGGER_DYNECO, "SPS.USER.ALL"),
            ("MeasuredCurrent", DEV_MEAS_I, "SPS.USER.ALL"),
            ("MeasuredBField", DEV_MEAS_B, "SPS.USER.ALL"),
        ]

        pyjapc_subscriptions = [
            ("ProgrammedCurrent", DEV_LSA_I, "SPS.USER.ALL"),
            ("ProgrammedBField", DEV_LSA_B, "SPS.USER.ALL"),
        ]

        log.debug("Setting up subscriptions.")
        log.debug("Performing GETs to initialize the buffer.")

        self._handle_acquisition(
            self._japc_simple.get(START_SUPERCYCLE, context="")
        )

        handle: Union[SubscriptionCallback, AsyncIOSubscription]
        log.debug("Subscribing to events.")
        for name, endpoint, selector in pyda_subscriptions:
            log.debug(f"Subscribing to {endpoint} with selector {selector}.")
            handle = self._japc_async.subscribe(
                PyJapcEndpoint.from_str(endpoint), context=selector
            )

            self._async_handles[name] = handle

        for name, endpoint, selector in pyjapc_subscriptions:
            log.debug(f"Subscribing to {endpoint} with selector {selector}.")
            handle = self._japc.subscribe(
                endpoint, context=selector, callback=self.data_acquired.emit
            )

            self._subscribe_handles[name] = handle

        return list(self._async_handles.values())

    def _handle_acquisition(
        self, response: PropertyRetrievalResponse, allow_empty: bool = False
    ) -> None:
        """
        Ensures that the acquisition does not contain any errors,
        and sends it through the :attr:`data_acquired` signal.
        """
        if response.notification_type == "FIRST_UPDATE":
            msg = (
                "Received first update for "
                f"{response.query.endpoint}@{response.query.context}. "
            )
            if str(response.query.endpoint) not in (DEV_LSA_I, DEV_LSA_B):
                msg += "Discarding it."
                log.debug(msg)
                return
            else:
                log.debug(msg)

        log.debug(f"Received acquisition event: {response.query.endpoint}.")
        if response.exception is not None:
            if not allow_empty and not str(response.exception).endswith(
                "is not mapped to any context."
            ):
                log.error(
                    "An error occurred trying to access value of event: "
                    f"{response.query.endpoint}:"
                    f"\n{str(response.exception)}."
                )
            else:
                log.debug("Empty response in event. Skipping it.")
            return

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

        selector = str(value.header.selector)
        cycle = self._pls_to_lsa.get(selector)
        endpoint = str(response.query.endpoint)
        if endpoint == START_SUPERCYCLE:
            self._on_start_supercycle(response)
            return
        elif endpoint == TRIGGER_EVENT:
            self._on_forewarning(response)
            return
        if cycle is None:
            msg = (
                "Received event from timing user not mapped in supercycle: "
                f"{value.header.selector}."
            )
            if response.notification_type == "FIRST_UPDATE":
                log.debug("First update: " + msg)
            else:
                log.error(msg)

            return

        if endpoint in ENDPOINT2SIG:
            assert cycle is not None
            assert cycle_timestamp is not None

            if endpoint == START_CYCLE:
                self.cycle_started.emit(selector, cycle, cycle_timestamp)

            self._buffer.dispatch_signal(
                ENDPOINT2SIG[endpoint], cycle, cycle_timestamp, selector
            )
            return
        elif endpoint in ENDPOINT2BF:
            data = response.value.get("value")
            if data is None:
                if allow_empty:
                    return
                log.error(f"Received event with no data from {endpoint}.")
                return

            assert cycle_timestamp is not None
            self._buffer.dispatch_data(
                ENDPOINT2BF[endpoint], cycle, cycle_timestamp, data
            )
            return
        else:
            log.error(f"Received event from unknown endpoint " f"{endpoint}.")
            return

    def _on_forewarning(self, response: PropertyRetrievalResponse) -> None:
        """
        Handler for the forewarning event. This is triggered 2500ms before
        the cycle plays.

        This function notify the buffer that a new cycle is about to start,
        and then retrieve the latest buffered data if it is available.
        """
        cycle = response.value.get("lsaCycleName")
        cycle_timestamp = response.value.header.cycle_timestamp
        assert cycle_timestamp is not None
        assert cycle is not None
        log.debug(
            f"Cycle {cycle} at {from_utc_ns(cycle_timestamp)} is about to "
            "start. Notifying buffer."
        )

        if cycle not in self.buffer.known_cycles:
            log.debug(
                f"Cycle {cycle} is not known to the buffer. "
                "Fetching LSA programs."
            )

            selector = self._lsa_to_pls[cycle]
            try:
                self._handle_acquisition(
                    self._japc.get(DEV_LSA_I, context=selector)
                )
                self._handle_acquisition(
                    self._japc.get(DEV_LSA_B, context=selector)
                )
            except:  # noqa: broad-except
                log.exception("Error fetching LSA programs.")
                return

        self.buffer.new_cycle(cycle, cycle_timestamp, self._lsa_to_pls[cycle])

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
            users: Iterable[str] | None, cycles: Iterable[str] | None
        ) -> tuple[dict[str, str], dict[str, str]]:
            lsa_to_pls = {}
            pls_to_lsa = {}

            if users is None or cycles is None:
                raise ValueError("Neither users and cycles can be None.")

            for user, cycle in zip(users, cycles):
                user = "SPS.USER." + user

                if self._pls_to_lsa.get(user) != cycle:
                    mappings_changed.append(user)

                pls_to_lsa[user] = cycle
                lsa_to_pls[cycle] = user

                log.debug(f"Mapping {user} to {cycle}")

            return lsa_to_pls, pls_to_lsa

        log.debug("Mapping normal cycles.")
        normal_lsa_to_pls, normal_pls_to_lsa = parse_mappings(
            value.get("normalUsers"), value.get("normalLsaCycleNames")
        )

        log.debug("Mapping spare cycles.")
        spare_lsa_to_pls, spare_pls_to_lsa = parse_mappings(
            value.get("spareUsers"), value.get("spareLsaCycleNames")
        )

        def log_line(text: str) -> str:
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
        new_lsa_to_pls = {**normal_lsa_to_pls, **spare_lsa_to_pls}
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

        supercycle_id = value["bcdId"]
        supercycle_name = value["bcdName"]

        log.debug(f"Supercycle {supercycle_id} ({supercycle_name}) started.")
        if supercycle_id != self._current_supercycle:
            self.supercycle_started.emit(supercycle_id, supercycle_name)
            self.supercycle_changed.emit()

            self._current_supercycle = supercycle_id
