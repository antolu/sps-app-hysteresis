"""
This module implements the acquisition of data, both measured
and reference, and publishes it for external use.
"""

from __future__ import annotations

import collections
import logging
import re
import typing
from datetime import datetime
from functools import partial
from signal import SIGINT, SIGTERM, signal

import numpy as np
import pyda._metadata
import pyda.clients.callback
import pyda.data
import pyda.providers
from hystcomp_utils.cycle_data import CycleData, unflatten_cycle_data
from pyda.data import PropertyRetrievalResponse
from qtpy import QtCore

try:
    from pyda_japc import JapcProvider
    from pyrbac import AuthenticationClient  # noqa F401
except ImportError:
    # for my macos laptop
    AuthenticationClient = None  # type: ignore

from ..utils import from_timestamp
from ._cycle_to_tgm import LSAContexts

__all__ = ["Acquisition"]

DEV_BUFFER = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.EVENT/CycleWarning"

DEV_MEAS_I = "MBI/LOG.I.MEAS"
DEV_MEAS_B = "SR.BMEAS-SP-B-SD/CycleSamples#samples"

TRIGGER_EVENT = "SX.CZERO-CTML/CycleWarning"
# TRIGGER_EVENT = (
#     "rda3://UCAP-NODE-CSS-DSB-TEST/SX.UCAP.SCY-SPR-NXT/CycleWarning"
# )
TRIGGER_DYNECO = "XTIM.SX.APECO-CT/Acquisition"
START_CYCLE = "XTIM.SX.SCY-CT/Acquisition"
START_SUPERCYCLE = "SX.CZERO-CTML/SuperCycle"


ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)  # noqa: E501


log = logging.getLogger(__name__)

from_utc_ns: typing.Callable[[int | float], datetime] = partial(
    from_timestamp, from_utc=True, unit="ns"
)


class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer: collections.deque[CycleData] = collections.deque(maxlen=size)
        self.timestamp_index: dict[float, CycleData] = {}

    def add(self, cycle_data: CycleData):
        if len(self.buffer) == self.size:
            oldest_data = self.buffer.popleft()
            del self.timestamp_index[oldest_data.cycle_timestamp]
        self.buffer.append(cycle_data)
        self.timestamp_index[cycle_data.cycle_timestamp] = cycle_data

    def __getitem__(self, cycle_timestamp: float) -> CycleData:
        if cycle_timestamp in self.timestamp_index:
            return self.timestamp_index[cycle_timestamp]
        else:
            msg = f"CycleData with timestamp {cycle_timestamp} not found."
            raise KeyError(msg)

    def __contains__(self, cycle_timestamp: float) -> bool:
        return cycle_timestamp in self.timestamp_index


class JapcEndpoint(pyda.data.StandardEndpoint):
    @classmethod
    def from_str(cls, value: str) -> JapcEndpoint:
        m = ENDPOINT_RE.match(value)

        if not m:
            raise ValueError(f"Not a valid endpoint: {value}.")

        return cls(
            device_name=m.group("device"),
            property_name=m.group("property"),
        )


class Acquisition(QtCore.QObject):
    newBufferData = QtCore.Signal(list)  # list[CycleData]

    cycle_started = QtCore.Signal(str, str, float)  # str, str, int

    new_measured_data = QtCore.Signal(CycleData)  # CycleData
    sig_new_programmed_cycle = QtCore.Signal(CycleData)  # CycleData
    onNewPrediction = QtCore.Signal(CycleData)  # CycleData

    def __init__(
        self,
        japc_provider: JapcProvider | pyda.providers.Provider | None = None,
        parent: QtCore.QObject | None = None,
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
        super().__init__(parent=parent)

        self._callback_handles: dict[
            str, pyda.clients.callback.CallbackSubscription
        ] = {}

        japc_provider = japc_provider or default_japc_provider()

        self._buffer = RingBuffer(20)

        self._field_ref: dict[str, np.ndarray] = {}

        # Get mapped contextd
        self._pls_to_lsa: dict[str, str]
        self._lsa_to_pls: dict[str, str]
        self._current_supercycle: int | None = None

        self._lsa_contexts = LSAContexts(machine="SPS")
        self._lsa_contexts.update()

        self._pls_to_lsa = self._lsa_contexts.pls_to_lsa
        self._lsa_to_pls = self._lsa_contexts.lsa_to_pls

        # set up PyDA and PyJApc
        self._japc_simple = pyda.SimpleClient(provider=japc_provider)
        self._japc = pyda.CallbackClient(provider=japc_provider)

        # This must be executed in the main thread (I think)
        signal(SIGINT, lambda *_: self.stop())
        signal(SIGTERM, lambda *_: self.stop())  # noqa

    # def run(self) -> Thread:
    #     th = Thread(target=self._run)
    #     log.debug("Starting acquisition in new thread: " + th.name)
    #     th.start()
    #
    #     return th

    def stop(self) -> None:
        """
        Stops the acquisition task and consequently the thread.
        """

        log.debug("Acquisition received stop signal.")

    def _run(self) -> None:
        """
        Starts the acquisition process in a separate thread.
        """
        # TODO: perform synchronous GETS to initialize the buffer
        try:
            self._setup_subscriptions()
        except:  # noqa F722
            log.exception("Error while setting up subscriptions.")
            return

        for key, handle in self._callback_handles.items():
            handle.start()

    run = _run

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    def reset_reference(self) -> None:
        self._field_ref.clear()

    def _setup_subscriptions(self) -> None:
        pyda_subscriptions = [
            ("StartSuperCycle", START_SUPERCYCLE, ""),
            ("StartCycle", START_CYCLE, "SPS.USER.ALL"),
            ("Buffer", DEV_BUFFER, "SPS.USER.ALL"),
            # ("PartialEconomy", TRIGGER_DYNECO, "SPS.USER.ALL"),
            ("MeasuredCurrent", DEV_MEAS_I, "SPS.USER.ALL"),
            ("MeasuredBField", DEV_MEAS_B, "SPS.USER.ALL"),
        ]

        log.debug("Setting up subscriptions.")
        log.debug("Performing GETs to initialize the buffer.")

        self._handle_acquisition(self._japc_simple.get(START_SUPERCYCLE, context=""))

        log.debug("Subscribing to events.")
        for name, endpoint, selector in pyda_subscriptions:
            log.debug(f"Subscribing to {endpoint} with selector {selector}.")
            handle = self._japc.subscribe(
                JapcEndpoint.from_str(endpoint),
                context=selector,
                callback=self._handle_with_error,
                receive_first_updates=False,
            )

            self._callback_handles[name] = handle

    def _handle_with_error(self, response: PropertyRetrievalResponse) -> None:
        try:
            self._handle_acquisition(response)
        except Exception:  # noqa: E722
            log.exception("An exception occurred while handling acquisition event.")

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
            from_utc_ns(cycle_timestamp) if cycle_timestamp is not None else "N/A"
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
        elif endpoint == DEV_BUFFER:
            self._on_forewarning(response)
            return
        elif endpoint == START_CYCLE:
            self._on_start_cycle(response)
            return
        elif endpoint == DEV_MEAS_I:
            self._on_measured_current(response)
            return
        elif endpoint == DEV_MEAS_B:
            self._on_measured_field(response)
            return
        else:
            msg = f"Unknown endpoint {endpoint}."
            log.error(msg)

    def _on_forewarning(self, response: PropertyRetrievalResponse) -> None:
        """
        Handler for the forewarning event. This is triggered 2500ms before
        the cycle plays.

        This function notify the buffer that a new cycle is about to start,
        and then retrieve the latest buffered data if it is available.
        """
        selector = str(response.value.header.selector)
        cycle = self._pls_to_lsa.get(selector, "N/A")
        if cycle == "N/A":
            log.warning(
                f"User {selector} is not mapped to any LSA cycle. "
                "Skipping this cycle."
            )
            return
        cycle_timestamp = response.value.header.cycle_timestamp
        assert cycle_timestamp is not None
        assert cycle is not None
        log.debug(
            f"Cycle {cycle} at {from_utc_ns(cycle_timestamp)} is about to "
            "start. Notifying buffer."
        )

        log.debug("Query buffer for latest buffered data.")
        buffer_dict = dict(response.value)
        buffer: list[CycleData] = unflatten_cycle_data(buffer_dict)

        last_cycle = buffer[-1]
        last_cycle.field_meas = None
        last_cycle.current_meas = None
        self._buffer.add(last_cycle)

        log.debug("Buffered data available. Sending it to listeners.")

        log.debug("Notifying listeners of new programmed cycle.")
        self.sig_new_programmed_cycle.emit(buffer[-1])

        self.newBufferData.emit(buffer)

    def _on_start_cycle(self, response: PropertyRetrievalResponse) -> None:
        """
        Handler for the cycle start event. This is triggered when the cycle
        starts playing.

        This function notifies the buffer that a new cycle has started, and
        then retrieves the latest buffered data if it is available.
        """

        cycle_timestamp = response.value.header.cycle_timestamp
        user = str(response.value.header.selector)
        cycle = self._pls_to_lsa.get(user, "N/A")
        self.cycle_started.emit(user, cycle, cycle_timestamp)

    def _on_measured_current(self, response: PropertyRetrievalResponse) -> None:
        """
        Add measured current. If measured field is available, send the
        combined data to listeners.

        Parameters
        ----------
        response : PropertyRetrievalResponse
            The response container on retrieval.
        """
        value = response.value
        cycle_timestamp = value.header.cycle_timestamp
        assert cycle_timestamp is not None
        cycle_time = from_utc_ns(cycle_timestamp)
        log.debug(f"Measured current received at {cycle_time}.")

        if cycle_timestamp not in self._buffer:
            log.warning(
                "Cycle data not available. This is probably the first "
                "cycle of the acquisition."
            )
            return  # skip this cycle

        cycle_data = self._buffer[cycle_timestamp]
        cycle_data.current_meas = value["value"].flatten()

        if cycle_data.field_meas is not None:
            log.debug("Measured field is available. Sending data to listeners.")
            self.new_measured_data.emit(cycle_data)

    def _on_measured_field(self, response: PropertyRetrievalResponse) -> None:
        """
        Add measured field. If measured current is available, send the
        combined data to listeners.

        Parameters
        ----------
        response : PropertyRetrievalResponse
            The response container on retrieval.
        """
        value = response.value
        cycle_timestamp = value.header.cycle_timestamp
        assert cycle_timestamp is not None
        cycle_time = from_utc_ns(cycle_timestamp)
        log.debug(f"Measured field received at {cycle_time}.")

        if cycle_timestamp not in self._buffer:
            log.debug(
                "Cycle data not available. This is probably the first "
                "cycle of the acquisition."
            )
            return

        cycle_data = self._buffer[cycle_timestamp]
        cycle_data.field_meas = value["value"].flatten() / 1e4

        if cycle_data.current_meas is not None:
            log.debug("Measured current is available. Sending data to listeners.")
            self.new_measured_data.emit(cycle_data)

    def new_predicted_data(self, cycle_data: CycleData) -> None:
        """
        Callback function for the new predicted data event.
        This function is called when the predicted data is available.

        :param cycle_data: CycleData
            The cycle data with predicted field.
        """
        log.debug(
            f"Predicted data received for cycle {cycle_data.cycle} "
            f"with cycle time {cycle_data.cycle_time}."
        )

        cycle_data = self._buffer[cycle_data.cycle_timestamp]

        if cycle_data.cycle not in self._field_ref:
            log.debug("Field reference not available. Setting it to predicted field.")
            self._field_ref[cycle_data.cycle] = cycle_data.field_pred

        if cycle_data.field_ref is None:
            log.debug("Field reference is None. Setting it to predicted field.")
            cycle_data.field_ref = self._field_ref[cycle_data.cycle]

        self.onNewPrediction.emit(cycle_data)

    def _on_start_supercycle(self, response: PropertyRetrievalResponse) -> None:
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
            users: typing.Iterable[str] | None,
            cycles: typing.Iterable[str] | None,
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

        supercycle_id = value["bcdId"]
        supercycle_name = value["bcdName"]

        log.debug(f"Supercycle {supercycle_id} ({supercycle_name}) started.")


def default_japc_provider() -> JapcProvider:
    """
    Returns the default JAPC provider for the acquisition module.
    """
    assert AuthenticationClient is not None
    rbac_client = AuthenticationClient()
    log.info("JapcProvider not provided, logging into RBAC by location.")
    token = rbac_client.login_location()
    log.info("RBAC login successful. " f"Identified as {token.user_name}.")
    japc_provider = JapcProvider(
        rbac_token=token,
    )
    japc_provider = pyda.providers.Provider(
        data_source=japc_provider,
        metadata_source=pyda._metadata.NoMetadataSource(),
    )

    return japc_provider
