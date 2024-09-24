from __future__ import annotations

import collections
import dataclasses
import datetime
import logging
import re
import typing

import hystcomp_utils.cycle_data
import pyda
import pyda._metadata
import pyda.clients.callback
import pyda.data
import pyda.providers
import pyda_japc
from qtpy import QtCore

log = logging.getLogger(__name__)


@dataclasses.dataclass
class Subscription:
    name: str
    parameter: str
    selector: str = "SPS.USER.ALL"
    ignore_first_updates: bool = False


class BufferedSubscription(Subscription):
    buffer_size: int = 1


ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)  # noqa: E501


T = typing.TypeVar(
    "T", pyda.data.PropertyRetrievalResponse, hystcomp_utils.cycle_data.CycleData
)


class CycleStampSubscriptionBuffer(typing.Generic[T]):
    def __init__(self, parameter: str, buffer_size: int = 1) -> None:
        self._parameter = parameter
        self._buffer: collections.deque[T] = collections.deque(maxlen=buffer_size)
        self._index: dict[float, int] = {}  # cycle_stamp -> index

    def __contains__(self, item: float) -> bool:
        return item in self._index

    def __getitem__(self, item: float) -> T:
        return self._buffer[self._index[item]]

    def __len__(self) -> int:
        return len(self._buffer)

    def __str__(self) -> str:
        return f"{self._parameter}: {len(self._buffer)} items"

    def append(self, fspv: T) -> None:
        self._buffer.append(fspv)
        if isinstance(fspv, hystcomp_utils.cycle_data.CycleData):
            cycle_timestamp = fspv.cycle_timestamp
        else:
            cycle_timestamp = fspv.value.header.cycle_timestamp
        self._index[cycle_timestamp] = len(self._buffer) - 1

    def clear(self) -> None:
        self._buffer.clear()
        self._index.clear()

    def clear_older_than(self, cycle_stamp: float) -> None:
        if not self._buffer:
            return

        while self._buffer[0].cycle_timestamp <= cycle_stamp:
            self._index.pop(self._buffer[0].cycle_timestamp)
            self._buffer.popleft()


def endpoint_from_str(value: str) -> pyda.data.StandardEndpoint:
    m = ENDPOINT_RE.match(value)

    if not m:
        msg = f"Not a valid endpoint: {value}."
        raise ValueError(msg)

    return pyda.data.StandardEndpoint(
        device_name=m.group("device"),
        property_name=m.group("property"),
    )


class EventBuilderAbc(QtCore.QObject):
    """
    Abstract base class for event builders.
    """

    cycleDataAvailable = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)

    def __init__(
        self,
        subscriptions: list[Subscription] | None = None,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        no_metadata_source: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        QtCore.QObject.__init__(self, parent=parent)

        self._subscriptions = subscriptions or []

        provider = provider or (pyda_japc.JapcProvider() if self._needs_da() else None)

        if not self._needs_da():
            self._da = None
        else:
            self._da = pyda.CallbackClient(
                provider=(
                    pyda.providers.Provider(
                        data_source=provider,
                        metadata_source=pyda._metadata.NoMetadataSource(),
                    )
                    if no_metadata_source
                    else provider
                ),
            )

        self._handles: dict[str, pyda.clients.callback.CallbackSubscription] = {}
        self._handles |= self._setup_subscriptions(subscriptions or [])

    def _needs_da(self) -> bool:
        return len(self._subscriptions) > 0

    def _setup_subscriptions(
        self, subscriptions: typing.Sequence[Subscription]
    ) -> dict[str, pyda.clients.callback.CallbackSubscription]:
        if self._da is None:
            if len(subscriptions) > 0:
                msg = "Cannot setup subscriptions without a DA."
                raise ValueError(msg)

            return {}

        handles = {}
        for sub in subscriptions:
            endpoint = endpoint_from_str(sub.parameter)
            handle = self._da.subscribe(
                endpoint=endpoint,
                context=sub.selector,
                receive_first_updates=not sub.ignore_first_updates,
                callback=self.handle_acquisition,
            )

            handles[sub.name] = handle

        return handles

    def start(self) -> None:
        for handle in self._handles.values():
            msg = f"Starting subscription for {handle.query}."
            log.debug(msg)
            handle.start()

    def handle_acquisition(self, fspv: pyda.data.PropertyRetrievalResponse) -> None:

        try:
            self._handle_acquisition_impl(fspv)
        except:  # noqa: E722
            msg = f"Error handling acquisitin for {fspv.query.endpoint}@{fspv.query.context}."
            log.exception(msg)

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        raise NotImplementedError

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = "This method is optional and should be implemented by the subclass."
        raise NotImplementedError(msg)


class BufferedSubscriptionEventBuilder(EventBuilderAbc):
    def __init__(
        self,
        subscriptions: list[Subscription] | None = None,
        buffered_subscriptions: list[BufferedSubscription] | None = None,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        no_metadata_source: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        self._buffers: dict[str, dict[str, pyda.data.PropertyRetrievalResponse]] = (
            {}
        )  # endpoint -> selector -> buffer
        self._buffered_subscriptions = (
            [sub.parameter for sub in buffered_subscriptions]
            if buffered_subscriptions
            else []
        )

        super().__init__(
            subscriptions,
            provider=provider,
            no_metadata_source=no_metadata_source,
            parent=parent,
        )

        if buffered_subscriptions:
            self._handles |= self._setup_subscriptions(buffered_subscriptions or [])

    def _needs_da(self) -> bool:
        return super()._needs_da() or len(self._buffered_subscriptions) > 0

    def handle_acquisition(self, fspv: pyda.data.PropertyRetrievalResponse) -> None:
        if str(fspv.query.endpoint) in self._buffered_subscriptions:
            try:
                self._handle_buffered_acquisition_impl(fspv)
            except:  # noqa: E722
                msg = f"Error handling buffered acquisition for {fspv.query.endpoint}@{fspv.query.context}."
                log.exception(msg)
            finally:
                return

        super().handle_acquisition(fspv)

    def _handle_buffered_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        if fspv.exception is not None:
            msg = f"Received exception for {fspv.query.endpoint}@{fspv.query.context}: {fspv.exception}."
            log.error(msg)
            return

        parameter = str(fspv.query.endpoint)
        selector = str(fspv.query.context)

        msg = f"Received buffered acquisition for {parameter}@{selector}."
        log.debug(msg)
        if parameter not in self._buffers:
            self._buffers[parameter] = {}

        self._buffers[parameter][selector] = fspv

    def _buffer_has_data(self, parameter: str, selector: str) -> bool:
        return parameter in self._buffers and selector in self._buffers[parameter]

    def _get_buffered_data(
        self, parameter: str, selector: str
    ) -> pyda.data.PropertyRetrievalResponse:
        return self._buffers[parameter][selector]


class CycleStampGroupedTriggeredEventBuilder(BufferedSubscriptionEventBuilder):
    def __init__(
        self,
        subscriptions: list[Subscription] | None = None,
        buffered_subscriptions: list[BufferedSubscription] | None = None,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        track_cycle_data: bool = False,
        buffer_size: int = 10,
        no_metadata_source: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions,
            buffered_subscriptions,
            provider=provider,
            no_metadata_source=no_metadata_source,
            parent=parent,
        )

        self._cycle_stamp_buffers: dict[
            str,
            dict[
                str, CycleStampSubscriptionBuffer[pyda.data.PropertyRetrievalResponse]
            ],
        ] = {sub.parameter: {} for sub in buffered_subscriptions or []}

        self._buffer_size = buffer_size

        self._track_cycle_data = track_cycle_data
        self._cycle_data_buffer: dict[
            str, CycleStampSubscriptionBuffer[hystcomp_utils.cycle_data.CycleData]
        ] = {}

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        if not self._track_cycle_data:
            msg = "Tracking cycle data is disabled."
            log.warning(msg)
            return

        selector = cycle_data.user
        if selector not in self._cycle_data_buffer:
            self._cycle_data_buffer[selector] = CycleStampSubscriptionBuffer(
                selector, buffer_size=self._buffer_size
            )

        self._cycle_data_buffer[selector].append(cycle_data)

        if self._check_ready(cycle_data.cycle_timestamp, selector):
            msg = f"Received all buffered acquisitions for {selector} on cycle time {cycle_data.cycle_time}."
            log.debug(msg)

            self.onCycleStampGroupTriggered(cycle_data.cycle_timestamp, selector)
            self._clear_older_than(cycle_data.cycle_timestamp, selector)

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        msg = f"{self.__class__.__name__} does not subscribe to triggers."
        raise NotImplementedError(msg)

    def onCycleStampGroupTriggered(self, cycle_stamp: float, selector: str) -> None:
        raise NotImplementedError

    def _clear_older_than(self, cycle_stamp: float, selector) -> None:
        cycle_time = datetime.datetime.fromtimestamp(cycle_stamp / 1e9)
        msg = f"Clearing older than {cycle_time} for {selector}."
        log.debug(msg)

        for buffer in self._cycle_stamp_buffers.values():
            if selector in buffer:
                buffer[selector].clear_older_than(cycle_stamp)
        if self._track_cycle_data and selector in self._cycle_data_buffer:
            self._cycle_data_buffer[selector].clear_older_than(cycle_stamp)

    def _check_ready(self, cycle_stamp: float, selector: str) -> bool:
        for buffer in self._cycle_stamp_buffers.values():
            if selector not in buffer:
                return False
            if cycle_stamp not in buffer[selector]:
                return False
        if self._track_cycle_data:
            if cycle_stamp not in self._cycle_data_buffer[selector]:
                return False
        return True

    def _handle_buffered_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        if fspv.exception is not None:
            msg = f"Received exception for {fspv.query.endpoint}@{fspv.query.context}: {fspv.exception}."
            log.error(msg)
            return

        parameter = str(fspv.query.endpoint)
        selector = str(fspv.query.context)

        msg = f"Received buffered acquisition for {parameter}@{selector}."
        log.debug(msg)

        self._add_to_buffer(parameter, selector, fspv)

        if self._check_ready(fspv.value.header.cycle_timestamp, selector):
            msg = f"Received all buffered acquisitions for {selector} with cycle time {fspv.value.header.cycle_time()}"
            log.debug(msg)
            self.onCycleStampGroupTriggered(fspv.value.header.cycle_timestamp, selector)
            self._clear_older_than(fspv.value.header.cycle_timestamp, selector)

    def _add_to_buffer(
        self, parameter: str, selector: str, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        if selector not in self._cycle_stamp_buffers[parameter]:
            self._cycle_stamp_buffers[parameter][selector] = (
                CycleStampSubscriptionBuffer(selector, buffer_size=self._buffer_size)
            )

        self._cycle_stamp_buffers[parameter][selector].append(fspv)

    def _buffer_has_data(self, parameter: str, selector: str) -> bool:
        return parameter in self._cycle_stamp_buffers

    def _get_buffered_data(
        self, parameter: str, selector: str
    ) -> pyda.data.PropertyRetrievalResponse:
        raise NotImplementedError("This method is not implemented.")
