from __future__ import annotations

import dataclasses
import logging
import collections
import re
import abc

import pyda
import pyda.data
import pyda.clients.callback
import pyda.providers
import pyda._metadata
import pyda_japc

import hystcomp_utils.cycle_data


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


class CycleStampSubscriptionBuffer:
    def __init__(self, parameter: str, buffer_size: int = 1) -> None:
        self._parameter = parameter
        self._buffer: collections.deque[
            pyda.data.PropertyRetrievalResponse
        ] = collections.deque(maxlen=buffer_size)
        self._index: dict[float, int] = {}  # cycle_stamp -> index

    def __contains__(self, item: float) -> bool:
        return item in self._index

    def __getitem__(self, item: float) -> pyda.data.PropertyRetrievalResponse:
        return self._buffer[self._index[item]]

    def __len__(self) -> int:
        return len(self._buffer)

    def __str__(self) -> str:
        return f"{self._parameter}: {len(self._buffer)} items"

    def append(self, fspv: pyda.data.PropertyRetrievalResponse) -> None:
        self._buffer.append(fspv)
        self._index[fspv.value.header.cycle_timestamp] = len(self._buffer) - 1

    def clear(self) -> None:
        self._buffer.clear()
        self._index.clear()


def endpoint_from_str(value: str) -> pyda.data.StandardEndpoint:
    m = ENDPOINT_RE.match(value)

    if not m:
        msg = f"Not a valid endpoint: {value}."
        raise ValueError(msg)

    return pyda.data.StandardEndpoint(
        device_name=m.group("device"),
        property_name=m.group("property"),
    )


class EventBuilderAbc(abc.ABC, QtCore.QObject):
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

        provider = provider or pyda_japc.JapcProvider()

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

        self._handles: dict[
            str, pyda.clients.callback.CallbackSubscription
        ] = {}
        self._handles |= self._setup_subscriptions(subscriptions or [])

    def _setup_subscriptions(
        self, subscriptions: list[Subscription]
    ) -> dict[str, pyda.clients.callback.CallbackSubscription]:
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
            handle.start()

    def handle_acquisition(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:

        try:
            self._handle_acquisition_impl(fspv)
        except:  # noqa: E722
            msg = f"Error handling acquisitin for {fspv.query.endpoint}@{fspv.query.context}."
            log.exception(msg)

    @abc.abstractmethod
    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        pass

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        msg = "This method is optional and should be implemented by the subclass."
        raise NotImplementedError(msg)


class BufferedSubscriptionEventBuilder(EventBuilderAbc):
    def __init__(
        self,
        subscriptions: list[Subscription] | None = None,
        buffered_subscriptions: list[Subscription] | None = None,
        provider: pyda_japc.JapcProvider | None = None,
        *,
        no_metadata_source: bool = False,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            subscriptions,
            provider=provider,
            no_metadata_source=no_metadata_source,
            parent=parent,
        )

        self._buffers: dict[
            str, dict[str, pyda.data.PropertyRetrievalResponse]
        ] = {}  # endpoint -> selector -> buffer
        self._buffered_subscriptions = (
            [sub.parameter for sub in buffered_subscriptions]
            if buffered_subscriptions
            else []
        )

        if buffered_subscriptions:
            self._handles |= self._setup_subscriptions(
                buffered_subscriptions or []
            )

    def handle_acquisition(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
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
        return (
            parameter in self._buffers and selector in self._buffers[parameter]
        )

    def _get_buffered_data(
        self, parameter: str, selector: str
    ) -> pyda.data.PropertyRetrievalResponse:
        return self._buffers[parameter][selector]
