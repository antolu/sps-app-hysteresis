from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Callable, Optional, Union

from jpype import JException as JavaException
from pyda.data import (
    AcquiredPropertyData,
    CompositeContext,
    PropertyAccessError,
    PropertyAccessQuery,
    PropertyRetrievalResponse,
    RetrievalHeader,
    StandardEndpoint,
    TimingSelector,
)
from pyjapc import PyJapc

log = logging.getLogger(__name__)


ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)  # noqa: E501


class PyJapcEndpoint(StandardEndpoint):
    @classmethod
    def from_str(cls, value: str) -> PyJapcEndpoint:
        m = ENDPOINT_RE.match(value)

        if not m:
            raise ValueError(f"Not a valid endpoint: {value}.")

        return cls(
            device_name=m.group("device"),
            property_name=m.group("property"),
        )


def notification_type(header: dict[str, Any]) -> str:
    if header["isFirstUpdate"]:
        return "FIRST_UPDATE"
    elif header["isImmediateUpdate"]:
        return "SETTING_UPDATE"
    else:
        return "SERVER_UPDATE"


class SubscriptionCallback:
    def __init__(
        self,
        param_name: str,
        selector: str,
        callback: Callable[[PropertyRetrievalResponse], None],
    ) -> None:
        self._callback = callback
        self._endpoint = PyJapcEndpoint.from_str(param_name)
        self._context = CompositeContext(TimingSelector(selector))
        self._query = PropertyAccessQuery(self._endpoint, self._context)

        self.handle: Any = None

    def on_value_received(
        self, param_name: str, value: dict[str, Any], header: dict[str, Any]
    ) -> None:
        acq_prop = pyjapc_acq_to_pyda(value, header)
        prop_response = PropertyRetrievalResponse(
            self._query,
            notification_type=notification_type(header),
            value=acq_prop,
        )

        self._callback(prop_response)

    def on_exception_received(
        self, param_name: str, exc_desc: str, exception: Any
    ) -> None:
        err_prop = PropertyAccessError(exception)

        prop_response = PropertyRetrievalResponse(
            self._query, exception=err_prop
        )

        self._callback(prop_response)


class PyJapc2Pyda:
    def __init__(self, pyjapc: Optional[PyJapc] = None) -> None:
        self._pyjapc = pyjapc or PyJapc(
            incaAcceleratorName=None, selector="SPS.USER.ALL"
        )
        self._pyjapc.rbacLogin()

    def get(
        self, endpoint: str, context: str = ""
    ) -> PropertyRetrievalResponse:
        value: Optional[AcquiredPropertyData] = None
        exception: Optional[PropertyAccessError] = None
        not_type: Optional[str] = None

        query = PropertyAccessQuery(
            endpoint=PyJapcEndpoint.from_str(endpoint),
            context=CompositeContext(TimingSelector(context)),
        )
        try:
            response, header = self._pyjapc.getParam(
                endpoint, timingSelectorOverride=context, getHeader=True
            )
            value = pyjapc_acq_to_pyda(response, header)
            not_type = notification_type(header)
        except JavaException as e:
            log.exception(
                f"An error occurred while getting {endpoint}@{context}"
            )
            exception = PropertyAccessError(str(e))

        return PropertyRetrievalResponse(
            query=query,
            notification_type=not_type,
            value=value,
            exception=exception,
        )

    def subscribe(
        self,
        endpoint: str,
        callback: Callable[[PropertyRetrievalResponse], None],
        context: str = "",
    ) -> SubscriptionCallback:
        sub_callback = SubscriptionCallback(endpoint, context, callback)
        try:
            handle = self._pyjapc.subscribeParam(
                endpoint,
                onValueReceived=sub_callback.on_value_received,  # type: ignore
                onException=sub_callback.on_exception_received,
                timingSelectorOverride=context,
                getHeader=True,
            )
            sub_callback.handle = handle
        except JavaException:
            raise

        return sub_callback


def pyjapc_acq_to_pyda(
    results: Any, header: dict[str, Union[str, datetime, bool]]
) -> AcquiredPropertyData:
    def datetime_to_ns(d: Optional[datetime]) -> float:
        assert d is not None

        if d.timestamp() == 0.0:
            return 0.0
        else:
            return d.timestamp() * 1e9

    acq_time = header.get("acqStamp")
    cycle_time = header.get("cycleStamp")

    assert isinstance(acq_time, (datetime, type(None)))
    assert isinstance(cycle_time, (datetime, type(None)))

    pyda_header = RetrievalHeader(
        acquisition_timestamp=datetime_to_ns(acq_time),
        cycle_timestamp=datetime_to_ns(cycle_time),
        set_timestamp=datetime_to_ns(header.get("setStamp")),  # type: ignore
        selector=header.get("selector"),  # type: ignore
    )

    if not isinstance(results, dict):
        results = {"value": results}

    return AcquiredPropertyData(header=pyda_header, data=results)
