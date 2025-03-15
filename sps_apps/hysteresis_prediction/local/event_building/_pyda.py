from __future__ import annotations

import re

import pyda.access

ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)


class JapcEndpoint(pyda.access.StandardEndpoint):
    """
    Endpoint for JAPC access, which bypasses the check in device_name and property_name
    """

    def __init__(self, *, device_name: str, property_name: str) -> None:
        self._dev = device_name
        self._prop = property_name

    @classmethod
    def from_str(cls, value: str) -> JapcEndpoint:
        m = ENDPOINT_RE.match(value)

        if not m:
            msg = f"Not a valid endpoint: {value}."
            raise ValueError(msg)

        return cls(
            device_name=m.group("device"),
            property_name=m.group("property"),
        )
