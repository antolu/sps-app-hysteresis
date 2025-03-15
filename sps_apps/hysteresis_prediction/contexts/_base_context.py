"""
This module contains a base context class that all other contexts should inherit from.

A context object is needed upon creation of the application, and is a global object that
is passed around to all other objects in the application, and determines the interaction
between the application and the outside world.
"""

from __future__ import annotations

import dataclasses
import datetime
import typing

from ..trim import TrimSettings


@dataclasses.dataclass
class ParameterNames:
    TRIGGER: str
    I_PROG: str
    B_PROG: str
    ADD_PROG_TRIGGER: str
    B_CORRECTION: str
    CYCLE_START: str

    I_PROG_DYNECO: str
    I_PROG_FULLECO: str
    FULLECO_TRIGGER: str

    RESET_REFERENCE_TRIGGER: str

    I_MEAS: str
    B_MEAS: str | None = None
    TRIM_SETTINGS: str | None = None


@dataclasses.dataclass
class UcapParameterNames:
    CYCLE_WARNING: str
    CYCLE_CORRECTION: str
    CYCLE_MEASURED: str
    RESET_REFERENCE: str
    SET_GAIN: str


class ApplicationContext:
    """Base context class that all other contexts should inherit from."""

    DEVICE: typing.Final[typing.Literal["MBI", "QF", "QD"]]
    B_MEAS_AVAIL: typing.Final[bool]

    PARAMS: typing.Final[ParameterNames]
    UCAP_PARAMS: typing.Final[UcapParameterNames | None]

    TRIM_SETTINGS: TrimSettings

    ONLINE: typing.Final[bool]

    TIMESTAMP: typing.Final[datetime.datetime]
    LOGDIR = "."

    def __init__(
        self,
        device: typing.Literal["MBI", "QF", "QD"],
        param_names: ParameterNames,
        trim_settings: TrimSettings,
        *,
        ucap_params: UcapParameterNames | None = None,
        b_meas_avail: bool | None = None,
    ):
        self.DEVICE = device
        self.PARAMS = param_names
        self.TRIM_SETTINGS = trim_settings
        self.B_MEAS_AVAIL = (
            self.PARAMS.B_MEAS is not None if b_meas_avail is None else b_meas_avail
        )
        self.UCAP_PARAMS = ucap_params

        self.ONLINE = ucap_params is not None

        self.TIMESTAMP = datetime.datetime.now()
