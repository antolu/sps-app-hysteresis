"""
This module contains functions for trimming LSA functions
"""
from __future__ import annotations

import logging
import re
import typing

import numpy as np
from jpype import JFloat
from pjlsa import LSAClient

log = logging.getLogger(__name__)


if typing.TYPE_CHECKING:
    from cern.lsa.domain.settings import StandAloneContext


PARAMETER_TYPE = typing.Literal[
    "function", "scalar_array", "scalar_bool", "scalar_float"
]


TGM_USER_RE = re.compile(r"\w+\.USER\.\w+")


# pyright: reportMissingModuleSource=false
class TrimManager:
    """Class to handle LSA stuff and pjlsa trims."""

    def __init__(self, lsa: LSAClient):
        self._lsa = lsa
        self._active_context: StandAloneContext | None = None
        self.logger = logging.getLogger(__name__)

        with self._lsa.java_api():
            from cern.lsa.client import ContextService
            from cern.lsa.client import ParameterService as LSAParameterService
            from cern.lsa.client import (
                ServiceLocator,
                SettingService,
                TrimService,
            )

        self._trim_service = ServiceLocator.getService(TrimService)
        self._parameter_service = ServiceLocator.getService(
            LSAParameterService
        )
        self._context_service = ServiceLocator.getService(ContextService)
        self._setting_service = ServiceLocator.getService(SettingService)

    @property
    def active_context(self) -> StandAloneContext:
        if self._active_context is None:
            raise ValueError("LSA context not yet initialized.")
        return self._active_context

    @active_context.setter
    def active_context(self, value: str | StandAloneContext) -> None:
        """
        Set the active context by name or by object.

        If the value is a string, it is assumed to be an LSA cycle
        or a TGM user, e.g. SPS.USER.SFTPRO1.

        To use a BeamProcess context, pass the object directly.
        """
        if isinstance(value, str):
            if TGM_USER_RE.match(value):
                value = self._context_service.findStandAloneContextByUser(
                    value
                )
            else:
                value = self._context_service.findStandAloneCycle(value)

        self._active_context = value

    def get_current_trim(
        self,
        parameter_name: str,
        part: typing.Literal["VALUE", "TARGET", "CORRECTION"] = "VALUE",
        parameter_type: PARAMETER_TYPE = "function",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the current LSA function for the given parameter.
        The part argument currently only handles TARGET
        but should be extended to allow for VALUE or CORRECTION as well.
        """
        #  TODO: Also differentiate between functions and scalars /
        #  scalar arrays
        with self._lsa.java_api():
            from cern.lsa.domain.settings import (
                ContextSettingsRequest,
                SettingPartEnum,
                Settings,
            )
        cycle = self._active_context
        param = self._parameter_service.findParameterByName(parameter_name)

        assert cycle is not None
        req = ContextSettingsRequest.byStandAloneContextAndParameters(
            cycle, [param]
        )
        ctx_settings = self._setting_service.findContextSettings(req)

        if parameter_type == "function":
            if part == "TARGET":
                vals = Settings.computeContextValue(
                    cycle,
                    ctx_settings.getParameterSettings(param),
                    SettingPartEnum.TARGET,
                )
            elif part == "VALUE":
                vals = Settings.computeContextValue(
                    cycle,
                    ctx_settings.getParameterSettings(param),
                    SettingPartEnum.VALUE,
                )
            elif part == "CORRECTION":
                vals = Settings.computeContextValue(
                    cycle,
                    ctx_settings.getParameterSettings(param),
                    SettingPartEnum.CORRECTION,
                )
            else:
                raise ValueError(
                    "'part' has to be either 'TARGET', 'VALUE', or "
                    "'CORRECTION'."
                )

            x_vals = np.array(list(vals.toXArray()))
            y_vals = np.array(list(vals.toYArray()))
            return x_vals, y_vals
        else:
            raise NotImplementedError("Can only fetch functions for now.")

    @typing.overload
    def send_trim(
        self,
        parameter_name: str,
        values: typing.Any,
        comment: str,
        parameter_type: PARAMETER_TYPE = "function",
    ):
        ...

    @typing.overload
    def send_trim(
        self,
        parameter_name: list[str],
        values: list[typing.Any],
        comment: str,
        parameter_type: list[PARAMETER_TYPE],
    ):
        ...

    def send_trim(
        self,
        parameter_name: str | list[str],
        values: typing.Any | list[typing.Any],
        comment: str,
        parameter_type: PARAMETER_TYPE | list[PARAMETER_TYPE] = "function",
    ):
        """
        Send trims via pjlsa. Function can be called either on lists of
        parameters, values, etc. or on a single one.
        If passed as list, will be all part of the same trim.
        """
        # TODO: to be improved -- make more readable and more generic.
        with self._lsa.java_api():
            from cern.accsoft.commons.value import ValueFactory
            from cern.lsa.domain.settings.factory import TrimRequestBuilder
            from jpype import JArray, JDouble

        assert self._active_context is not None
        trim_builder = (
            TrimRequestBuilder()
            .setContext(self._active_context)
            .setPropagateToChildren(True)  # True
            .setDrive(True)  # True
            .setDescription(comment)
        )

        if isinstance(parameter_name, list):
            if not isinstance(values, list):
                raise ValueError(
                    "If parameter_name is a list, values has to be a"
                    "list as well."
                )
            if not isinstance(parameter_type, list):
                raise ValueError(
                    "If parameter_name is a list, parameter_type has "
                    "to be a list as well."
                )

            if len(parameter_name) != len(values):
                raise ValueError(
                    "parameter_name and values have to be of the same "
                    "length."
                )
            if len(parameter_name) != len(parameter_type):
                raise ValueError(
                    "parameter_name and parameter_type have to be of "
                    "the same length."
                )

        if not isinstance(parameter_name, list):
            parameter_name = [parameter_name]
        if not isinstance(values, list):
            values = [values]
        if not isinstance(parameter_type, list):
            parameter_type = [parameter_type]

        for param, val, typ in zip(parameter_name, values, parameter_type):
            parameter = self._parameter_service.findParameterByName(param)
            if typ == "function":
                func = ValueFactory.createFunction(
                    JArray(JDouble)(val[0]), JArray(JDouble)(val[1])
                )
                trim_builder.addFunction(parameter, func)
            elif typ == "scalar_array":
                scalar_array = ValueFactory.createScalarArray(
                    JArray(JDouble)(val)
                )
                trim_builder.addScalar(parameter, scalar_array)
            elif typ == "scalar_bool":
                scalar = ValueFactory.createScalar(val)
                trim_builder.addScalar(parameter, scalar)
            elif typ == "scalar_float":
                scalar = ValueFactory.createScalar(
                    JFloat(np.array(val, dtype="float"))
                )
                trim_builder.addScalar(parameter, scalar)
            else:
                raise ValueError(
                    f"TrimManager: Parameter type {parameter_type} not known."
                )

        trim_request = trim_builder.build()
        trim_response = self._trim_service.trimSettings(trim_request)
        return trim_response
