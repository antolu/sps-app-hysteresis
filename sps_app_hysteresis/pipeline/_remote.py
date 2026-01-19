"""
RemotePipeline: Display-only mode subscribing to UCAP device properties.

This pipeline subscribes to published properties from the UCAP hysteresis
compensation device and displays results. All processing (inference, correction,
metrics) is performed by the UCAP device.

TODO: JAPC SET commands for:
- Trim enabled/disabled
- Gain per cycle
- Trim time start/end
- Model loading (name + version) - prediction converter would need to subscribe
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
import typing

import hystcomp_utils.cycle_data
import pyda._metadata
import pyda.access
import pyda.clients
import pyda.data
import pyda.metadata
import pyda.providers
import pyda_japc
from hystcomp_event_builder import JapcEndpoint
from qtpy import QtCore

from ..contexts import RemoteParameterNames, app_context
from ._pipeline import Pipeline

log = logging.getLogger(__name__)
ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)


class RemotePipeline(Pipeline, QtCore.QObject):
    """Pipeline that subscribes to UCAP device for display-only mode."""

    _onCycleForewarning = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCorrectionCalculated = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCycleMeasured = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCycleStart = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onMetricsAvailable = QtCore.Signal(dict)
    _onNewReference = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)

    # Dummy signals for features not implemented in UCAP
    _dummy = QtCore.Signal()
    # NOTE: onTrimApplied not implemented - UCAP doesn't have trim functionality yet
    _dummyTrim = QtCore.Signal(
        hystcomp_utils.cycle_data.CycleData, datetime.datetime, str
    )

    def __init__(
        self,
        *,
        provider: pyda_japc.JapcProvider,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._provider = provider

        self._da = pyda.AsyncIOClient(
            provider=pyda.providers.Provider(
                data_source=self._provider,
                metadata_source=pyda.metadata.NoMetadataSource(),
            )
        )

        remote_params = app_context().REMOTE_PARAMS
        if remote_params is None:
            msg = "REMOTE_PARAMS not defined in app_context."
            raise ValueError(msg)

        self._remote_params = typing.cast(RemoteParameterNames, remote_params)

        self._should_stop = False
        self._handles: list[pyda.clients.asyncio.AsyncIOSubscription] = []

    def _setup_subscriptions(self) -> list[pyda.clients.asyncio.AsyncIOSubscription]:
        endpoints = [
            self._remote_params.CYCLE_WARNING,
            self._remote_params.CYCLE_CORRECTION,
            self._remote_params.CYCLE_MEASURED,
            self._remote_params.METRICS,
        ]

        return [
            self._da.subscribe(
                endpoint=JapcEndpoint.from_str(endpoint),
                context="SPS.USER.ALL",
            )
            for endpoint in endpoints
            if ENDPOINT_RE.match(endpoint) is not None
        ]

    def start(self) -> None:
        asyncio.run(self._start())

    async def _start(self) -> None:
        self._handles = self._setup_subscriptions()

        async for response in pyda.AsyncIOClient.merge_subscriptions(*self._handles):
            if self._should_stop:
                break

            if response.exception is not None:
                log.error(f"Error in subscription: {response.exception}")
                continue

            endpoint_str = str(response.query.endpoint)

            if endpoint_str == self._remote_params.CYCLE_WARNING:
                await self._handle_cycle_forewarning(response)
            elif endpoint_str == self._remote_params.CYCLE_CORRECTION:
                await self._handle_cycle_correction(response)
            elif endpoint_str == self._remote_params.CYCLE_MEASURED:
                await self._handle_cycle_measured(response)
            elif endpoint_str == self._remote_params.METRICS:
                await self._handle_metrics(response)
            else:
                log.warning(f"Received unknown endpoint: {endpoint_str}")

    def stop(self) -> None:
        self._should_stop = True

    # -------------------------------------------------------------------------
    # Signal properties (Pipeline interface)
    # -------------------------------------------------------------------------

    @property
    def onModelLoaded(self) -> QtCore.Signal:
        # NOTE: UCAP doesn't publish model loaded status.
        # Would need JAPC GET/SET for model configuration.
        return self._dummy

    @property
    def resetState(self) -> QtCore.Signal:
        # NOTE: Requires JAPC SET command to reset inference state.
        # Not implemented in display-only mode.
        return self._dummy

    @property
    def onCycleStart(self) -> QtCore.Signal:
        return self._onCycleStart

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        return self._onCycleForewarning

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        return self._onCorrectionCalculated

    @property
    def onTrimApplied(self) -> QtCore.Signal:
        # NOTE: UCAP doesn't have trim functionality yet.
        # When implemented, subscribe to UCAP Trim property.
        return self._dummyTrim

    @property
    def onNewReference(self) -> QtCore.Signal:
        return self._onNewReference

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        return self._onCycleMeasured

    @property
    def onMetricsAvailable(self) -> QtCore.Signal:
        return self._onMetricsAvailable

    # -------------------------------------------------------------------------
    # Slots (Pipeline interface)
    # -------------------------------------------------------------------------

    @QtCore.Slot(str)
    def onResetReference(self, cycle: str) -> None:
        # TODO: Implement JAPC SET to UCAP ResetReference property
        log.warning(
            f"onResetReference({cycle}) called but JAPC SET not implemented in display-only mode"
        )

    @QtCore.Slot(str, float)
    def setGain(self, cycle: str, gain: float) -> None:
        # TODO: Implement JAPC SET to UCAP Gain property
        log.warning(
            f"setGain({cycle}, {gain}) called but JAPC SET not implemented in display-only mode"
        )

    # -------------------------------------------------------------------------
    # Subscription handlers
    # -------------------------------------------------------------------------

    async def _handle_cycle_forewarning(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        cycle_data = _extract_cycle_data(response.data)
        self._onCycleForewarning.emit(cycle_data)
        # TODO: Subscribe to UCAP cycle start property (e.g., CycleDataMeasInit or SCY timing)
        # instead of emitting from forewarning as a workaround
        self._onCycleStart.emit(cycle_data)

    async def _handle_cycle_correction(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        cycle_data = _extract_cycle_data(response.data)
        self._onCorrectionCalculated.emit(cycle_data)

        # Emit new reference signal when reference_timestamp changes
        if cycle_data.reference_timestamp is not None:
            self._onNewReference.emit(cycle_data)

    async def _handle_cycle_measured(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        cycle_data = _extract_cycle_data(response.data)
        self._onCycleMeasured.emit(cycle_data)

    async def _handle_metrics(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        metrics = dict(response.data.copy())
        self._onMetricsAvailable.emit(metrics)


def _extract_cycle_data(
    apv: pyda.data.ImmutableLazyMapping,
) -> hystcomp_utils.cycle_data.CycleData:
    return hystcomp_utils.cycle_data.CycleData.from_dict(apv.copy())
