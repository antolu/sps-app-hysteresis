from __future__ import annotations

import asyncio
import logging
import re

import hystcomp_utils.cycle_data
import pyda._metadata
import pyda.access
import pyda.clients
import pyda.data
import pyda.metadata
import pyda.providers
import pyda_japc
from qtpy import QtCore

from ..contexts import UcapParameterNames
from ..data import JapcEndpoint, StartCycleEventBuilder
from ._data_flow import DataFlow, FlowWorker

log = logging.getLogger(__name__)
ENDPOINT_RE = re.compile(
    r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$"
)


CYCLE_WARNING = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataFCY"
CYCLE_CORRECTION = (
    "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataCorrection"
)
CYCLE_MEASURED = (
    "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataMeasRef"
)
RESET_REFERENCE = (
    "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/ResetReference"
)
SET_GAIN = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/Gain"


class UcapFlowWorker(FlowWorker):
    def __init__(
        self,
        ucap_params: UcapParameterNames,
        *,
        provider: pyda_japc.JapcProvider,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._ucap_params = ucap_params
        self._provider = provider

    def _init_data_flow_impl(self) -> None:
        self._data_flow = UcapDataFlow(
            ucap_params=self._ucap_params,
            provider=self._provider,
        )


class UcapDataFlow(DataFlow, QtCore.QObject):
    _onCycleForewarning = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCorrectionCalculated = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCycleMeasured = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)

    _dummy = QtCore.Signal()

    def __init__(
        self,
        ucap_params: UcapParameterNames,
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

        self._start_cycle_event_builder = StartCycleEventBuilder(
            provider=self._provider,
        )

        self._ucap_params = ucap_params

        self._should_stop = False
        self._handles: list[pyda.clients.asyncio.AsyncIOSubscription] = []

    def _setup_subscriptions(self) -> list[pyda.clients.asyncio.AsyncIOSubscription]:
        matches = [
            ENDPOINT_RE.match(self._ucap_params.CYCLE_WARNING),
            ENDPOINT_RE.match(self._ucap_params.CYCLE_CORRECTION),
            ENDPOINT_RE.match(self._ucap_params.CYCLE_MEASURED),
        ]
        return [
            self._da.subscribe(
                endpoint=JapcEndpoint.from_str(endpoint),
                context="SPS.USER.ALL",
            )
            for match, endpoint in zip(
                matches,
                [
                    self._ucap_params.CYCLE_WARNING,
                    self._ucap_params.CYCLE_CORRECTION,
                    self._ucap_params.CYCLE_MEASURED,
                ],
                strict=False,
            )
            if match is not None
        ]

    def start(self) -> None:
        self._start_cycle_event_builder.start()

        asyncio.run(self._start())

    async def _start(self) -> None:
        self._handles = self._setup_subscriptions()

        async for response in pyda.AsyncIOClient.merge_subscriptions(*self._handles):
            if self._should_stop:
                break

            if response.exception is not None:
                log.error(f"Error in subscription: {response.exception}")
                continue

            if str(response.query.endpoint) == self._ucap_params.CYCLE_WARNING:
                await self.handle_cycle_forewarning(response)
            elif str(response.query.endpoint) == self._ucap_params.CYCLE_CORRECTION:
                await self.handle_cycle_correction_calculated(response)
            elif str(response.query.endpoint) == self._ucap_params.CYCLE_MEASURED:
                await self.handle_cycle_measured(response)
            else:
                log.warning(f"Received unknown endpoint: {response.query.endpoint}")

    def stop(self) -> None:
        self._should_stop = True

    @property
    def onModelLoaded(self) -> QtCore.Signal:
        return self._dummy

    @property
    def resetState(self) -> QtCore.Signal:
        return self._dummy

    @property
    def onCycleStart(self) -> QtCore.Signal:
        return self._start_cycle_event_builder.cycleDataAvailable

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        return self._onCycleForewarning

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        return self._onCorrectionCalculated

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        return self._onCycleMeasured

    async def handle_cycle_forewarning(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        cycle_data = extract_cycle_data(response.data)
        self._onCycleForewarning.emit(cycle_data)

    async def handle_cycle_correction_calculated(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        cycle_data = extract_cycle_data(response.data)
        self._onCorrectionCalculated.emit(cycle_data)

    async def handle_cycle_measured(
        self,
        response: pyda.access.PropertyRetrievalResponse[
            pyda.access.PropertyAccessQuery
        ],
    ) -> None:
        cycle_data = extract_cycle_data(response.data)
        self._onCycleMeasured.emit(cycle_data)

    async def handle_set_gain(
        self, response: pyda.access.PropertyUpdateResponse
    ) -> None:
        pass


def extract_cycle_data(
    apv: pyda.data.ImmutableLazyMapping,
) -> hystcomp_utils.cycle_data.CycleData:
    return hystcomp_utils.cycle_data.CycleData.from_dict(apv.copy())
