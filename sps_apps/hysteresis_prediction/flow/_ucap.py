from __future__ import annotations

import logging


from qtpy import QtCore
import pyda_japc
import pyda._metadata
import pyda.providers
import asyncio
import pyda.data
import pyda.clients
import re
import hystcomp_utils.cycle_data

from ..data import StartCycleEventBuilder

from ._data_flow import DataFlow, FlowWorker

log = logging.getLogger(__name__)
ENDPOINT_RE = re.compile(r"^(?P<device>(?P<protocol>.+:\/\/)?[\w\/\.-]+)/(?P<property>[\w\#\.-]+)$" )  # noqa: E501


CYCLE_WARNING = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataFCY"
CYCLE_CORRECTION = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataCorrection"
CYCLE_MEASURED = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/CycleDataMeasRef"
RESET_REFERENCE = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/ResetReference"
SET_GAIN = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/SPS.HYSTCOMP.MBI.ECO/Gain"


class UcapFlowWorker(FlowWorker):
    def __init__(self, provider: pyda_japc.JapcProvider, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)

        self._provider = provider

    def _init_data_flow_impl(self) -> None:
        self._data_flow = UcapDataFlow(provider=self._provider)


class UcapDataFlow(DataFlow, QtCore.QObject):
    _onCycleForewarning = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCorrectionCalculated = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)
    _onCycleMeasured = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)

    def __init__(self, provider: pyda_japc.JapcProvider, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent=parent)
        self._provider = provider

        self._da = pyda.AsyncIOClient(
            provider=pyda.providers.Provider(
                data_source=self._provider,
                metadata_source=pyda._metadata.NoMetadataSource(),
            )
        )

        self._start_cycle_event_builder = StartCycleEventBuilder(
            provider=self._provider,
        )

        self._should_stop = False
        self._handles = self._setup_subscriptions()

    def _setup_subscriptions(self) -> list[pyda.clients.asyncio.AsyncIOSubscription]:
        return [
            self._da.subscribe(CYCLE_WARNING, context="SPS.USER.ALL"),
            self._da.subscribe(CYCLE_CORRECTION, context="SPS.USER.ALL"),
            self._da.subscribe(CYCLE_MEASURED, context="SPS.USER.ALL"),
        ]

    def start(self) -> None:
        self._start_cycle_event_builder.start()

        asyncio.create_task(self._start())

    async def _start(self) -> None:
        async for sub in pyda.AsyncIOClient.merge_subscriptions(*self._handles):
            if self._should_stop:
                break

            response = await sub

            if response.exception is not None:
                log.error(f"Error in subscription: {response.exception}")
                continue

            if str(response.query.endpoint) == CYCLE_WARNING:
                await self.handle_cycle_forewarning(response)
            elif str(response.query.endpoint) == CYCLE_CORRECTION:
                await self.handle_cycle_correction_calculated(response)
            elif str(response.query.endpoint) == CYCLE_MEASURED:
                await self.handle_cycle_measured(response)
            else:
                log.warning(f"Received unknown endpoint: {response.query.endpoint}")

    def stop(self) -> None:
        self._should_stop = True

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

    async def handle_cycle_forewarning(self, response: pyda.data.PropertyRetrievalResponse[pyda.data.PropertyAccessQuery]) -> None:
        cycle_data = extract_cycle_data(response.value)
        self._onCycleForewarning.emit(cycle_data)

    async def handle_cycle_correction_calculated(self, response: pyda.data.PropertyRetrievalResponse[pyda.data.PropertyAccessQuery]) -> None:
        cycle_data = extract_cycle_data(response.value)
        self._onCorrectionCalculated.emit(cycle_data)

    async def handle_cycle_measured(self, response: pyda.data.PropertyRetrievalResponse[pyda.data.PropertyAccessQuery]) -> None:
        cycle_data = extract_cycle_data(response.value)
        self._onCycleMeasured.emit(cycle_data)

    async def handle_set_gain(self, response: pyda.data.PropertyUpdateResponse) -> None:
        pass



def extract_cycle_data(apv: pyda.data.AcquiredPropertyData) -> hystcomp_utils.cycle_data.CycleData:

    return hystcomp_utils.cycle_data.CycleData.from_dict(
        apv.mutable_copy()
    )