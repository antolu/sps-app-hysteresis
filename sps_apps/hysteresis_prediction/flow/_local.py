from __future__ import annotations

import logging
import typing

import pyda_japc
from qtpy import QtCore

from ..data import (
    AddMeasurementReferencesEventBuilder,
    AddMeasurementsEventBuilder,
    AddProgrammedEventBuilder,
    BufferEventbuilder,
    CreateCycleEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    StartCycleEventBuilder,
    TrackDynEcoEventBuilder,
    TrackFullEcoEventBuilder,
)
from ..signals import TrackPrecycleEventBuilder
from ..inference import CalculateCorrection, Inference
from ._data_flow import DataFlow, FlowWorker


log = logging.getLogger(__name__)



class LocalFlowWorker(FlowWorker):
    def __init__(
            self,
            provider: pyda_japc.JapcProvider,
            buffer_size: int = 60000,
            parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._data_flow: LocalDataFlow | None = None

        self._provider = provider
        self._buffer_size = buffer_size

    def _init_data_flow_impl(self) -> None:
        self._data_flow = LocalDataFlow(
            provider=self._provider,
            buffer_size=self._buffer_size,
            parent=self.parent(),
        )

    @property
    def data_flow(self) -> LocalDataFlow:
        if self._data_flow is None:
            msg = "Data flow not initialized. Call init_data_flow() first."
            raise ValueError(msg)
        return self._data_flow


class LocalDataFlow(DataFlow):
    def __init__(
            self,
            provider: pyda_japc.JapcProvider,
            buffer_size: int = 60000,
            parent: QtCore.QObject | None = None,
    ) -> None:
        self._create_cycle = CreateCycleEventBuilder(provider=provider, parent=parent)
        self._add_measurements_pre = AddMeasurementsEventBuilder(
            provider=provider, parent=parent
        )
        self._buffer = BufferEventbuilder(buffer_size=buffer_size, parent=parent)
        self._predict = Inference(parent=parent)
        self._correction = CalculateCorrection(parent=parent)
        self._start_cycle = StartCycleEventBuilder(provider=provider, parent=parent)
        self._add_programmed = AddProgrammedEventBuilder(
            provider=provider, parent=parent
        )
        self._add_measurement_post = CycleStampedAddMeasurementsEventBuilder(
            provider=provider, parent=parent
        )
        self._add_measurement_ref = AddMeasurementReferencesEventBuilder(parent=parent)

        self._track_dyneco = TrackDynEcoEventBuilder(provider=provider, parent=parent)
        self._track_fulleco = TrackFullEcoEventBuilder(provider=provider, parent=parent)
        self._track_precycle = TrackPrecycleEventBuilder(
            precycle_sequence=["SPS.USER.LHCPILOT", "SPS.USER.MD1"], parent=parent
        )

        self._connect_signals()

    def start(self) -> None:
        self._create_cycle.start()
        self._add_measurements_pre.start()
        self._buffer.start()
        self._predict.start()
        self._correction.start()
        self._start_cycle.start()
        self._add_programmed.start()
        self._add_measurement_post.start()
        self._add_measurement_ref.start()
        self._track_dyneco.start()
        self._track_fulleco.start()

    def stop(self) -> None:
        self._create_cycle.stop()
        self._add_measurements_pre.stop()
        self._buffer.stop()
        self._predict.stop()
        self._correction.stop()
        self._start_cycle.stop()
        self._add_programmed.stop()
        self._add_measurement_post.stop()
        self._add_measurement_ref.stop()
        self._track_dyneco.stop()
        self._track_fulleco.stop()

    def resetReference(self) -> None:
        try:
            self._add_measurement_ref.resetReference()
            self._correction.resetReference()
        except Exception:
            log.exception("Error resetting reference.")

    def _connect_signals(self) -> None:
        self._create_cycle.cycleDataAvailable.connect(
            self._add_measurements_pre.onNewCycleData
        )
        self._add_measurements_pre.cycleDataAvailable.connect(
            self._track_precycle.onNewCycleData
        )
        self._track_precycle.cycleDataAvailable.connect(self._buffer.onNewCycleData)
        self._add_measurements_pre.cycleDataAvailable.connect(
            self._start_cycle.onNewCycleData
        )
        self._buffer.newBufferAvailable.connect(self._predict.onNewCycleDataBuffer)
        self._buffer.newEcoBufferAvailable.connect(self._predict.onNewCycleDataBuffer)
        self._predict.cycleDataAvailable.connect(self._correction.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._add_programmed.onNewCycleData)

        self._add_programmed.cycleDataAvailable.connect(self._buffer.onNewProgCycleData)
        self._add_programmed.cycleDataAvailable.connect(
            self._add_measurement_post.onNewCycleData
        )
        self._add_measurement_post.cycleDataAvailable.connect(
            self._add_measurement_ref.onNewCycleData
        )

        self._add_measurement_post.cycleDataAvailable.connect(
            self._buffer.onNewMeasCycleData
        )

        self._correction.cycleDataAvailable.connect(self._track_dyneco.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._track_fulleco.onNewCycleData)
        self._track_dyneco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)
        self._track_fulleco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        return self._create_cycle.cycleDataAvailable

    @property
    def onCycleStart(self) -> QtCore.Signal:
        return self._start_cycle.cycleDataAvailable

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        return self._correction.cycleDataAvailable

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        return self._add_measurement_ref.cycleDataAvailable

    def setGain(self, selector: str, gain: float) -> None:
        self._correction.setGain(selector, gain)

