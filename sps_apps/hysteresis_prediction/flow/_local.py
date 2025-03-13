from __future__ import annotations

import datetime
import logging

import numpy as np
import pyda_japc
from hystcomp_utils.cycle_data import CycleData
from PyQt5.QtCore import QObject
from qtpy import QtCore

from ..contexts import app_context
from ..local import CalculateCorrection, Inference, LocalTrim
from ..local.event_building import (
    AddMeasurementReferencesEventBuilder,
    AddMeasurementsEventBuilder,
    AddProgrammedEventBuilder,
    BufferEventbuilder,
    CreateCycleEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    StartCycleEventBuilder,
    TrackDynEcoEventBuilder,
    TrackFullEcoEventBuilder,
    TrackReferenceChangedEventBuilder,
)
from ..local.track_precycle import TrackPrecycleEventBuilder
from ._data_flow import DataFlow, FlowWorker

log = logging.getLogger(__name__)


class LocalFlowWorker(FlowWorker):
    def __init__(
        self,
        provider: pyda_japc.JapcProvider,
        *,
        buffer_size: int = 60000,
        meas_b_avail: bool = True,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._data_flow: LocalDataFlow | None = None

        self._provider = provider
        self._buffer_size = buffer_size
        self._meas_b_avail = meas_b_avail

    def _init_data_flow_impl(self) -> None:
        self._data_flow = LocalDataFlow(
            meas_b_avail=self._meas_b_avail,
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


class LocalDataFlow(DataFlow, QtCore.QObject):
    _resetState = QtCore.Signal()
    _trimApplied = QtCore.Signal(CycleData, np.ndarray, datetime.datetime, str)

    def __init__(
        self,
        provider: pyda_japc.JapcProvider,
        *,
        buffer_size: int = 60000,
        meas_b_avail: bool = True,
        parent: QtCore.QObject | None = None,
    ) -> None:
        QObject.__init__(self, parent=parent)
        param_names = app_context().PARAMS

        self.meas_b_avail = meas_b_avail

        self._create_cycle = CreateCycleEventBuilder(
            cycle_warning=param_names.TRIGGER,
            param_b_prog=param_names.B_PROG,
            param_i_prog=param_names.I_PROG,
            param_b_correction=param_names.B_CORRECTION,
            provider=provider,
            parent=parent,
        )
        self._add_measurements_pre = AddMeasurementsEventBuilder(
            param_i_meas=param_names.I_MEAS,
            param_b_meas=param_names.B_MEAS,
            provider=provider,
            parent=parent,
        )
        self._buffer = BufferEventbuilder(buffer_size=buffer_size, parent=parent)
        self._predict = Inference(parent=parent)
        self._correction = CalculateCorrection(
            trim_settings=app_context().TRIM_SETTINGS, parent=parent
        )
        self._start_cycle = StartCycleEventBuilder(
            trigger=param_names.CYCLE_START, provider=provider, parent=parent
        )
        self._add_programmed = AddProgrammedEventBuilder(
            param_i_prog=param_names.I_PROG,
            param_b_prog=param_names.B_PROG,
            trigger=param_names.ADD_PROG_TRIGGER,
            provider=provider,
            parent=parent,
        )
        self._add_measurement_post = CycleStampedAddMeasurementsEventBuilder(
            param_i_meas=param_names.I_MEAS,
            param_b_meas=param_names.B_MEAS,
            provider=provider,
            parent=parent,
        )
        if meas_b_avail:
            self._add_measurement_ref = AddMeasurementReferencesEventBuilder(
                parent=parent
            )

        self._track_dyneco = TrackDynEcoEventBuilder(
            param_dyneco_iref=param_names.I_PROG_DYNECO,
            provider=provider,
            parent=parent,
        )
        self._track_fulleco = TrackFullEcoEventBuilder(
            param_fulleco_iref=param_names.I_PROG_FULLECO,
            param_fulleco_trigger=param_names.FULLECO_TRIGGER,
            provider=provider,
            parent=parent,
        )
        self._track_precycle = TrackPrecycleEventBuilder(
            precycle_sequence=["SPS.USER.LHCPILOT", "SPS.USER.MD1"], parent=parent
        )
        self._track_reference_changed = TrackReferenceChangedEventBuilder(
            param_trigger=param_names.RESET_REFERENCE_TRIGGER,
            param_start_cycle=param_names.CYCLE_START,
            provider=provider,
            parent=parent,
        )

        self._trim = LocalTrim(
            param_b_corr="SPSBEAM/BHYS",
            settings=app_context().TRIM_SETTINGS,
            parent=parent,
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
        if self.meas_b_avail:
            assert hasattr(self, "_add_measurement_ref")
            assert self._add_measurement_ref is not None
            self._add_measurement_ref.start()
        self._track_dyneco.start()
        self._track_fulleco.start()
        self._track_precycle.start()

    def stop(self) -> None:
        self._create_cycle.stop()
        self._add_measurements_pre.stop()
        self._buffer.stop()
        self._predict.stop()
        self._correction.stop()
        self._start_cycle.stop()
        self._add_programmed.stop()
        self._add_measurement_post.stop()
        if self.meas_b_avail:
            assert hasattr(self, "_add_measurement_ref")
            assert self._add_measurement_ref is not None
            self._add_measurement_ref.stop()
        self._track_dyneco.stop()
        self._track_fulleco.stop()
        self._track_precycle.stop()

    @QtCore.Slot(str)
    def onResetReference(self, cycle: str) -> None:
        try:
            if self.meas_b_avail:
                assert hasattr(self, "_add_measurement_ref")
                assert self._add_measurement_ref is not None
                self._add_measurement_ref.resetReference(cycle_name=cycle)
            self._correction.resetReference(cycle_name=cycle)
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
        if self.meas_b_avail:
            assert hasattr(self, "_add_measurement_ref")
            assert self._add_measurement_ref is not None
            self._add_measurement_post.cycleDataAvailable.connect(
                self._add_measurement_ref.onNewCycleData
            )
            self._add_measurement_ref.cycleDataAvailable.connect(
                self._buffer.onNewMeasCycleData
            )
        else:
            self._add_measurement_post.cycleDataAvailable.connect(
                self._buffer.onNewMeasCycleData
            )

        self._correction.cycleDataAvailable.connect(self._track_dyneco.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._track_fulleco.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._trim.onNewPrediction)
        self._trim.trimApplied.connect(self._trimApplied.emit)
        self._track_dyneco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)
        self._track_fulleco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)
        self._track_reference_changed.resetReference.connect(self.onResetReference)

        self._resetState.connect(self._predict.reset_state)

    @property
    def onModelLoaded(self) -> QtCore.Signal:
        return self._predict.model_loaded

    @property
    def resetState(self) -> QtCore.Signal:
        return self._resetState

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
    def onTrimApplied(self) -> QtCore.Signal:
        return self._trimApplied

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        return self._add_measurement_ref.cycleDataAvailable

    @property
    def onNewReference(self) -> QtCore.Signal:
        return self._correction.newReference

    @QtCore.Slot(str, float)
    def setGain(self, cycle: str, gain: float) -> None:
        app_context().TRIM_SETTINGS.gain[cycle] = gain
