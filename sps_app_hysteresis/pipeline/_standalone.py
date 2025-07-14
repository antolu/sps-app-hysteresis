from __future__ import annotations

import datetime
import logging

import numpy as np
import pyda_japc
from hystcomp_utils.cycle_data import CycleData
from PyQt6.QtCore import QObject
from qtpy import QtCore

from ..contexts import app_context
from ..standalone import CalculateCorrection, Inference, StandaloneTrim
from ..standalone._inference import PredictionMode
from ..standalone.event_building import (
    AddMeasurementReferencesEventBuilder,
    AddMeasurementsEventBuilder,
    AddProgrammedEventBuilder,
    BufferEventbuilder,
    CalculateMetricsConverter,
    CreateCycleEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    StartCycleEventBuilder,
    TrackDynEcoEventBuilder,
    TrackReferenceChangedEventBuilder,
)
from ..standalone.track_precycle import TrackPrecycleEventBuilder
from ._pipeline import Pipeline

log = logging.getLogger(__name__)


class StandalonePipeline(Pipeline, QtCore.QObject):
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
            param_fulleco_iref=param_names.I_PROG_FULLECO,
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
        self._predict.load_eddy_current_model(
            app_context().EDDY_CURRENT_MODEL.NAME,
            app_context().EDDY_CURRENT_MODEL.VERSION,
        )
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
        self._track_precycle = TrackPrecycleEventBuilder(
            precycle_sequence=["SPS.USER.LHCPILOT", "SPS.USER.MD1"], parent=parent
        )
        self._track_reference_changed = TrackReferenceChangedEventBuilder(
            param_trigger=param_names.RESET_REFERENCE_TRIGGER,
            param_start_cycle=param_names.CYCLE_START,
            provider=provider,
            parent=parent,
        )
        self._calculate_metrics = CalculateMetricsConverter(parent=parent)

        self._trim = StandaloneTrim(
            param_b_corr="SPSBEAM/BHYS",
            settings=app_context().TRIM_SETTINGS,
            parent=parent,
        )

        # Store current prediction mode
        self._prediction_mode = PredictionMode.COMBINED

        self._connect_signals()

    def start(self) -> None:
        for builder in (
            self._create_cycle,
            self._add_measurements_pre,
            self._buffer,
            self._predict,
            self._correction,
            self._start_cycle,
            self._add_programmed,
            self._add_measurement_post,
            self._track_dyneco,
            self._track_precycle,
            self._track_reference_changed,
        ):
            log.info(f"Starting {builder}")
            builder.start()

        if self.meas_b_avail:
            assert hasattr(self, "_add_measurement_ref")
            assert self._add_measurement_ref is not None
            self._add_measurement_ref.start()
            self._calculate_metrics.start()

    def stop(self) -> None:
        for builder in (
            self._create_cycle,
            self._add_measurements_pre,
            self._buffer,
            self._predict,
            self._correction,
            self._start_cycle,
            self._add_programmed,
            self._add_measurement_post,
            self._track_dyneco,
            self._track_precycle,
            self._track_reference_changed,
        ):
            log.info(f"Stopping {builder}")
            builder.stop()

        if self.meas_b_avail:
            assert hasattr(self, "_add_measurement_ref")
            assert self._add_measurement_ref is not None
            self._add_measurement_ref.stop()
            self._calculate_metrics.stop()

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
            self._add_measurement_ref.cycleDataAvailable.connect(
                self._calculate_metrics.onNewCycleData
            )
        else:
            self._add_measurement_post.cycleDataAvailable.connect(
                self._buffer.onNewMeasCycleData
            )

        self._correction.cycleDataAvailable.connect(self._track_dyneco.onNewCycleData)
        self._correction.cycleDataAvailable.connect(self._trim.onNewPrediction)
        self._trim.trimApplied.connect(self._trimApplied.emit)
        self._trim.flatteningApplied.connect(self._trimApplied.emit)
        self._trim.flatteningApplied.connect(self._on_flattening_applied)
        self._track_dyneco.cycleDataAvailable.connect(self._buffer.onNewEcoCycleData)
        # self._track_reference_changed.resetReference.connect(self.onResetReference)

        self._resetState.connect(self._predict.reset_state)

        # Set initial prediction mode for correction system
        self._correction.set_prediction_mode(self._prediction_mode)

        # Set correction system reference for trim (for any remaining reference updates)
        self._trim.set_correction_system(self._correction)

    @QtCore.Slot(CycleData, np.ndarray, datetime.datetime, str)
    def _on_flattening_applied(
        self,
        cycle_data: CycleData,
        delta: np.ndarray,
        trim_time: datetime.datetime,
        comment: str,
    ) -> None:
        """Handle flattening correction applied - reset reference for the cycle."""
        log.info(
            f"Flattening correction applied for cycle {cycle_data.cycle}. Resetting reference."
        )
        self.onResetReference(cycle_data.cycle)

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

    @property
    def onMetricsAvailable(self) -> QtCore.Signal:
        return self._calculate_metrics.newMetricsAvailable

    @QtCore.Slot(str, float)
    def setGain(self, cycle: str, gain: float) -> None:
        app_context().TRIM_SETTINGS.gain[cycle] = gain

    def set_prediction_mode(self, mode: PredictionMode) -> None:
        """Set the prediction mode for both inference and correction systems."""
        self._prediction_mode = mode
        self._predict.set_prediction_mode(mode)
        self._correction.set_prediction_mode(mode)
