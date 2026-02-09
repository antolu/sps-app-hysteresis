from __future__ import annotations

import datetime
import logging
import sys

import numpy as np
import pyda_japc
from hystcomp_actions import Correction, Inference, Trim
from hystcomp_actions.qt import QtCorrectionAdapter, QtInferenceAdapter, QtTrimAdapter
from hystcomp_event_builder import EventBuilderConfig, SynchronousOrchestrator
from hystcomp_event_builder.qt import QtEventBuilderAdapter, QtMetricsAdapter
from hystcomp_utils.cycle_data import CorrectionMode, CycleData
from op_app_context import context
from qtpy import QtCore

from ..contexts import app_context
from ._pipeline import Pipeline

log = logging.getLogger(__name__)


class StandalonePipeline(Pipeline):
    _resetState = QtCore.Signal()
    _trimApplied = QtCore.Signal(CycleData, datetime.datetime, str) # I don't think this ever emits

    def __init__(
        self,
        provider: pyda_japc.JapcProvider,
        *,
        buffer_size: int = 60000,
        meas_b_avail: bool = True,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        param_names = app_context().PARAMS
        self.meas_b_avail = meas_b_avail

        # 1. Create Configuration
        config = EventBuilderConfig(
            trigger=param_names.TRIGGER,
            cycle_start=param_names.CYCLE_START,
            add_prog_trigger=param_names.ADD_PROG_TRIGGER,
            param_i_prog=param_names.I_PROG,
            param_b_prog=param_names.B_PROG,
            param_b_correction=param_names.B_CORRECTION,
            param_bdot_prog=param_names.BDOT_PROG,
            param_bdot_played=param_names.BDOT_PLAYED,
            param_fulleco_iref=param_names.I_PROG_FULLECO,
            param_dyneco_iref=param_names.I_PROG_DYNECO,
            param_i_meas=param_names.I_MEAS,
            param_b_meas=param_names.B_MEAS,
            param_bdot_meas=param_names.BDOT_MEAS,
            reset_reference_trigger=param_names.RESET_REFERENCE_TRIGGER,
            buffer_size=buffer_size,
            meas_b_avail=meas_b_avail,
        )

        # 2. Instantiate Actions
        # Inference
        self._inference_core = Inference(
            device="cpu"
        )  # Default to cpu or make configurable
        self._inference_core.load_eddy_current_model(
            app_context().EDDY_CURRENT_MODEL.NAME,
            app_context().EDDY_CURRENT_MODEL.VERSION,
        )
        if meas_b_avail:
            self._inference_core.load_measurement_eddy_current_model(
                app_context().MEASUREMENT_EDDY_CURRENT_MODEL.NAME,
                app_context().MEASUREMENT_EDDY_CURRENT_MODEL.VERSION,
            )
        self._predict = QtInferenceAdapter(self._inference_core, parent=parent)

        # Correction
        self._correction_core = Correction(trim_settings=app_context().TRIM_SETTINGS)
        self._correction = QtCorrectionAdapter(self._correction_core, parent=parent)

        trim_type="flat"

        self._trim_core = Trim(
                param_b_corr=param_names.LSA_TRIM_PARAM or "",
                settings=app_context().TRIM_SETTINGS,
                lsa_provider=context.lsa_provider,
                trim_threshold=app_context().TRIM_MIN_THRESHOLD,
                dry_run=False,
        )

        self._trim = QtTrimAdapter(self._trim_core, parent=parent)

        # 3. Instantiate Orchestrator
        self._orchestrator = SynchronousOrchestrator(
            provider=provider,
            config=config,
            inference=self._inference_core,
            correction=self._correction_core,
            trim=self._trim_core,
        )

        # 4. Expose Builders via Qt Adapters
        self._create_cycle = QtEventBuilderAdapter(
            self._orchestrator.create_cycle, parent=parent
        )
        self._add_measurements_pre = QtEventBuilderAdapter(
            self._orchestrator.add_measurements_pre, parent=parent
        )
        self._buffer = QtEventBuilderAdapter(self._orchestrator.buffer, parent=parent)
        self._start_cycle = QtEventBuilderAdapter(
            self._orchestrator.start_cycle, parent=parent
        )
        self._add_programmed = QtEventBuilderAdapter(
            self._orchestrator.add_programmed, parent=parent
        )
        self._add_measurement_post = QtEventBuilderAdapter(
            self._orchestrator.add_measurements_post, parent=parent
        )
        self._track_dyneco = QtEventBuilderAdapter(
            self._orchestrator.track_dyneco, parent=parent
        )
        self._track_reference_changed = QtEventBuilderAdapter(
            self._orchestrator.track_reference_changed, parent=parent
        )
        self._calculate_metrics = QtMetricsAdapter(
            self._orchestrator.calculate_metrics, parent=parent
        )

        if meas_b_avail:
            assert self._orchestrator.add_measurement_ref is not None
            self._add_measurement_ref = QtEventBuilderAdapter(
                self._orchestrator.add_measurement_ref, parent=parent
            )

        # Store current prediction mode
        self._prediction_mode = CorrectionMode.COMBINED

        self._connect_signals()

    def start(self) -> None:
        # Orchestrator handles starting all builders (except actions which are passive/event-driven)
        self._orchestrator.start()

    def stop(self) -> None:
        self._orchestrator.stop()

    @QtCore.Slot(str)
    def onResetReference(self, cycle: str) -> None:
        try:
            if self.meas_b_avail:
                assert self._orchestrator.add_measurement_ref is not None
                self._orchestrator.add_measurement_ref.reset_reference(cycle_name=cycle)
            self._correction_core.reset_reference(cycle_name=cycle)
        except Exception:
            log.exception("Error resetting reference.")

    def _connect_signals(self) -> None:
        # Additional signals not handled by orchestrator (mostly UI feedback)
        self._trim.trimApplied.connect(self._trimApplied.emit)
        self._trim.flatteningApplied.connect(self._trimApplied.emit)
        self._trim.flatteningApplied.connect(self._on_flattening_applied)

        # Connect reset signal for inference state
        self._resetState.connect(self._inference_core.reset_state)

        # Set initial prediction mode for correction system
        self._correction_core.set_prediction_mode(self._prediction_mode)

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
        return self._predict.modelLoaded

    @property
    def resetState(self) -> QtCore.Signal:
        return self._resetState

    @property
    def onCycleForewarning(self) -> QtCore.Signal:
        # Qt adapter exposes camelCase signal
        return self._create_cycle.cycleDataAvailable

    @property
    def onCycleStart(self) -> QtCore.Signal:
        return self._start_cycle.cycleDataAvailable

    @property
    def onCyclePredictionCompleted(self) -> QtCore.Signal:
        return self._predict.cycleDataAvailable

    @property
    def onCycleCorrectionCalculated(self) -> QtCore.Signal:
        return self._correction.cycleDataAvailable

    @property
    def onTrimApplied(self) -> QtCore.Signal:
        return self._trimApplied

    @property
    def onCycleMeasured(self) -> QtCore.Signal:
        assert self._add_measurement_ref is not None
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

    def set_prediction_mode(self, mode: CorrectionMode) -> None:
        """Set the prediction mode for both inference and correction systems."""
        self._prediction_mode = mode
        self._inference_core.set_prediction_mode(mode)
        self._correction_core.set_prediction_mode(mode)
