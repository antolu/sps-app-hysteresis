from __future__ import annotations

import datetime
import logging
import typing

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from accwidgets import graph as accgraph
from accwidgets.lsa_selector import (
    AbstractLsaSelectorContext,
    LsaSelector,
    LsaSelectorAccelerator,
    LsaSelectorModel,
)
from hystcomp_actions.utils import cycle_metadata
from hystcomp_utils.cycle_data import CycleData
from op_app_context import context
from qtpy import QtCore, QtWidgets

from ...utils import mute_signals
from .._widgets import ToggleButton
from ._model import TrimModel

log = logging.getLogger(__package__)


class TrimInfoWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.LsaServerLabel = QtWidgets.QLabel("LSA Server", parent=self)
        self.LsaServerLineValue = QtWidgets.QLabel(context.lsa_server, parent=self)

        self.LastTrimLabel = QtWidgets.QLabel("Last Trim", parent=self)
        self.LastTrimLineValue = QtWidgets.QLabel("N/A", parent=self)

        self.LastCommentLabel = QtWidgets.QLabel("Last Comment", parent=self)
        self.LastCommentLineValue = QtWidgets.QLabel("N/A", parent=self)
        self.LastCommentLineValue.setWordWrap(True)

        self.BeamInLabel = QtWidgets.QLabel("Beam In", parent=self)
        self.BeamInLineValue = QtWidgets.QLabel("N/A", parent=self)
        self.BeamInLineValue.setMinimumWidth(120)

        grid_layout = QtWidgets.QGridLayout(self)
        self.setLayout(grid_layout)
        grid_layout.addWidget(self.LsaServerLabel, 0, 0)
        grid_layout.addWidget(self.LsaServerLineValue, 0, 1)
        grid_layout.addWidget(self.LastTrimLabel, 1, 0)
        grid_layout.addWidget(self.LastTrimLineValue, 1, 1)
        grid_layout.addWidget(self.LastCommentLabel, 2, 0)
        grid_layout.addWidget(self.LastCommentLineValue, 2, 1)
        grid_layout.addWidget(self.BeamInLabel, 3, 0)
        grid_layout.addWidget(self.BeamInLineValue, 3, 1)

    @QtCore.Slot(CycleData, np.ndarray, datetime.datetime, str)
    def on_trim_applied(
        self,
        cycle_data: CycleData,
        _: np.ndarray,
        trim_time: datetime.datetime,
        trim_comment: str,
    ) -> None:
        self.LastTrimLineValue.setText(trim_time.strftime("%Y%m%d-%H:%M:%S:%f")[:-4])
        self.LastCommentLineValue.setText(trim_comment)

    @QtCore.Slot(str)
    def onContextChanged(self, cycle: str) -> None:
        beam_in = cycle_metadata.beam_in(cycle)
        beam_out = cycle_metadata.beam_out(cycle)

        self.BeamInLineValue.setText(f"C{beam_in} - C{beam_out}")


class TrimSettingsWidget(QtWidgets.QWidget):
    flatteningRequested = QtCore.Signal(str, float)  # cycle, constant_field
    testTrim = QtCore.Signal(str, float)  # cycle, constant_field

    def __init__(self, model: TrimModel, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.model = model

        self.GainLabel = QtWidgets.QLabel("Gain", parent=self)
        self.GainSpinBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.GainSpinBox.setRange(0.0, 5.0)
        self.GainSpinBox.setSingleStep(0.1)
        self.GainSpinBox.setValue(1.0)
        self.GainSpinBox.setMaximumWidth(50)

        self.TrimTMinLabel = QtWidgets.QLabel("Trim T Min", parent=self)
        self.TrimTMinSpinBox = QtWidgets.QSpinBox(parent=self)
        self.TrimTMinSpinBox.setEnabled(False)
        self.TrimTMinSpinBox.setMaximumWidth(80)

        self.TrimTMaxLabel = QtWidgets.QLabel("Trim T Max", parent=self)
        self.TrimTMaxSpinBox = QtWidgets.QSpinBox(parent=self)
        self.TrimTMaxSpinBox.setEnabled(False)
        self.TrimTMaxSpinBox.setMaximumWidth(80)

        self.ToggleButton = ToggleButton(parent=self)
        self.ToggleButton.initializeState(
            label_s2="Enable Trim",
            label_s1="Disable Trim",
            initial_state=ToggleButton.State.STATE2,
        )

        self.testTrimButton = QtWidgets.QPushButton(
            "Test Trim", parent=self
        )

        self.ResetReferenceButton = QtWidgets.QPushButton(
            "Reset Reference", parent=self
        )

        self.FlatteningButton = QtWidgets.QPushButton("Apply Flattening", parent=self)
        self.FlatteningButton.setEnabled(False)

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.testTrimButton, 0, 0)
        layout.addWidget(self.GainLabel, 1, 0)
        layout.addWidget(self.GainSpinBox, 1, 1)
        layout.addWidget(self.TrimTMinLabel, 2, 0)
        layout.addWidget(self.TrimTMinSpinBox, 2, 1)
        layout.addWidget(self.TrimTMaxLabel, 3, 0)
        layout.addWidget(self.TrimTMaxSpinBox, 4, 1)
        layout.addWidget(self.ToggleButton, 4, 0, 1, 2)
        layout.addWidget(self.ResetReferenceButton, 5, 0, 1, 2)
        layout.addWidget(self.FlatteningButton, 6, 0, 1, 2)
        self.setLayout(layout)

        self.GainSpinBox.valueChanged.connect(self.onGainChanged)
        self.TrimTMinSpinBox.valueChanged.connect(self.onMinValueChanged)
        self.TrimTMaxSpinBox.valueChanged.connect(self.onMaxValueChanged)
        self.ToggleButton.stateChanged.connect(self.onEnableTrim)
        self.FlatteningButton.clicked.connect(self.onFlatteningRequested)
        self.testTrimButton.clicked.connect(self.onTestTrimClicked)

        # disable the trim settings until a context has been set
        self.GainSpinBox.setEnabled(False)
        self.TrimTMinSpinBox.setEnabled(False)
        self.TrimTMaxSpinBox.setEnabled(False)
        self.ToggleButton.setEnabled(False)
        self.ResetReferenceButton.setEnabled(False)
        self.FlatteningButton.setEnabled(False)
        self.testTrimButton.setEnabled(False)

        self.model.contextChanged.connect(self.onContextChanged)
        self._cycle: str | None = None
        self._is_eddy_current_only: bool = False

    @QtCore.Slot(str)
    def onContextChanged(self, cycle: str) -> None:
        self.GainSpinBox.setEnabled(True)
        self.TrimTMinSpinBox.setEnabled(True)
        self.TrimTMaxSpinBox.setEnabled(True)
        self.ToggleButton.setEnabled(True)
        self.ResetReferenceButton.setEnabled(True)
        self.testTrimButton.setEnabled(True)
        self._update_flattening_button_state()

        with mute_signals(self.GainSpinBox, self.TrimTMinSpinBox, self.TrimTMaxSpinBox):
            self.TrimTMinSpinBox.setRange(
                cycle_metadata.beam_in(cycle), cycle_metadata.beam_out(cycle) - 1
            )
            self.TrimTMaxSpinBox.setRange(
                cycle_metadata.beam_in(cycle) + 1, cycle_metadata.beam_out(cycle)
            )

            self.GainSpinBox.setValue(self.model.settings.gain[cycle])
            self.TrimTMinSpinBox.setValue(self.model.settings.trim_start[cycle])
            self.TrimTMaxSpinBox.setValue(self.model.settings.trim_end[cycle])

        if self.model.settings.trim_enabled[cycle]:
            self.ToggleButton.setState(ToggleButton.State.STATE1)
        else:
            self.ToggleButton.setState(ToggleButton.State.STATE2)

        self._cycle = cycle

    @QtCore.Slot(float)
    def onGainChanged(self, value: float) -> None:
        if value <= 1.0:
            self.GainSpinBox.setSingleStep(0.02)
        else:
            self.GainSpinBox.setSingleStep(0.2)

        self.model.settings.gain[self.model.cycle] = value

    @QtCore.Slot(int)
    def onMinValueChanged(self, value: int) -> None:
        # ensure that the min value is less than the max value
        self.TrimTMaxSpinBox.setMinimum(value + 1)

        if value > self.TrimTMaxSpinBox.value():
            self.TrimTMaxSpinBox.setValue(value)

        self.model.settings.trim_start[self.model.cycle] = value

    @QtCore.Slot(int)
    def onMaxValueChanged(self, value: int) -> None:
        # ensure that the max value is greater than the min value
        self.TrimTMinSpinBox.setMaximum(value - 1)

        if value < self.TrimTMinSpinBox.value():
            self.TrimTMinSpinBox.setValue(value)

        self.model.settings.trim_end[self.model.cycle] = value

    @QtCore.Slot(ToggleButton.State)
    def onEnableTrim(self, state: ToggleButton.State) -> None:
        log.debug(
            f"[{self.model.cycle}] Trim enabled: {state == ToggleButton.State.STATE1}"
        )
        self.model.settings.trim_enabled[self.model.cycle] = (
            state == ToggleButton.State.STATE1
        )

    @QtCore.Slot(bool)
    def onCorrectionModeChanged(self, is_eddy_current_only: bool) -> None:  # noqa: FBT001
        """Update prediction mode and enable/disable flattening button accordingly."""
        self._is_eddy_current_only = is_eddy_current_only
        self._update_flattening_button_state()

    def _update_flattening_button_state(self) -> None:
        """Update flattening button enabled state based on prediction mode."""
        # Only enable flattening button when in eddy current only mode and context is set
        enabled = self._is_eddy_current_only and self._cycle is not None
        self.FlatteningButton.setEnabled(enabled)

    @QtCore.Slot()
    def onFlatteningRequested(self) -> None:
        """Handle flattening button click."""
        if self._cycle is None:
            log.error("No cycle selected for flattening correction.")
            return

        # Use constant field of 0.0 as requested
        constant_field = 0.0
        log.info(
            f"Requesting flattening correction for cycle {self._cycle} with constant field {constant_field}"
        )
        self.flatteningRequested.emit(self._cycle, constant_field)

    @QtCore.Slot()
    def onTestTrimClicked(self) -> None:
        """Handle test button click."""
        if self._cycle is None:
            log.error("No cycle selected for flattening correction.")
            return

        # Use constant field of 0.0 as requested
        constant_field = 0.0
        log.info(
            f"Requesting flattening correction for cycle {self._cycle} with constant field {constant_field}"
        )
        self.flatteningRequested.emit(self._cycle, constant_field)


class TrimWidgetView(QtWidgets.QWidget):
    _thread: QtCore.QThread | None = None

    referenceReset = QtCore.Signal(str)
    flatteningRequested = QtCore.Signal(str, float)  # cycle, constant_field

    def __init__(self, model: TrimModel, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self._model = model

        selector_model = LsaSelectorModel(
            accelerator=LsaSelectorAccelerator.SPS, lsa=context.lsa_client
        )
        self.LsaSelector = LsaSelector(model=selector_model, parent=self)

        self.menu_bar = QtWidgets.QMenuBar(parent=self)
        self.menu_bar.setNativeMenuBar(False)
        self.file_menu = QtWidgets.QMenu("&File", parent=self.menu_bar)
        self.actionRefreshLsaSelector = QtWidgets.QAction(
            "Refresh LSA Selector", parent=self
        )
        self.file_menu.addAction(self.actionRefreshLsaSelector)
        self.actionRefreshLsaSelector.triggered.connect(self.LsaSelector.model.refetch)
        self.menu_bar.addMenu(self.file_menu)

        self.TrimInfoWidget = TrimInfoWidget(parent=self)
        model.contextChanged.connect(self.TrimInfoWidget.onContextChanged)
        self.TrimSettingsWidget = TrimSettingsWidget(model=model, parent=self)
        self.TrimSettingsWidget.ResetReferenceButton.clicked.connect(
            self.onResetReference
        )
        self.TrimSettingsWidget.flatteningRequested.connect(
            self.flatteningRequested.emit
        )

        self.left_frame = QtWidgets.QFrame(parent=self)
        self.left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self.left_frame)
        layout.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(self.LsaSelector)
        layout.addWidget(self.TrimInfoWidget)
        layout.addWidget(self.TrimSettingsWidget)
        self.left_frame.setLayout(layout)
        self.left_frame.setMaximumSize(300, 16777215)
        self.left_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )

        self.plotWidget = accgraph.StaticPlotWidget(parent=self, background="w")

        self.setLayout(QtWidgets.QHBoxLayout(self))
        self.layout().addWidget(self.left_frame)
        self.layout().addWidget(self.plotWidget)
        self.layout().setMenuBar(self.menu_bar)

        self.setWindowTitle("SPS Hysteresis Prediction Trims")

        self.setMinimumSize(800, 400)

        self.LsaSelector.contextSelectionChanged.connect(self.onUserSelectionChanged)
        self._plot_source = accgraph.UpdateSource()

        self._setup_plots()

    def _setup_plots(self) -> None:
        self.plotWidget.addCurve(
            data_source=self._plot_source, pen=pg.mkPen(color="k", width=2)
        )

    def onTrimApplied(
        self,
        cycle_data: CycleData,
        values: tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]],
        trim_time: datetime.datetime,
        comment: str,
        *_: typing.Any,
    ) -> None:
        log.info("Trim applied. Updating plot.")
        curve = accgraph.CurveData(*values)

        self._plot_source.send_data(curve)

    @QtCore.Slot(AbstractLsaSelectorContext)
    def onUserSelectionChanged(self, context: AbstractLsaSelectorContext) -> None:
        self._model.setCycle(context.name)

    @QtCore.Slot()
    def onResetReference(self) -> None:
        selected_context = self.LsaSelector.selected_context
        if selected_context is None:
            log.error("No context selected.")
            return

        log.debug(f"Resetting reference for {selected_context.name}")
        self.referenceReset.emit(selected_context.name)

    @QtCore.Slot(bool)
    def onCorrectionModeChanged(self, is_eddy_current_only: bool) -> None:  # noqa: FBT001
        """Update prediction mode in the trim settings widget."""
        self.TrimSettingsWidget.onCorrectionModeChanged(is_eddy_current_only)
