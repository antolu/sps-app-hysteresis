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
from hystcomp_utils.cycle_data import CycleData
from op_app_context import context
from qtpy import QtCore, QtWidgets

from ...trim import cycle_metadata
from .._widgets import ToggleButton
from ._model import TrimModel

log = logging.getLogger(__name__)


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
    def __init__(self, model: TrimModel, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.model = model

        self.GainLabel = QtWidgets.QLabel("Gain", parent=self)
        self.GainSpinBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.GainSpinBox.setRange(0.0, 5.0)
        self.GainSpinBox.setSingleStep(0.1)
        self.GainSpinBox.setValue(1.0)
        self.GainSpinBox.setMaximumWidth(50)

        self.DryRunLabel = QtWidgets.QLabel("Dry Run", parent=self)
        self.DryRunCheckBox = QtWidgets.QCheckBox(parent=self)

        self.TrimTMinLabel = QtWidgets.QLabel("Trim T Min", parent=self)
        self.TrimTMinSpinBox = QtWidgets.QSpinBox(parent=self)
        self.TrimTMinSpinBox.setEnabled(False)
        self.TrimTMinSpinBox.setMaximumWidth(80)

        self.TrimTMaxLabel = QtWidgets.QLabel("Trim T Max", parent=self)
        self.TrimTMaxSpinBox = QtWidgets.QSpinBox(parent=self)
        self.TrimTMaxSpinBox.setEnabled(False)
        self.TrimTMaxSpinBox.setMaximumWidth(80)

        self.toggle_button = ToggleButton(parent=self)
        self.toggle_button.initializeState(
            label_s2="Enable Trim",
            label_s1="Disable Trim",
            initial_state=ToggleButton.State.STATE2,
        )

        layout = QtWidgets.QGridLayout(self)
        layout.addWidget(self.GainLabel, 0, 0)
        layout.addWidget(self.GainSpinBox, 0, 1)
        layout.addWidget(self.TrimTMinLabel, 1, 0)
        layout.addWidget(self.TrimTMinSpinBox, 1, 1)
        layout.addWidget(self.TrimTMaxLabel, 2, 0)
        layout.addWidget(self.TrimTMaxSpinBox, 2, 1)
        layout.addWidget(self.toggle_button, 3, 0, 1, 2)
        self.setLayout(layout)

        self.DryRunCheckBox.stateChanged.connect(self.onDryRunChanged)
        self.GainSpinBox.valueChanged.connect(self.onGainChanged)
        self.TrimTMinSpinBox.valueChanged.connect(self.onMinValueChanged)
        self.TrimTMaxSpinBox.valueChanged.connect(self.onMaxValueChanged)
        self.toggle_button.stateChanged.connect(self.onEnableTrim)

        # disable the trim settings until a context has been set
        self.GainSpinBox.setEnabled(False)
        self.DryRunCheckBox.setEnabled(False)
        self.TrimTMinSpinBox.setEnabled(False)
        self.TrimTMaxSpinBox.setEnabled(False)

        self.model.contextChanged.connect(self.on_context_changed)

    @QtCore.Slot(str)
    def onContextChanged(self) -> None:
        try:
            cycle = self.model.cycle
        except ValueError:
            return

        self.GainSpinBox.setEnabled(True)
        self.TrimTMinSpinBox.setEnabled(True)
        self.TrimTMaxSpinBox.setEnabled(True)
        self.toggle_button.setEnabled(True)

        self.TrimTMinSpinBox.setMinimum(cycle_metadata.beam_in(cycle))
        self.TrimTMaxSpinBox.setMaximum(cycle_metadata.beam_out(cycle))

        self.GainSpinBox.setValue(self.model.settings.gain[cycle])
        self.TrimTMinSpinBox.setValue(self.model.settings.trim_start[cycle])
        self.TrimTMaxSpinBox.setValue(self.model.settings.trim_end[cycle])

        if self.model.settings.trim_enabled[cycle]:
            self.toggle_button.setState(ToggleButton.State.STATE2)
        else:
            self.toggle_button.setState(ToggleButton.State.STATE1)

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

    @QtCore.Slot(bool)
    def onEnableTrim(self, state: ToggleButton.State) -> None:
        self.model.settings.trim_enabled[self.model.cycle] = (
            state == ToggleButton.State.STATE2
        )


class TrimWidgetView(QtWidgets.QWidget):
    _thread: QtCore.QThread | None = None

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

        self.TrimInfoWidget = TrimInfoWidget(parent=self)
        model.contextChanged.connect(self.TrimInfoWidget.onContextChanged)
        self.TrimSettingsWidget = TrimSettingsWidget(model=model, parent=self)

        self.left_frame = QtWidgets.QFrame(parent=self)
        self.left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        layout = QtWidgets.QVBoxLayout(self.left_frame)
        layout.setAlignment(QtCore.Qt.AlignLeft)
        layout.addWidget(self.LsaSelector)
        layout.addWidget(self.TrimInfoWidget)
        layout.addWidget(self.TrimSettingsWidget)
        layout.addWidget(self.toggle_button)
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

    def onUserSelectionChanged(self, context: AbstractLsaSelectorContext) -> None:
        self._model.setCycle(context.name)
