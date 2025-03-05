from __future__ import annotations

import datetime
import logging
import typing

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from accwidgets import graph as accgraph
from accwidgets.lsa_selector import (
    LsaSelector,
    LsaSelectorAccelerator,
    LsaSelectorModel,
)
from op_app_context import context
from qtpy import QtCore, QtWidgets

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

    def on_trim_applied(
        self, _: typing.Any, trim_time: datetime.datetime, trim_comment: str
    ) -> None:
        self.LastTrimLineValue.setText(trim_time.strftime("%Y%m%d-%H:%M:%S:%f")[:-4])
        self.LastCommentLineValue.setText(trim_comment)

    def on_new_beam_in_time(self, beam_in: int, beam_out: int) -> None:
        self.BeamInLineValue.setText(f"C{beam_in} - C{beam_out}")


class TrimSettingsWidget(QtWidgets.QWidget):
    DryRunChanged = QtCore.Signal(bool)
    FlattenChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

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

        self.FlattenLabel = QtWidgets.QLabel("Flatten", parent=self)
        self.FlattenCheckBox = QtWidgets.QCheckBox(parent=self)
        self.FlattenCheckBox.setEnabled(False)
        self.FlattenLabel.setToolTip(
            "Flatten field between trim T min and trim T max, "
            "with the value of the field at trim T min."
        )

        self.setLayout(QtWidgets.QGridLayout(self))
        self.layout().addWidget(self.GainLabel, 0, 0)
        self.layout().addWidget(self.GainSpinBox, 0, 1)
        self.layout().addWidget(self.DryRunLabel, 1, 0)
        self.layout().addWidget(self.DryRunCheckBox, 1, 1)
        self.layout().addWidget(self.TrimTMinLabel, 2, 0)
        self.layout().addWidget(self.TrimTMinSpinBox, 2, 1)
        self.layout().addWidget(self.TrimTMaxLabel, 3, 0)
        self.layout().addWidget(self.TrimTMaxSpinBox, 3, 1)
        self.layout().addWidget(self.FlattenLabel, 4, 0)
        self.layout().addWidget(self.FlattenCheckBox, 4, 1)

        self.DryRunCheckBox.stateChanged.connect(self.on_dry_run_changed)
        self.GainSpinBox.valueChanged.connect(self.on_gain_changed)
        self.TrimTMinSpinBox.valueChanged.connect(self.on_min_value_changed)
        self.TrimTMaxSpinBox.valueChanged.connect(self.on_max_value_changed)
        self.FlattenCheckBox.stateChanged.connect(self.on_flatten_changed)

    @QtCore.Slot(int)
    def on_dry_run_changed(self, state: QtCore.Qt.CheckState) -> None:
        value = state == QtCore.Qt.Checked
        self.DryRunChanged.emit(value)

    @QtCore.Slot(int)
    def on_flatten_changed(self, state: QtCore.Qt.CheckState) -> None:
        value = state == QtCore.Qt.Checked
        self.FlattenChanged.emit(value)

    @QtCore.Slot(float)
    def on_gain_changed(self, value: float) -> None:
        if value <= 1.0:
            self.GainSpinBox.setSingleStep(0.02)
        else:
            self.GainSpinBox.setSingleStep(0.2)

    @QtCore.Slot(int, int)
    def on_new_beam_in_time(self, beam_in: int, beam_out: int) -> None:
        if not self.TrimTMinSpinBox.isEnabled():
            self.TrimTMinSpinBox.setEnabled(True)
            self.TrimTMaxSpinBox.setEnabled(True)
            self.FlattenCheckBox.setEnabled(True)

        self.TrimTMinSpinBox.setMinimum(beam_in)
        self.TrimTMinSpinBox.setMaximum(beam_out)
        self.TrimTMaxSpinBox.setMinimum(beam_in)
        self.TrimTMaxSpinBox.setMaximum(beam_out)

        self.TrimTMaxSpinBox.setValue(beam_out)
        self.TrimTMinSpinBox.setValue(beam_in)

    @QtCore.Slot(int)
    def on_min_value_changed(self, value: int) -> None:
        self.TrimTMaxSpinBox.setMinimum(value + 1)

        if value > self.TrimTMaxSpinBox.value():
            self.TrimTMaxSpinBox.setValue(value)

    @QtCore.Slot(int)
    def on_max_value_changed(self, value: int) -> None:
        self.TrimTMinSpinBox.setMaximum(value - 1)

        if value < self.TrimTMinSpinBox.value():
            self.TrimTMinSpinBox.setValue(value)


class TrimWidgetView(QtWidgets.QWidget):
    windowClosed = QtCore.Signal()

    _thread: QtCore.QThread | None = None

    def __init__(self, model: TrimModel, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)
        self._model = None

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
        self.TrimSettingsWidget = TrimSettingsWidget(parent=self)

        self.toggle_button = ToggleButton(parent=self)
        self.toggle_button.initializeState(
            label_s2="Enable Trim",
            label_s1="Disable Trim",
            initial_state=ToggleButton.State.STATE2,
        )

        self.left_frame = QtWidgets.QFrame(parent=self)
        self.left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_frame.setLayout(QtWidgets.QVBoxLayout(self.left_frame))
        self.left_frame.layout().setAlignment(QtCore.Qt.AlignLeft)
        self.left_frame.layout().addWidget(self.LsaSelector)
        self.left_frame.layout().addWidget(self.TrimInfoWidget)
        self.left_frame.layout().addWidget(self.TrimSettingsWidget)
        self.left_frame.layout().addWidget(self.toggle_button)
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

        self.LsaSelector.userSelectionChanged.connect(self.on_user_selected_lsa)
        self.toggle_button.setEnabled(False)

        self._plot_source = accgraph.UpdateSource()

        self.model = model

        self._setup_plots()

        if self._thread is None:
            self._thread = QtCore.QThread()
            self._thread.start()

    def _setup_plots(self) -> None:
        self.plotWidget.addCurve(data_source=self._plot_source, pen=pg.mkPen(color="k"))

    @property
    def model(self) -> TrimModel | None:
        return self._model

    @model.setter
    def model(self, value: TrimModel) -> None:
        if self._model is not None:
            self._disconnect_model(self._model)

        self._connect_model(value)
        self._model = value

    def _connect_model(self, model: TrimModel) -> None:
        model.trimApplied.connect(self.TrimInfoWidget.on_trim_applied)
        model.trimApplied.connect(self._on_trim_applied)
        model.beamInRetrieved.connect(self.TrimInfoWidget.on_new_beam_in_time)
        model.beamInRetrieved.connect(self.TrimSettingsWidget.on_new_beam_in_time)

        self.toggle_button.state1Activated.connect(model.enable_trim)
        self.toggle_button.state2Activated.connect(model.disable_trim)

        self.TrimSettingsWidget.DryRunChanged.connect(model.set_dry_run)
        self.TrimSettingsWidget.GainSpinBox.valueChanged.connect(model.set_gain)
        self.TrimSettingsWidget.TrimTMaxSpinBox.valueChanged.connect(
            model.set_trim_t_max
        )
        self.TrimSettingsWidget.TrimTMinSpinBox.valueChanged.connect(
            model.set_trim_t_min
        )
        self.TrimSettingsWidget.FlattenChanged.connect(model.set_flatten)

    def _disconnect_model(self, model: TrimModel) -> None:
        model.trimApplied.disconnect(self.TrimInfoWidget.on_trim_applied)
        model.trimApplied.disconnect(self._on_trim_applied)
        model.beamInRetrieved.disconnect(self.TrimInfoWidget.on_new_beam_in_time)
        model.beamInRetrieved.disconnect(self.TrimSettingsWidget.on_new_beam_in_time)

        self.toggle_button.state1Activated.disconnect(model.enable_trim)
        self.toggle_button.state2Activated.disconnect(model.disable_trim)
        self.toggle_button.setEnabled(False)

        self.TrimSettingsWidget.DryRunChanged.disconnect(model.set_dry_run)
        self.TrimSettingsWidget.GainSpinBox.valueChanged.disconnect(model.set_gain)
        self.TrimSettingsWidget.TrimTMaxSpinBox.valueChanged.disconnect(
            model.set_trim_t_max
        )
        self.TrimSettingsWidget.TrimTMinSpinBox.valueChanged.disconnect(
            model.set_trim_t_min
        )
        self.TrimSettingsWidget.FlattenChanged.disconnect(model.set_flatten)

    def _on_trim_applied(
        self,
        values: tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]],
        *_: typing.Any,
    ) -> None:
        log.info("Trim applied. Updating plot.")
        curve = accgraph.CurveData(*values)

        self._plot_source.send_data(curve)

    def on_user_selected_lsa(self, user: str) -> None:
        if self.model is None:
            msg = "Model is not set."
            raise RuntimeError(msg)

        if user != self.model.selector:
            self.toggle_button.setEnabled(False)

            if self.toggle_button.state == ToggleButton.State.STATE2:
                self.toggle_button.toggle()

            if self._thread is not None:
                self.model.selector = user

                self.toggle_button.setEnabled(True)


if __name__ == "__main__":
    from qtpy import QtWidgets

    app = QtWidgets.QApplication([])
    model = TrimModel()
    view = TrimWidgetView(model=model)
    view.show()
    app.exec_()
