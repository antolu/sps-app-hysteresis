from __future__ import annotations

import datetime
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


class TrimInfoWidget(QtWidgets.QWidget):
    DryRunChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        super().__init__(parent=parent)

        self.LsaServerLabel = QtWidgets.QLabel("LSA Server", parent=self)
        self.LsaServerLineValue = QtWidgets.QLabel(
            context.lsa_server, parent=self
        )

        self.LastTrimLabel = QtWidgets.QLabel("Last Trim", parent=self)
        self.LastTrimLineValue = QtWidgets.QLabel("N/A", parent=self)

        self.LastCommentLabel = QtWidgets.QLabel("Last Comment", parent=self)
        self.LastCommentLineValue = QtWidgets.QLabel("N/A", parent=self)
        self.LastCommentLineValue.setWordWrap(True)

        # vertical spacer
        spacer = QtWidgets.QSpacerItem(
            10,
            50,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )

        self.GainLabel = QtWidgets.QLabel("Gain", parent=self)
        self.GainLabel.setMaximumWidth(100)
        self.GainSpinBox = QtWidgets.QDoubleSpinBox(parent=self)
        self.GainSpinBox.setRange(0.0, 5.0)
        self.GainSpinBox.setSingleStep(0.1)
        self.GainSpinBox.setValue(1.0)
        self.GainSpinBox.setMaximumWidth(50)

        self.DryRunLabel = QtWidgets.QLabel("Dry Run", parent=self)
        self.DryRunLabel.setMaximumWidth(100)
        self.DryRunCheckBox = QtWidgets.QCheckBox(parent=self)

        grid_layout = QtWidgets.QGridLayout(self)
        self.setLayout(grid_layout)
        grid_layout.addWidget(self.LsaServerLabel, 0, 0)
        grid_layout.addWidget(self.LsaServerLineValue, 0, 1)
        grid_layout.addWidget(self.LastTrimLabel, 1, 0)
        grid_layout.addWidget(self.LastTrimLineValue, 1, 1)
        grid_layout.addWidget(self.LastCommentLabel, 2, 0)
        grid_layout.addWidget(self.LastCommentLineValue, 2, 1)
        grid_layout.addItem(spacer, 3, 0, 1, 0)
        grid_layout.addWidget(self.GainLabel, 4, 0)
        grid_layout.addWidget(self.GainSpinBox, 4, 1)
        grid_layout.addWidget(self.DryRunLabel, 5, 0)
        grid_layout.addWidget(self.DryRunCheckBox, 5, 1)

        self.DryRunCheckBox.stateChanged.connect(self.on_dry_run_changed)
        self.GainSpinBox.valueChanged.connect(self.on_gain_changed)

    def on_trim_applied(
        self, _: typing.Any, trim_time: datetime.datetime, trim_comment: str
    ) -> None:
        self.LastTrimLineValue.setText(
            trim_time.strftime("%Y%m%d-%H:%M:%S:%f")[:-4]
        )
        self.LastCommentLineValue.setText(trim_comment)

    @QtCore.Slot(int)
    def on_dry_run_changed(self, state: QtCore.Qt.CheckState) -> None:
        value = state == QtCore.Qt.Checked
        self.DryRunChanged.emit(value)

    @QtCore.Slot(float)
    def on_gain_changed(self, value: float) -> None:
        if value <= 1.0:
            self.GainSpinBox.setSingleStep(0.02)
        else:
            self.GainSpinBox.setSingleStep(0.2)


class TrimWidgetView(QtWidgets.QWidget):
    windowClosed = QtCore.Signal()

    _thread: QtCore.QThread | None = None

    def __init__(
        self, model: TrimModel, parent: QtWidgets.QWidget | None = None
    ):
        super().__init__(parent=parent)
        self._model = None

        selector_model = LsaSelectorModel(
            accelerator=LsaSelectorAccelerator.SPS, lsa=context.lsa_client
        )
        self.LsaSelector = LsaSelector(model=selector_model, parent=self)

        self.TrimInfoWidget = TrimInfoWidget(parent=self)

        self.toggle_button = ToggleButton(parent=self)
        self.toggle_button.initializeState(
            label_s2="Enable Trim",
            label_s1="Disable Trim",
            initial_state=ToggleButton.State.STATE2,
        )

        self.left_frame = QtWidgets.QFrame(parent=self)
        self.left_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.left_frame.setLayout(QtWidgets.QVBoxLayout(self.left_frame))
        self.left_frame.layout().addWidget(self.LsaSelector)
        self.left_frame.layout().addWidget(self.TrimInfoWidget)
        self.left_frame.layout().addWidget(self.toggle_button)
        self.left_frame.setMaximumSize(300, 16777215)
        self.left_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )

        self.plotWidget = accgraph.StaticPlotWidget(
            parent=self, background="w"
        )

        self.setLayout(QtWidgets.QHBoxLayout(self))
        self.layout().addWidget(self.left_frame)
        self.layout().addWidget(self.plotWidget)

        self.setMinimumSize(800, 400)

        self.LsaSelector.userSelectionChanged.connect(
            self.on_user_selected_lsa
        )
        self.toggle_button.setEnabled(False)

        self._plot_source = accgraph.UpdateSource()

        self.model = model

        self._setup_plots()

        if self._thread is None:
            self._thread = QtCore.QThread()
            self._thread.start()

    def _setup_plots(self) -> None:
        self.plotWidget.addCurve(
            data_source=self._plot_source, pen=pg.mkPen(color="k")
        )

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

        self.toggle_button.state1Activated.connect(model.enable_trim)
        self.toggle_button.state2Activated.connect(model.disable_trim)

        self.TrimInfoWidget.DryRunChanged.connect(model.set_dry_run)
        self.TrimInfoWidget.GainSpinBox.valueChanged.connect(model.set_gain)

    def _disconnect_model(self, model: TrimModel) -> None:
        model.trimApplied.disconnect(self.TrimInfoWidget.on_trim_applied)
        model.trimApplied.disconnect(self._on_trim_applied)

        self.toggle_button.state1Activated.disconnect(model.enable_trim)
        self.toggle_button.state2Activated.disconnect(model.disable_trim)
        self.toggle_button.setEnabled(False)

        self.TrimInfoWidget.DryRunChanged.disconnect(model.set_dry_run)

        self.TrimInfoWidget.GainSpinBox.valueChanged.disconnect(model.set_gain)

    def _on_trim_applied(
        self,
        values: tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]],
        *_: typing.Any,
    ) -> None:
        curve = accgraph.CurveData(*values)

        self._plot_source.send_data(curve)

    def on_user_selected_lsa(self, user: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not set.")

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
