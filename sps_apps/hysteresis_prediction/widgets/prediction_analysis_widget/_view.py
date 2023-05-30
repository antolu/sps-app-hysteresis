"""
This file contains the view for the prediction analysis widget.
"""
from __future__ import annotations

import logging
import typing

import pyqtgraph as pg
from qtpy import QtCore, QtWidgets

from ...generated.prediction_analysis_widget_ui import (
    Ui_PredictionAnalysisWidget,
)
from ...generated.reference_selector_dialog_ui import (
    Ui_ReferenceSelectorDialog,
)
from ._model import PlotMode, PredictionAnalysisModel

log = logging.getLogger(__name__)


class ReferenceSelectorDialog(QtWidgets.QDialog, Ui_ReferenceSelectorDialog):
    def __init__(
        self,
        model: QtCore.QAbstractListModel | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)

        if model is not None:
            proxy_model = QtCore.QIdentityProxyModel()
            proxy_model.setSourceModel(model)
            self._model = proxy_model
        else:
            self._model = None

        self.listView.setModel(self._model)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def _get_model(self) -> QtCore.QAbstractProxyModel:
        if self._model is None:
            raise ValueError("Model has not been set.")
        return self._model

    def _set_model(self, model: QtCore.QAbstractListModel) -> None:
        proxy_model = QtCore.QIdentityProxyModel()
        proxy_model.setSourceModel(model)
        self._model = proxy_model
        self.listView.setModel(self._model)

    model = property(_get_model, _set_model)

    @property
    def selected_item(self) -> QtCore.QModelIndex | None:
        try:
            return self.listView.selectedIndexes()[0]
        except IndexError:
            return None


class PredictionAnalysisWidget(QtWidgets.QWidget, Ui_PredictionAnalysisWidget):
    def __init__(
        self,
        model: PredictionAnalysisModel | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)
        self.plotPredWidget = pg.PlotItem()
        self.plotDiffWidget = pg.PlotItem()
        # self.plotPredWidget.setBackgroundColor("w")  # type: ignore
        # self.plotDiffWidget.setBackgroundColor("w")  # type: ignore
        self.plotDiffWidget.setLabel("left", "E-4")
        self.plotDiffWidget.setMinimumHeight(100)
        self.plotDiffWidget.setMaximumHeight(300)
        self.plotDiffWidget.vb.setYRange(-10, 10)
        self.plotPredWidget.vb.setXLink(self.plotDiffWidget.vb)
        self.widget.setBackground("w")

        self._model: PredictionAnalysisModel | None = None
        self.model = model or PredictionAnalysisModel()

        self.menubar = QtWidgets.QMenuBar(self)
        self.horizontalLayout.setMenuBar(self.menubar)

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(self.actionExport_Predictions)
        file_menu.addSeparator()
        file_menu.addAction(self.actionExit)

        tools_menu = self.menubar.addMenu("&Tools")
        tools_menu.addAction(self.actionClear_Buffer)

        self.widget.addItem(self.plotDiffWidget, row=0, col=0)
        self.widget.addItem(self.plotPredWidget, row=1, col=0, rowspan=3)

        # self.plotDiffWidget.addItem(
        #     pg.InfiniteLine(
        #         pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine)
        #     )
        # )

        self._connect_slots()

    def _connect_slots(self) -> None:
        def num_predictions_changed() -> None:
            num_predictions = self.spinBoxNumPredictions.value()
            self.model.set_max_buffer_samples(num_predictions)

            self.spinBoxSCPatience.setMaximum(num_predictions)
            if self.checkBox.isEnabled():
                self.spinBoxSCPatience.editingFinished.emit()

        self.spinBoxNumPredictions.editingFinished.connect(
            num_predictions_changed
        )

        self.spinBoxYMax.valueChanged.connect(
            lambda value: self.plotPredWidget.setYRange(
                (self.spinBoxYMin.value(), value)
            )
        )
        self.spinBoxYMin.valueChanged.connect(
            lambda value: self.plotPredWidget.setYRange(
                (value, self.spinBoxYMax.value())
            )
        )

        self.checkBox.stateChanged.connect(self.spinBoxSCPatience.setEnabled)
        self.buttonReference.clicked.connect(self._on_select_new_reference)

        self.actionExit.triggered.connect(self.close)

    def _get_model(self) -> PredictionAnalysisModel:
        if self._model is None:
            raise ValueError("Model has not been set.")

        return self._model

    def _set_model(self, model: PredictionAnalysisModel) -> None:
        if self._model is not None:
            self._disconnect_model(self._model)

        self._connect_model(model)
        self._model = model

    model = property(_get_model, _set_model)

    def _connect_model(self, model: PredictionAnalysisModel) -> None:
        self.listPredictions.setModel(model.list_model)

        self.spinBoxNumPredictions.editingFinished.connect(
            lambda: model.maxBufferSizeChanged.emit(
                self.spinBoxNumPredictions.value()
            )
        )

        self.checkBox.stateChanged.connect(
            lambda state: model.set_watch_supercycle(state)
        )

        self.spinBoxSCPatience.editingFinished.connect(
            lambda: model.set_supercycle_patience(
                self.spinBoxSCPatience.value()
            )
        )

        self.LsaSelector.userSelectionChanged.connect(
            lambda selector: model.set_selector(selector)
        )

        self.listPredictions.clicked.connect(model.item_clicked)

        def radio_changed(*_: typing.Any) -> None:
            if self.radioPredicted.isChecked():
                model.plotModeChanged.emit(PlotMode.PredictedOnly)
                self.buttonReference.setEnabled(False)
                self.buttonZoomFT.setEnabled(True)
                self.buttonZoomFB.setEnabled(True)
            elif self.radioMeasured.isChecked():
                model.plotModeChanged.emit(PlotMode.VsMeasured)
                self.buttonReference.setEnabled(True)
                self.buttonZoomFT.setEnabled(False)
                self.buttonZoomFB.setEnabled(False)
            elif self.radioDpp.isChecked():
                model.plotModeChanged.emit(PlotMode.dpp)
                self.buttonReference.setEnabled(True)
                self.buttonZoomFT.setEnabled(False)
                self.buttonZoomFB.setEnabled(False)

        self.radioPredicted.clicked.connect(radio_changed)
        # self.radioMeasured.toggled.connect(radio_changed)
        self.radioDpp.clicked.connect(radio_changed)

        model.plot_model.plotAdded.connect(self.plotPredWidget.addItem)
        model.plot_model.plotRemoved.connect(self.plotPredWidget.removeItem)
        model.plot_model.plotAdded_dpp.connect(self.plotDiffWidget.addItem)
        model.plot_model.plotRemoved_dpp.connect(
            self.plotDiffWidget.removeItem
        )

    def _disconnect_model(self, model: PredictionAnalysisModel) -> None:
        raise NotImplementedError("Disconnect model not implemented.")

    def _on_select_new_reference(self) -> None:
        """
        Select a new reference cycle for plots, use a QIdentityProxyModel,
        but select only one.
        """
        list_model = self.model.list_model
        assert list_model is not None
        dialog = ReferenceSelectorDialog(model=list_model, parent=self)

        def on_dialog_accepted() -> None:
            selected_index = dialog.selected_item
            if selected_index is not None:
                item = self.model.list_model.itemAt(selected_index)

                self.model.plot_model.newReference.emit(item)

        dialog.accepted.connect(on_dialog_accepted)

        dialog.open()
