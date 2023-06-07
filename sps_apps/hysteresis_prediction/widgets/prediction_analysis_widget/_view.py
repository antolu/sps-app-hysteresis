"""
This file contains the view for the prediction analysis widget.
"""
from __future__ import annotations

import logging
import typing
from functools import partial

import pandas as pd
import pyqtgraph as pg
from accwidgets import lsa_selector
from qtpy import QtCore, QtGui, QtWidgets

from ...core.application_context import context
from ...generated.prediction_analysis_widget_ui import (
    Ui_PredictionAnalysisWidget,
)
from ...generated.reference_selector_dialog_ui import (
    Ui_ReferenceSelectorDialog,
)
from ._model import PlotMode, PredictionAnalysisModel

log = logging.getLogger(__name__)


class ReferenceSelectorDialog(QtWidgets.QDialog, Ui_ReferenceSelectorDialog):
    _model: QtCore.QAbstractProxyModel | None

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
    windowClosed = QtCore.Signal()

    def __init__(
        self,
        model: PredictionAnalysisModel | None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)

        selector_model = lsa_selector.LsaSelectorModel(
            accelerator=lsa_selector.LsaSelectorAccelerator.SPS,
            lsa=context.lsa,
            categories={
                lsa_selector.AbstractLsaSelectorContext.Category.MD,
                lsa_selector.AbstractLsaSelectorContext.Category.OPERATIONAL,
            },
        )
        self.LsaSelector = lsa_selector.LsaSelector(
            model=selector_model, parent=parent
        )
        self.frame.layout().replaceWidget(self._LsaSelector, self.LsaSelector)
        self._LsaSelector.deleteLater()

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

        self.menubar = QtWidgets.QMenuBar(self)
        self.horizontalLayout.setMenuBar(self.menubar)

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(self.actionImport_Predictions)
        file_menu.addAction(self.actionExport_Predictions)
        file_menu.addSeparator()
        file_menu.addAction(self.actionExit)

        tools_menu = self.menubar.addMenu("&Tools")
        tools_menu.addAction(self.actionClear_Buffer)

        view_menu = self.menubar.addMenu("&View")
        self.actionResetAxes = QtWidgets.QAction(self)
        self.actionResetAxes.setText("&Reset Axes")
        view_menu.addAction(self.actionResetAxes)

        self.widget.addItem(self.plotDiffWidget, row=0, col=0)
        self.widget.addItem(self.plotPredWidget, row=1, col=0, rowspan=3)

        self.buttonStartStop.initializeState("Start", "Stop")

        self.spinBoxNumPredictions.setMaximum(20)

        self.plotDiffWidget.addItem(
            pg.InfiniteLine(
                pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine)
            )
        )
        self.buttonZoomFB.setEnabled(True)
        self.buttonZoomFT.setEnabled(True)
        self.buttonReference.hide()

        self._model: PredictionAnalysisModel | None = None
        self.model = model or PredictionAnalysisModel()

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
        self.actionExport_Predictions.triggered.connect(self._on_save_clicked)
        self.actionImport_Predictions.triggered.connect(self._on_load_clicked)

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
                self.buttonReference.hide()
                self.buttonZoomFT.setEnabled(True)
                self.buttonZoomFB.setEnabled(True)
                self.buttonZoomFT.show()
                self.buttonZoomFB.show()
            elif self.radioMeasured.isChecked():
                model.plotModeChanged.emit(PlotMode.VsMeasured)

                self.buttonReference.setEnabled(False)
                self.buttonZoomFT.setEnabled(False)
                self.buttonZoomFB.setEnabled(False)
                self.buttonReference.hide()
                self.buttonZoomFT.hide()
                self.buttonZoomFB.hide()
            elif self.radioDpp.isChecked():
                model.plotModeChanged.emit(PlotMode.dpp)

                self.buttonReference.setEnabled(True)
                self.buttonZoomFT.setEnabled(False)
                self.buttonZoomFB.setEnabled(False)
                self.buttonReference.show()
                self.buttonZoomFT.hide()
                self.buttonZoomFB.hide()

        self.radioPredicted.clicked.connect(radio_changed)
        self.radioMeasured.toggled.connect(radio_changed)
        self.radioDpp.clicked.connect(radio_changed)

        self.actionClear_Buffer.triggered.connect(model.clear)

        self.buttonStartStop.state1Activated.connect(model.disable_acquisition)
        self.buttonStartStop.state2Activated.connect(model.enable_acquisition)
        self.buttonZoomFT.clicked.connect(model.plot_model.zoomFlatTop.emit)
        self.buttonZoomFB.clicked.connect(model.plot_model.zoomFlatBottom.emit)
        self.buttonZoomBI.clicked.connect(model.plot_model.zoomBeamIn.emit)
        self.actionResetAxes.triggered.connect(model.plot_model.resetAxes.emit)

        model.plot_model.plotAdded.connect(self.plotPredWidget.addItem)
        model.plot_model.plotRemoved.connect(self.plotPredWidget.removeItem)
        model.plot_model.plotAdded_dpp.connect(self.plotDiffWidget.addItem)
        model.plot_model.plotRemoved_dpp.connect(
            self.plotDiffWidget.removeItem
        )
        model.plot_model.setXRange.connect(
            partial(self.plotPredWidget.vb.setXRange, padding=0)
        )
        model.plot_model.setYRange.connect(self.plotPredWidget.vb.setYRange)
        model.userChanged.connect(self.LsaSelector.select_user)

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

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.windowClosed.emit()
        event.accept()

    def _on_save_clicked(self) -> None:
        if self._model is None:
            raise ValueError("Model has not been set.")

        log.debug("Saving predictions to file.")
        file_name, ok = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export predictions", "", "parquet (*.parquet)"
        )

        if not ok or not file_name or file_name == "":
            log.debug("Export cancelled.")
            return

        df = self.model.to_pandas()

        df.to_parquet(file_name)

    def _on_load_clicked(self) -> None:
        if self._model is None:
            raise ValueError("Model has not been set.")

        log.debug("Loading predictions from file.")
        file_name, ok = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import predictions", "", "parquet (*.parquet)"
        )

        if not ok or not file_name or file_name == "":
            log.debug("Import cancelled.")
            return

        df = pd.read_parquet(file_name)

        self.model.from_pandas(df)
