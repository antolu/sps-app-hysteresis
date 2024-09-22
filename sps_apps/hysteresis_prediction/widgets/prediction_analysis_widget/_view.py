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
from op_app_context import context
from qtpy import QtCore, QtGui, QtWidgets

from ...generated.prediction_analysis_widget_ui import (
    Ui_PredictionAnalysisWidget,
)
from ...generated.reference_selector_dialog_ui import (
    Ui_ReferenceSelectorDialog,
)
from ._model import PredictionAnalysisModel
from ._dataclass import Plot, DiffPlotMode, MeasPlotMode

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

        self.LsaSelector = self._setup_lsa_selector()

        self.plotPredWidget = pg.PlotItem()
        self.plotDiffWidget = pg.PlotItem()
        self.plotIMeasWidget = pg.PlotItem()
        self.plotBMeasWidget = pg.PlotItem()
        self.plotDeltaWidget = pg.PlotItem()
        self.plotRefDiffWidget = pg.PlotItem()
        self._setup_plots()

        self.menubar = QtWidgets.QMenuBar(self)
        self.horizontalLayout.setMenuBar(self.menubar)

        self.actionRefreshLsaSelector = QtWidgets.QAction(self)
        self.actionRefreshLsaSelector.setText("&Refresh LSA Selector")
        self.actionRefreshLsaSelector.triggered.connect(self.LsaSelector.model.refetch)

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(self.actionRefreshLsaSelector)
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

        self.buttonStartStop.initializeState("Start", "Stop")

        self.spinBoxNumPredictions.setMaximum(20)

        self._model: PredictionAnalysisModel | None = None
        self.model = model or PredictionAnalysisModel()

        self._connect_slots()

    def _setup_lsa_selector(self) -> lsa_selector.LsaSelector:
        selector_model = lsa_selector.LsaSelectorModel(
            accelerator=lsa_selector.LsaSelectorAccelerator.SPS,
            lsa=context.lsa_client,
            categories={
                lsa_selector.AbstractLsaSelectorContext.Category.MD,
                lsa_selector.AbstractLsaSelectorContext.Category.OPERATIONAL,
            },
        )
        LsaSelector = lsa_selector.LsaSelector(model=selector_model, parent=self)
        self.frame.layout().replaceWidget(self._LsaSelector, LsaSelector)
        self._LsaSelector.deleteLater()

        return LsaSelector

    def _setup_plots(self) -> None:
        """
        ┌─────────┬──────────┐
        │  Diff   │  B pred  │
        │  Widget │  Widget  │
        ├─────────┼──────────┤
        │  I meas │  B meas  │
        │  Widget │  widget  │
        └─────────┴──────────┘

        All widgets are pg.PlotItem and have shared x-axis.

        The diff widget can switch show difference between
        predictions and measurements or show the difference
        between the reference and predictions.

        The widgets for measured values can show raw values or downsampled.
        """
        self.plotDiffWidget.setLabel("left", "E-4 T")
        self.plotDiffWidget.setTitle("Difference [T]")
        self.plotDiffWidget.vb.setYRange(-10, 10)

        self.plotPredWidget.setLabel("left", "T")
        self.plotPredWidget.setTitle("Predicted field [T]")
        self.plotPredWidget.vb.setXLink(self.plotDiffWidget.vb)
        self.plotPredWidget.vb.setYRange(0.0, 2.1)

        self.plotIMeasWidget.setLabel("left", "A")
        self.plotIMeasWidget.setTitle("Measured current [A]")
        self.plotIMeasWidget.vb.setXLink(self.plotDiffWidget.vb)
        self.plotIMeasWidget.vb.setYRange(0.0, 6000)

        self.plotBMeasWidget.setLabel("left", "T")
        self.plotBMeasWidget.setTitle("Measured field [T]")
        self.plotBMeasWidget.vb.setXLink(self.plotDiffWidget.vb)
        self.plotBMeasWidget.vb.setYRange(0.0, 2.1)

        self.plotDeltaWidget.setLabel("left", "E-4 T")
        self.plotDeltaWidget.setTitle("Delta applied [T]")
        self.plotDeltaWidget.vb.setXLink(self.plotDiffWidget.vb)
        self.plotDeltaWidget.vb.setYRange(-5, 5)

        self.plotRefDiffWidget.setLabel("left", "E-4 T")
        self.plotRefDiffWidget.setTitle("Reference difference [T]")
        self.plotRefDiffWidget.vb.setXLink(self.plotDiffWidget.vb)
        self.plotRefDiffWidget.vb.setYRange(-10, 10)

        self.widget.setBackground("w")

        self.widget.addItem(self.plotDiffWidget, row=0, col=0)
        self.widget.addItem(self.plotPredWidget, row=0, col=1)
        self.widget.addItem(self.plotIMeasWidget, row=1, col=0)
        self.widget.addItem(self.plotBMeasWidget, row=1, col=1)
        self.widget.addItem(self.plotDeltaWidget, row=2, col=0)
        self.widget.addItem(self.plotRefDiffWidget, row=2, col=1)

        self.plotDiffWidget.addItem(
            pg.InfiniteLine(pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        )
        self.plotRefDiffWidget.addItem(
            pg.InfiniteLine(pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        )
        self.plotDeltaWidget.addItem(
            pg.InfiniteLine(pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        )

    def _connect_slots(self) -> None:
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
            lambda: model.maxBufferSizeChanged.emit(self.spinBoxNumPredictions.value())
        )

        self.checkBox.stateChanged.connect(
            lambda state: model.set_watch_supercycle(state)
        )

        self.spinBoxSCPatience.editingFinished.connect(
            lambda: model.set_supercycle_patience(self.spinBoxSCPatience.value())
        )

        self.LsaSelector.userSelectionChanged.connect(
            lambda selector: model.set_selector(selector)
        )

        self.listPredictions.clicked.connect(model.item_clicked)

        def on_diff_radio_changed(*_: typing.Any) -> None:
            if self.radioPredVPred.isChecked():
                model.diffPlotModeChanged.emit(DiffPlotMode.PredVsPred)
            elif self.radioPredVMeas.isChecked():
                model.diffPlotModeChanged.emit(DiffPlotMode.PredVsMeas)
            elif self.radioPredVRef.isChecked():
                model.diffPlotModeChanged.emit(DiffPlotMode.PredVsRef)

        self.radioPredVPred.clicked.connect(on_diff_radio_changed)
        self.radioPredVMeas.clicked.connect(on_diff_radio_changed)
        self.radioPredVRef.clicked.connect(on_diff_radio_changed)

        def on_meas_radio_changed(*_: typing.Any) -> None:
            if self.radioMeas.isChecked():
                model.measPlotModeChanged.emit(MeasPlotMode.RawMeas)
            elif self.radioDownsampledMeas.isChecked():
                model.measPlotModeChanged.emit(MeasPlotMode.DownsampledMeas)

        self.radioMeas.clicked.connect(on_meas_radio_changed)
        self.radioDownsampledMeas.clicked.connect(on_meas_radio_changed)

        self.actionClear_Buffer.triggered.connect(model.clear)

        self.buttonStartStop.state1Activated.connect(model.disable_acquisition)
        self.buttonStartStop.state2Activated.connect(model.enable_acquisition)
        self.buttonZoomBI.clicked.connect(model.plot_model.zoomBeamIn.emit)
        self.actionResetAxes.triggered.connect(model.plot_model.resetAxes.emit)

        model.plot_model.plotAdded.connect(self.onPlotAdded)
        model.plot_model.plotRemoved.connect(self.onPlotRemoved)

        model.plot_model.setXRange.connect(
            partial(self.plotPredWidget.vb.setXRange, padding=0)
        )

        model.userChanged.connect(self.LsaSelector.select_user)

    @QtCore.Slot(pg.PlotCurveItem, Plot)
    def onPlotAdded(self, plot: pg.PlotCurveItem, plot_type: Plot) -> None:
        match plot_type:
            case Plot.Diff:
                self.plotDiffWidget.addItem(plot)
            case Plot.Pred:
                self.plotPredWidget.addItem(plot)
            case Plot.MeasI:
                self.plotIMeasWidget.addItem(plot)
            case Plot.MeasB:
                self.plotBMeasWidget.addItem(plot)
            case Plot.Delta:
                self.plotDeltaWidget.addItem(plot)
            case Plot.RefDiff:
                self.plotRefDiffWidget.addItem(plot)
            case _:
                raise ValueError(f"Invalid plot type: {plot_type}")

    @QtCore.Slot(pg.PlotCurveItem, Plot)
    def onPlotRemoved(self, plot: pg.PlotCurveItem, plot_type: Plot) -> None:
        match plot_type:
            case Plot.Diff:
                self.plotDiffWidget.removeItem(plot)
            case Plot.Pred:
                self.plotPredWidget.removeItem(plot)
            case Plot.MeasI:
                self.plotIMeasWidget.removeItem(plot)
            case Plot.MeasB:
                self.plotBMeasWidget.removeItem(plot)
            case Plot.Delta:
                self.plotDeltaWidget.removeItem(plot)
            case Plot.RefDiff:
                self.plotRefDiffWidget.removeItem(plot)
            case _:
                raise ValueError(f"Invalid plot type: {plot_type}")

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
