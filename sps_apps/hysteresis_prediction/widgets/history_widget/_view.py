from __future__ import annotations

import logging

import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from ...generated.prediction_history_widget_ui import Ui_PredictionAnalysisWidget
from ...generated.reference_selector_dialog_ui import Ui_ReferenceSelectorDialog
from ...history import HistoryListModel
from ._model import PredictionListModel
from ._plot_model import PredictionPlotModel

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
            msg = "Model has not been set."
            raise ValueError(msg)
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


class PlotContainer(QtCore.QObject):
    def __init__(
        self, parent: pg.GraphicsLayoutWidget, *, plot_measured: bool = True
    ) -> None:
        super().__init__(parent)

        self.widget = parent
        self.plot_measured = plot_measured

        self.measuredCurrentPlot = pg.PlotItem(name="Measured current")
        self.predictedFieldPlot = pg.PlotItem(name="Predicted field")

        self.predictedFieldRefDiffPlot = pg.PlotItem(
            name="Predicted field - reference prediction"
        )
        self.deltaPlot = pg.PlotItem(name="Delta computed w.r.t. reference (trim)")

        if plot_measured:
            self.measuredFieldPlot = pg.PlotItem(name="Measured field")
            self.measuredFieldRefDiffPlot = pg.PlotItem(
                name="Measured field - reference prediction"
            )
        else:
            self.measuredFieldPlot = None
            self.measuredFieldRefDiffPlot = None

        self._setup_plots()
        self._default_lines()
        self._add_plots()

    def _setup_plots(self) -> None:
        """
        ┌───────────────┌────────────────┌────────────────┐
        │               │                │                │
        │               │                │                │
        │  Meas. I      │  Meas. B       │  Pred B        │
        │               │                │                │
        │               │                │                │
        ├───────────────├────────────────├────────────────┤
        │               │                │                │
        │               │                │                │
        │  Delta appl.  │  Meas. B Ref.  │  Pred B Ref    │
        │               │                │                │
        │               │                │                │
        └───────────────└────────────────└────────────────┘

        All widgets are pg.PlotItem and have shared x-axis.

        The diff widget can switch show difference between
        predictions and measurements or show the difference
        between the reference and predictions.

        The widgets for measured values can show raw values or downsampled.
        """
        self.measuredCurrentPlot.setLabel("left", "A")
        self.measuredCurrentPlot.setTitle("Measured current [A]")
        assert self.measuredCurrentPlot.vb is not None
        self.measuredCurrentPlot.vb.setYRange(0.0, 6000.0)

        self.predictedFieldPlot.setLabel("left", "T")
        self.predictedFieldPlot.setTitle("Predicted Field [t]")
        assert self.predictedFieldPlot.vb is not None
        self.predictedFieldPlot.vb.setXLink(self.measuredCurrentPlot.vb)
        self.predictedFieldPlot.vb.setYRange(0.0, 2.1)

        self.predictedFieldRefDiffPlot.setLabel("left", "E-4 T")
        self.predictedFieldRefDiffPlot.setTitle("Predicted field w.r.t reference [T]")
        assert self.predictedFieldRefDiffPlot.vb is not None
        self.predictedFieldRefDiffPlot.vb.setXLink(self.measuredCurrentPlot.vb)
        self.predictedFieldRefDiffPlot.vb.setYRange(-10, 10)

        if self.plot_measured:
            assert self.measuredFieldPlot is not None
            assert self.measuredFieldRefDiffPlot is not None
            self.measuredFieldPlot.setLabel("left", "T")
            self.measuredFieldPlot.setTitle("Measured field [T]")
            assert self.measuredFieldPlot.vb is not None
            self.measuredFieldPlot.vb.setXLink(self.measuredCurrentPlot.vb)
            self.measuredFieldPlot.vb.setYLink(self.predictedFieldPlot.vb)
            self.measuredFieldPlot.vb.setYRange(0, 2.1)

            self.measuredFieldRefDiffPlot.setLabel("left", "E-4 T")
            self.measuredFieldRefDiffPlot.setTitle(
                "Measured field w.r.t. reference [T]"
            )
            assert self.measuredFieldRefDiffPlot.vb is not None
            self.measuredFieldRefDiffPlot.vb.setXLink(self.measuredCurrentPlot.vb)
            self.measuredFieldRefDiffPlot.vb.setYLink(self.predictedFieldRefDiffPlot.vb)
            self.measuredFieldRefDiffPlot.vb.setYRange(-5, 5)

        self.deltaPlot.setLabel("left", "E-4 T")
        self.deltaPlot.setTitle("Reference difference (trim applied) [T]")
        assert self.deltaPlot.vb is not None
        self.deltaPlot.vb.setXLink(self.measuredCurrentPlot.vb)
        self.deltaPlot.vb.setYRange(-10, 10)

    def _add_plots(self) -> None:
        self.widget.setBackground("w")

        self.widget.addItem(self.measuredCurrentPlot, row=0, col=0)
        self.widget.addItem(self.deltaPlot, row=1, col=0)

        if self.plot_measured:
            self.widget.addItem(self.measuredFieldPlot, row=0, col=1)
            self.widget.addItem(self.measuredFieldRefDiffPlot, row=1, col=1)

            self.widget.addItem(self.predictedFieldPlot, row=0, col=2)
            self.widget.addItem(self.predictedFieldRefDiffPlot, row=1, col=2)
        else:
            self.widget.addItem(self.predictedFieldPlot, row=0, col=1)
            self.widget.addItem(self.predictedFieldRefDiffPlot, row=1, col=1)

    def _default_lines(self) -> None:
        self.predictedFieldRefDiffPlot.addItem(
            pg.InfiniteLine(pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        )
        self.deltaPlot.addItem(
            pg.InfiniteLine(pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine))
        )
        if self.plot_measured:
            assert self.measuredFieldRefDiffPlot is not None
            self.measuredFieldRefDiffPlot.addItem(
                pg.InfiniteLine(
                    pos=None, angle=0, pen=pg.mkPen(style=QtCore.Qt.DashLine)
                )
            )

    @QtCore.Slot(pg.PlotCurveItem)
    def addMeasuredCurrentPlot(self, plot: pg.PlotCurveItem) -> None:
        self.measuredCurrentPlot.addItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def addMeasuredFieldPlot(self, plot: pg.PlotCurveItem) -> None:
        if self.measuredFieldPlot is None:
            msg = "Measured field plot is not enabled. Cannot plot."
            log.error(msg)
            return
        self.measuredFieldPlot.addItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def addMeasuredFieldRefDiffPlot(self, plot: pg.PlotCurveItem) -> None:
        if self.measuredFieldRefDiffPlot is None:
            msg = "Measured field ref diff plot is not enabled. Cannot plot."
            log.error(msg)
            return
        self.measuredFieldRefDiffPlot.addItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def addPredictedFieldPlot(self, plot: pg.PlotCurveItem) -> None:
        self.predictedFieldPlot.addItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def addPredictedFieldRefDiffPlot(self, plot: pg.PlotCurveItem) -> None:
        self.predictedFieldRefDiffPlot.addItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def addDeltaPlot(self, plot: pg.PlotCurveItem) -> None:
        self.deltaPlot.addItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def removeMeasuredCurrentPlot(self, plot: pg.PlotCurveItem) -> None:
        self.measuredCurrentPlot.removeItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def removeMeasuredFieldPlot(self, plot: pg.PlotCurveItem) -> None:
        if self.measuredFieldPlot is None:
            msg = "Measured field plot is not enabled. Cannot remove plot."
            log.error(msg)
            return
        self.measuredFieldPlot.removeItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def removeMeasuredFieldRefDiffPlot(self, plot: pg.PlotCurveItem) -> None:
        if self.measuredFieldRefDiffPlot is None:
            msg = "Measured field ref diff plot is not enabled. Cannot remove plot."
            log.error(msg)
            return
        self.measuredFieldRefDiffPlot.removeItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def removePredictedFieldPlot(self, plot: pg.PlotCurveItem) -> None:
        self.predictedFieldPlot.removeItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def removePredictedFieldRefDiffPlot(self, plot: pg.PlotCurveItem) -> None:
        self.predictedFieldRefDiffPlot.removeItem(plot)

    @QtCore.Slot(pg.PlotCurveItem)
    def removeDeltaPlot(self, plot: pg.PlotCurveItem) -> None:
        self.deltaPlot.removeItem(plot)

    @QtCore.Slot(float, float)
    def setXRange(self, x: float, y: float) -> None:
        assert self.measuredCurrentPlot.vb is not None
        self.measuredCurrentPlot.vb.setXRange(x, y)


class HistoryPlotWidget(QtWidgets.QWidget, Ui_PredictionAnalysisWidget):
    def __init__(
        self,
        history: HistoryListModel,
        parent: QtWidgets.QWidget | None = None,
        *,
        plot_measured: bool = True,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)
        self.plot_measured = plot_measured

        self.lmodel = PredictionListModel(data_source=history, parent=self)
        self.listPredictions.setModel(self.lmodel)

        self.plots = PlotContainer(parent=self.widget, plot_measured=plot_measured)
        self.pmodel = PredictionPlotModel(parent=self)

        self._connect_list_model()
        self._connect_plot_model()
        self.buttonReference.clicked.connect(self._on_select_new_reference)

    def _connect_list_model(self) -> None:
        self.listPredictions.clicked.connect(self.itemClicked)
        self.listPredictions.clicked.connect(self.lmodel.clicked)
        self.lmodel.itemAdded.connect(self.pmodel.showCycle)
        self.lmodel.itemUpdated.connect(self.pmodel.updateCycle)
        self.lmodel.itemRemoved.connect(self.pmodel.removeCycle)
        self.lmodel.modelReset.connect(self.pmodel.removeAll)

    def _connect_plot_model(self) -> None:
        self.pmodel.measuredCurrentAdded.connect(self.plots.addMeasuredCurrentPlot)
        self.pmodel.measuredCurrentRemoved.connect(self.plots.removeMeasuredCurrentPlot)
        self.pmodel.measuredFieldAdded.connect(self.plots.addMeasuredFieldPlot)
        self.pmodel.measuredFieldRemoved.connect(self.plots.removeMeasuredFieldPlot)
        self.pmodel.predictedFieldAdded.connect(self.plots.addPredictedFieldPlot)
        self.pmodel.predictedFieldRemoved.connect(self.plots.removePredictedFieldPlot)
        self.pmodel.refMeasuredFieldAdded.connect(
            self.plots.addMeasuredFieldRefDiffPlot
        )
        self.pmodel.refMeasuredFieldRemoved.connect(
            self.plots.removeMeasuredFieldRefDiffPlot
        )
        self.pmodel.refPredictedFieldAdded.connect(
            self.plots.addPredictedFieldRefDiffPlot
        )
        self.pmodel.refPredictedFieldRemoved.connect(
            self.plots.removePredictedFieldRefDiffPlot
        )
        self.pmodel.deltaFieldAdded.connect(self.plots.addDeltaPlot)
        self.pmodel.deltaFieldRemoved.connect(self.plots.removeDeltaPlot)

        self.pmodel.setXRange.connect(self.plots.setXRange)

    @QtCore.Slot(QtCore.QModelIndex)
    def itemClicked(self, index: QtCore.QModelIndex) -> None:
        item = self.lmodel.itemAt(index)
        if item.is_shown:
            self.pmodel.removeCycle(item)
        else:
            self.pmodel.showCycle(item)

    @QtCore.Slot()
    def resetAxes(self) -> None:
        self.model.plot_model.resetAxes.emit()

    def _on_select_new_reference(self) -> None:
        """
        Select a new reference cycle for plots, use a QIdentityProxyModel,
        but select only one.
        """
        dialog = ReferenceSelectorDialog(model=self.lmodel, parent=self)

        def on_dialog_accepted() -> None:
            selected_index = dialog.selected_item
            if selected_index is not None:
                item = self.lmodel.itemAt(selected_index)

                self.pmodel.setReference(item)
                self.lmodel.setReference(item)

        dialog.accepted.connect(on_dialog_accepted)

        dialog.open()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.windowClosed.emit()
        event.accept()
