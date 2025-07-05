from __future__ import annotations

import logging

import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from ...generated.prediction_history_widget_ui import Ui_PredictionAnalysisWidget
from ._plot_model import UnifiedPlotModel
from ._unified_model import CycleListModel

log = logging.getLogger(__package__)


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
        self.deltaPlot.vb.setYRange(-4, 4)

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


class UnifiedHistoryPlotWidget(QtWidgets.QWidget, Ui_PredictionAnalysisWidget):
    """
    Unified history plot widget using CycleListModel and UnifiedPlotModel.

    This simplified version eliminates the HistoryListModel→PredictionListModel
    chain and uses the unified architecture directly.
    """

    def __init__(
        self,
        cycle_model: CycleListModel,
        parent: QtWidgets.QWidget | None = None,
        *,
        plot_measured: bool = True,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)
        self.plot_measured = plot_measured

        # Use the unified model directly - much simpler!
        self.cycle_model = cycle_model

        # Plot container handles the actual plotting widgets
        self.plots = PlotContainer(parent=self.widget, plot_measured=plot_measured)

        # Unified plot model using adapter pattern
        self.plot_model = UnifiedPlotModel(cycle_model=self.cycle_model, parent=self)

        # Connect the signals
        self._connect_plot_model()

    def _connect_plot_model(self) -> None:
        """Connect plot model signals to plot container."""
        # Plot addition signals - direct connections, no complex chains!
        self.plot_model.measuredCurrentAdded.connect(self.plots.addMeasuredCurrentPlot)
        self.plot_model.measuredFieldAdded.connect(self.plots.addMeasuredFieldPlot)
        self.plot_model.predictedFieldAdded.connect(self.plots.addPredictedFieldPlot)
        self.plot_model.deltaFieldAdded.connect(self.plots.addDeltaPlot)
        self.plot_model.refMeasuredFieldAdded.connect(
            self.plots.addMeasuredFieldRefDiffPlot
        )
        self.plot_model.refPredictedFieldAdded.connect(
            self.plots.addPredictedFieldRefDiffPlot
        )

        # Plot removal signals
        self.plot_model.measuredCurrentRemoved.connect(
            self.plots.removeMeasuredCurrentPlot
        )
        self.plot_model.measuredFieldRemoved.connect(self.plots.removeMeasuredFieldPlot)
        self.plot_model.predictedFieldRemoved.connect(
            self.plots.removePredictedFieldPlot
        )
        self.plot_model.deltaFieldRemoved.connect(self.plots.removeDeltaPlot)
        self.plot_model.refMeasuredFieldRemoved.connect(
            self.plots.removeMeasuredFieldRefDiffPlot
        )
        self.plot_model.refPredictedFieldRemoved.connect(
            self.plots.removePredictedFieldRefDiffPlot
        )

        # Axis control
        self.plot_model.setXRange.connect(self.plots.setXRange)

    @QtCore.Slot(QtCore.QModelIndex)
    def itemClicked(self, index: QtCore.QModelIndex) -> None:
        """Handle list item click - much simpler now!"""
        # The unified model handles everything - no complex coordination needed
        self.cycle_model.clicked(index)
        log.debug(f"Item clicked at row {index.row()}")

    @QtCore.Slot()
    def resetAxes(self) -> None:
        """Reset plot axes."""
        self.plot_model.reset_axes()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle widget close."""
        # Clean up plots
        self.plot_model.hide_all()
        event.accept()


# Provide backward compatibility alias
HistoryPlotWidget = UnifiedHistoryPlotWidget
