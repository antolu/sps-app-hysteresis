from __future__ import annotations

import logging
from typing import Optional

import pyqtgraph as pg
from accwidgets.graph import (
    ExPlotWidgetConfig,
    LivePlotCurve,
    PlotWidgetStyle,
    TimeSpan,
)
from accwidgets.graph.widgets.plotwidget import ScrollingPlotWidget
from qtpy.QtWidgets import QVBoxLayout, QWidget

from ._model import PlotModel
from ._sources import LocalTimerTimingSource

log = logging.getLogger(__name__)


AXES_RANGE_KWARGS = {"field": (0.0, 2.2), "current": (0.0, 6000)}


class PlotWidget(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._model: Optional[PlotModel] = None
        self._curves: tuple[set[LivePlotCurve], ...] = (set(), set())

        layout = QVBoxLayout(self)

        self._timing_source = LocalTimerTimingSource()

        self.plotDiscr = ScrollingPlotWidget(
            parent=self,
            time_span=60,
            timing_source=self._timing_source,
            time_progress_line=True,
        )
        self.plotCurField = ScrollingPlotWidget(
            parent=self,
            time_span=60,
            timing_source=self._timing_source,
            time_progress_line=True,
        )

        layout.addWidget(self.plotDiscr)
        layout.addWidget(self.plotCurField)

        self._setup_plots()

    def reset_axes_range(self) -> None:
        self.plotDiscr.autoRange()
        self.plotCurField.autoRange()

        self.plotDiscr.setRange(dpp_fixed=(-5, 5))  # noqa
        self.plotCurField.setRange(**AXES_RANGE_KWARGS)  # noqa

    def set_time_span(self, min: int, max: int) -> None:
        """
        Set the time span of the plot. From min to max.
        Relative to the latest time.
        """
        time_span = TimeSpan(left=min, right=max)
        plot_config = ExPlotWidgetConfig(
            plotting_style=PlotWidgetStyle.SCROLLING_PLOT,
            time_progress_line=True,
            time_span=time_span,
        )
        self.plotCurField.update_config(config=plot_config)
        self.plotDiscr.update_config(config=plot_config)

        self.plotCurField.setRange(**AXES_RANGE_KWARGS)  # noqa

    def _setup_plots(self) -> None:
        # self.plotCurField.setXLink(self.plotDiscr)
        self.plotCurField.addLegend()
        self.plotDiscr.addLegend()

        self.plotCurField.add_layer(layer_id="current", unit="A")
        self.plotCurField.add_layer(layer_id="field", unit="T")
        self.plotCurField.hideAxis("left")

        self.plotDiscr.add_layer(layer_id="dpp_fixed")
        # self.plotDiscr.add_layer(layer_id="dpp_dynamic")
        self.plotDiscr.hideAxis("left")
        self.plotCurField.setXLink(self.plotDiscr)

        self.reset_axes_range()

    def _set_model(self, model: PlotModel) -> None:
        if self._model is not None:
            self._disconnect_model(self._model)

        self._connect_model(model)

        self._model = model

    def _get_model(self) -> Optional[PlotModel]:
        return self._model

    model = property(_get_model, _set_model)

    def _connect_model(self, model: PlotModel) -> None:
        current_meas = self.plotCurField.addCurve(
            data_source=model.current_meas_source,
            layer="current",
            pen=pg.mkPen(color="#96939B", width=2),
            unit="A",
            name="Measured I",
        )
        field_meas = self.plotCurField.addCurve(
            data_source=model.field_meas_source,
            layer="field",
            pen=pg.mkPen(color="#1F5673", width=2),
            unit="T",
            name="Measured B",
        )
        current_prog = self.plotCurField.addCurve(
            data_source=model.current_prog_source,
            layer="current",
            pen=pg.mkPen(color="#E8E8E8", width=2),
            unit="A",
            name="Programmed I",
        )
        field_pred = self.plotCurField.addCurve(
            data_source=model.field_predict_source,
            layer="field",
            pen=pg.mkPen(color="#FC814A", width=2),
            unit="T",
            name="Predicted B",
        )
        field_ref_disc = self.plotDiscr.addCurve(
            data_source=model.field_ref_dpp_source,
            pen=pg.mkPen(color="#BFBFBF", width=2),
            layer="dpp_fixed",
            name="dp/p w.r.t. Ref.",
            unit="E-4",
        )
        field_meas_discr = self.plotDiscr.addCurve(
            data_source=model.field_meas_dpp_source,
            pen=pg.mkPen(color="#1F5673", width=2),
            layer="dpp_fixed",
            name="dp/p w.r.t. Meas.",
            unit="E-4",
        )
        # field_ref_disc = self.plotDiscr.addCurve(
        #     data_source=model.field_ref_dpp_source,
        #     pen=pg.mkPen(color="#BFBFBF", width=2),
        #     layer="dpp_dynamic",
        #     name="dp/p w.r.t. Ref.",
        # )
        # field_meas_discr = self.plotDiscr.addCurve(
        #     data_source=model.field_meas_dpp_source,
        #     pen=pg.mkPen(color="#1F5673", width=2),
        #     layer="dpp_dynamic",
        #     name="dp/p w.r.t. Meas.",
        # )

        for curve in (current_meas, field_meas, current_prog, field_pred):
            self._curves[1].add(curve)

        for curve in (field_ref_disc, field_meas_discr):
            self._curves[0].add(curve)

    def _disconnect_model(self, model: PlotModel) -> None:
        for curve in self._curves[0]:
            self.plotDiscr.removeItem(curve)
        for curve in self._curves[1]:
            self.plotCurField.removeItem(curve)

        self._curves[0].clear()
        self._curves[1].clear()
