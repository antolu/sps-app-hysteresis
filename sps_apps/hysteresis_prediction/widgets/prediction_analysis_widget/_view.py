"""
This file contains the view for the prediction analysis widget.
"""
from __future__ import annotations

import logging

from qtpy.QtWidgets import QMenuBar, QWidget

from ...generated.prediction_analysis_widget_ui import (
    Ui_PredictionAnalysisWidget,
)
from ._model import PredictionAnalysisModel

log = logging.getLogger(__name__)


class PredictionAnalysisWidget(QWidget, Ui_PredictionAnalysisWidget):
    def __init__(
        self,
        model: PredictionAnalysisModel | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.setupUi(self)

        self._model: PredictionAnalysisModel | None = None
        self.model = model or PredictionAnalysisModel()

        self.menubar = QMenuBar(self)
        self.horizontalLayout.setMenuBar(self.menubar)

        file_menu = self.menubar.addMenu("&File")
        file_menu.addAction(self.actionExport_Predictions)
        file_menu.addSeparator()
        file_menu.addAction(self.actionExit)

        tools_menu = self.menubar.addMenu("&Tools")
        tools_menu.addAction(self.actionShow_GT)
        tools_menu.addAction(self.actionClear_Buffer)

        self._connect_slots()

    def _connect_slots(self) -> None:
        self.spinBoxNumPredictions.editingFinished.connect(
            lambda *_: self.model.set_max_buffer_samples(
                self.spinBoxNumPredictions.value()
            )
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
            model.maxBufferSizeChanged.emit
        )

    def _disconnect_model(self, model: PredictionAnalysisModel) -> None:
        ...
