from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget

from ...generated.plot_settings_widget_ui import Ui_PlotSettingsWidget

log = logging.getLogger(__name__)


class PlotSettingsWidget(Ui_PlotSettingsWidget, QWidget):
    timespan_changed = Signal(int, int)  # min, max

    def __init__(self, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)

        self.spinBoxTimespan.valueChanged.connect(self._timespan_changed)
        self.buttonResetAxis.clicked.connect(self._timespan_changed)

    def _timespan_changed(self, *_) -> None:
        self.timespan_changed.emit(self.spinBoxTimespan.value(), 0)
