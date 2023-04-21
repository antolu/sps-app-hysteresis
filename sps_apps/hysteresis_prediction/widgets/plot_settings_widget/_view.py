from __future__ import annotations

import logging
from typing import Optional

from qtpy.QtWidgets import QWidget

from ...generated.plot_settings_widget_ui import Ui_PlotSettingsWidget

log = logging.getLogger(__name__)


class PlotSettingsWidget(Ui_PlotSettingsWidget, QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        QWidget.__init__(self, parent=parent)

        self.setupUi(self)
