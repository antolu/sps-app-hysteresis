from typing import Optional

from accwidgets.app_frame import ApplicationFrame
from accwidgets.log_console import LogConsole
from accwidgets.timing_bar import TimingBar, TimingBarDomain, TimingBarModel
from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .generated.main_window_ui import Ui_main_window
from .settings import context


class MainWindow(Ui_main_window, ApplicationFrame):
    def __init__(self, parent: Optional[QWidget] = None):
        ApplicationFrame.__init__(self, parent)
        Ui_main_window.__init__(self)

        self.setupUi(self)

        log_console = LogConsole(self)
        self.log_console = log_console
        log_console.toggleExpandedMode()

        timing_model = TimingBarModel(
            domain=TimingBarDomain.SPS, japc=context.japc
        )
        timing_bar = TimingBar(self, model=timing_model)
        self.timing_bar = timing_bar
        # self.centralWidget().layout().addWidget(timing_bar)
