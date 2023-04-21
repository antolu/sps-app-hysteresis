import logging
import sys

from accwidgets.qt import exec_app_interruptable
from PyQt5.QtWidgets import QApplication

from . import __version__
from .main_window import MainWindow


def setup_logger(logging_level: int = 0) -> None:
    log = logging.getLogger()

    if logging_level >= 1:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.WARNING)

    ch = logging.StreamHandler()

    formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    log.addHandler(ch)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PyQt5.uic").setLevel(logging.WARNING)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("cern").setLevel(logging.WARNING)


def main() -> None:
    setup_logger()

    application = QApplication([])
    application.setApplicationVersion(__version__)
    application.setOrganizationName("CERN")
    application.setOrganizationDomain("cern.ch")
    application.setApplicationName("SPS Hysteresis Prediction")

    main_window = MainWindow()
    main_window.show()

    sys.exit(exec_app_interruptable(application))
