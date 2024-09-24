import logging
import sys
from argparse import ArgumentParser

import pyrbac
import torch
from accwidgets.qt import exec_app_interruptable
from qtpy import QtWidgets
from rich.logging import RichHandler

from . import __version__
from ._data_flow import LocalDataFlow
from .main_window import MainWindow

torch.set_float32_matmul_precision("high")


def setup_logger(logging_level: int = 0) -> None:
    log = logging.getLogger()

    ch = RichHandler()

    formatter = logging.Formatter(
        "%(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)

    if logging_level >= 2:
        log.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    elif logging_level >= 1:
        log.setLevel(logging.INFO)
        log.setLevel(logging.INFO)
    else:
        log.setLevel(logging.WARNING)
        log.setLevel(logging.WARNING)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PyQt5.uic").setLevel(logging.WARNING)
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("cern").setLevel(logging.WARNING)
    logging.getLogger("pyda").setLevel(logging.WARNING)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    parser.add_argument(
        "-b",
        "--buffer-size",
        type=int,
        default=150000,
        dest="buffer_size",
        help="Buffer size for acquisition.",
    )
    parser.add_argument(
        "--lsa-server",
        dest="lsa_server",
        choices=["sps", "next"],
        default="next",
        help="LSA server to use.",
    )
    args = parser.parse_args()
    setup_logger(args.verbose)

    application = QtWidgets.QApplication([])
    application.setApplicationVersion(__version__)
    application.setOrganizationName("CERN")
    application.setOrganizationDomain("cern.ch")
    application.setApplicationName("SPS Hysteresis Prediction")

    from op_app_context import context, settings

    context.lsa_server = args.lsa_server
    settings.configure_application(application)

    try:
        rbac_token = pyrbac.AuthenticationClient().login_location()
        context.set_rbac_token(rbac_token)

        logging.getLogger(__name__).info(f"Logged in as {rbac_token.username}")
    except:  # noqa: E722
        pass

    data_flow = LocalDataFlow(
        buffer_size=args.buffer_size,
        provider=context.japc_provider,
        parent=application,
    )

    main_window = MainWindow(data_flow)
    main_window.show()

    data_flow.start()

    sys.exit(exec_app_interruptable(application))
