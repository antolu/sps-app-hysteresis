import logging
import os.path
import sys
from argparse import ArgumentParser

import pyrbac
import torch
from accwidgets.qt import exec_app_interruptable
from qtpy import QtCore, QtWidgets
from rich.logging import RichHandler

from . import __version__
from .contexts import app_context, set_context
from .flow import LocalFlowWorker, UcapFlowWorker
from .io.metrics import TensorboardWriter, TextWriter
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

    set_module_logging("pyda", logging.WARNING)
    set_module_logging("pyccda", logging.WARNING)
    set_module_logging("torch", logging.WARNING)


def add_file_handler(
    logger: logging.Logger,
    log_file: str,
    level: int = logging.DEBUG,
    formatter: logging.Formatter | None = None,
) -> None:
    """
    Add a file handler to the logger.

    Args:
        logger (logging.Logger): The logger to add the file handler to.
        log_file (str): The path to the log file.
        level (int, optional): The logging level for the file handler. Defaults to logging.DEBUG.
        formatter (logging.Formatter, optional): The formatter for the file handler. Defaults to None.
    """
    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def set_module_logging(pattern: str, log_level: int = logging.WARNING) -> None:
    for name in logging.root.manager.loggerDict:
        if name.startswith(pattern):
            logging.getLogger(name).setLevel(log_level)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("-v", dest="verbose", action="count", default=0)
    parser.add_argument(
        "-b",
        "--buffer-size",
        dest="buffer_size",
        type=int,
        default=60000,
        help="Buffer size for acquisition.",
    )
    parser.add_argument(
        "-d",
        "--device",
        choices=["MBI", "QF", "QD"],
        required=True,
        help="Device to apply field compensation to. Available magnetic circuits are SPS main dipoles (MBI), SPS focusing quadrupoles (QF) and SPS defocusing quadrupoles (QD).",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Online prediction and trim monitoring. No predictions are done locally.",
    )
    parser.add_argument(
        "--lsa-server",
        dest="lsa_server",
        choices=["sps", "next"],
        default="next",
        help="LSA server to use.",
    )
    parser.add_argument(
        "--logdir",
        dest="logdir",
        default=None,
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--metrics-writer",
        dest="metrics_writer",
        choices=["txt", "tensorboard"],
        default="txt",
        help="Metrics writer to use.",
    )

    args = parser.parse_args()
    setup_logger(args.verbose)

    # Set up logging to a file in the log directory
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir, exist_ok=True)
    log_file = os.path.join(args.logdir, "application.log")
    add_file_handler(
        logging.getLogger(),
        log_file,
        level=logging.DEBUG,
        formatter=logging.Formatter(
            "%(asctime)s %(levelname)s %(message)s",
            "%Y-%m-%d %H:%M:%S",
        ),
    )

    application = QtWidgets.QApplication([])
    application.setApplicationVersion(__version__)
    application.setOrganizationName("CERN")
    application.setOrganizationDomain("cern.ch")
    application.setApplicationName("SPS Hysteresis Prediction")

    from op_app_context import context, settings  # noqa: PLC0415

    context.lsa_server = args.lsa_server
    settings.configure_application(application)

    set_context(args.device, online=args.online)

    app_context().LOGDIR = args.logdir

    try:
        rbac_token = pyrbac.AuthenticationClient().login_location()
        context.set_rbac_token(rbac_token)

        logging.getLogger(__name__).info(f"Logged in as {rbac_token.user_name}")
    except:  # noqa: E722
        logging.getLogger(__name__).exception("Failed to login with RBAC.")
        logging.getLogger(__name__).warning(
            "No RBAC by location, you will have to login manually."
        )

    data_thread = QtCore.QThread()
    if not args.online:
        assert not app_context().ONLINE
        flow_worker = LocalFlowWorker(
            buffer_size=args.buffer_size,
            provider=context.japc_provider,
            meas_b_avail=app_context().B_MEAS_AVAIL,
        )
    else:
        ucap_params = app_context().UCAP_PARAMS
        if ucap_params is None:
            msg = "UCAP parameters not available for this device."
            raise ValueError(msg)
        flow_worker = UcapFlowWorker(
            provider=context.japc_provider,
        )

    flow_worker.moveToThread(data_thread)
    flow_worker.init_data_flow()

    data_thread.started.connect(flow_worker.start)

    # quit the worker when the application is about to quit
    application.aboutToQuit.connect(flow_worker.stop)
    application.aboutToQuit.connect(data_thread.quit)
    application.aboutToQuit.connect(data_thread.wait)

    writer = (
        TextWriter(output_dir=app_context().LOGDIR)
        if args.metrics_writer == "txt"
        else TensorboardWriter(output_dir=app_context().LOGDIR)
    )

    flow_worker.data_flow.onMetricsAvailable.connect(writer.onNewMetrics)

    main_window: MainWindow | None = None

    def exit_if_fail() -> None:
        if main_window is None:
            sys.exit(1)

    data_thread.finished.connect(exit_if_fail)
    data_thread.start()

    main_window = MainWindow(data_flow=flow_worker.data_flow, parent=None)
    main_window.show()

    sys.exit(exec_app_interruptable(application))
