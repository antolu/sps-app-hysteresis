from __future__ import annotations

import traceback
import typing

from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication

from ._threadutil import run_in_main_thread


@run_in_main_thread
def set_cursor_busy() -> None:
    QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))


@run_in_main_thread
def set_cursor_normal() -> None:
    QApplication.restoreOverrideCursor()


class load_cursor:
    """
    Convenience class for showing a loading cursor while doing some time
    intensive task.
    """

    def __enter__(self) -> load_cursor:
        set_cursor_busy()

        return self

    def __exit__(self, exc_type: typing.Type[Exception], exc_val: Exception, exc_tb: traceback.FrameSummary) -> None:  # type: ignore
        set_cursor_normal()
