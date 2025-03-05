from __future__ import annotations

import types
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


class load_cursor:  # noqa: N801
    """
    Convenience class for showing a loading cursor while doing some time
    intensive task.
    """

    def __enter__(self) -> typing.Self:
        set_cursor_busy()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:  # type: ignore
        set_cursor_normal()
