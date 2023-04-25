from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication


class load_cursor:
    """
    Convenience class for showing a loading cursor while doing some time
    intensive task.
    """

    def __enter__(self) -> load_cursor:
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        QApplication.restoreOverrideCursor()
