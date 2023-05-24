from __future__ import annotations

import logging

from qtpy.QtCore import QObject, Signal

from .plot_settings_widget import AppStatus

log = logging.getLogger(__name__)


class StatusManager(QObject):
    setStatus = Signal(AppStatus)
    """Signal emits when the status is updated. """

    statusChanged = Signal(AppStatus)
    """ Send information to this signal to change global status. """

    def __init__(self, parent: QObject = None):
        super().__init__(parent)

        self._current_state: set[AppStatus] = set()

        self.statusChanged.connect(self._on_status_changed)

    def _on_status_changed(self, status: AppStatus) -> None:  # noqa: C901
        """
        Update the status and emit a signal if it has changed.
        """

        if status in self._current_state:
            return

        if status == AppStatus.INFERENCE_IDLE:
            if AppStatus.INFERENCE_RUNNING in self._current_state:
                self._current_state.remove(AppStatus.INFERENCE_RUNNING)

            if AppStatus.BUFFER_WAITING in self._current_state:
                self.setStatus.emit(AppStatus.BUFFER_WAITING)
            else:
                self.setStatus.emit(status)

        elif status == AppStatus.INFERENCE_RUNNING:
            if AppStatus.INFERENCE_IDLE in self._current_state:
                self._current_state.remove(AppStatus.INFERENCE_IDLE)
                self.setStatus.emit(status)

        elif status == AppStatus.INFERENCE_ENABLED:
            if AppStatus.INFERENCE_DISABLED in self._current_state:
                self._current_state.remove(AppStatus.INFERENCE_DISABLED)

                if AppStatus.BUFFER_WAITING in self._current_state:
                    self.setStatus.emit(AppStatus.BUFFER_WAITING)
                else:
                    self.setStatus.emit(AppStatus.INFERENCE_IDLE)
            if AppStatus.INFERENCE_IDLE in self._current_state:
                self.setStatus.emit(AppStatus.INFERENCE_IDLE)

        elif status == AppStatus.INFERENCE_DISABLED:
            if AppStatus.INFERENCE_ENABLED in self._current_state:
                self._current_state.remove(AppStatus.INFERENCE_ENABLED)
                self.setStatus.emit(status)

        elif status == AppStatus.BUFFER_WAITING:
            if AppStatus.BUFFER_FULL in self._current_state:
                self._current_state.remove(AppStatus.BUFFER_FULL)

        elif status == AppStatus.BUFFER_FULL:
            if AppStatus.BUFFER_WAITING in self._current_state:
                self._current_state.remove(AppStatus.BUFFER_WAITING)
                self._current_state.add(AppStatus.INFERENCE_IDLE)

                if AppStatus.INFERENCE_ENABLED in self._current_state:
                    self.setStatus.emit(AppStatus.INFERENCE_IDLE)
                else:
                    self.setStatus.emit(AppStatus.INFERENCE_DISABLED)
            if AppStatus.MODEL_LOADED not in self._current_state:
                self.setStatus.emit(AppStatus.NO_MODEL)

        elif status == AppStatus.MODEL_LOADED:
            self._current_state.add(status)
            if AppStatus.BUFFER_FULL in self._current_state:
                if AppStatus.INFERENCE_ENABLED in self._current_state:
                    self._current_state.add(AppStatus.INFERENCE_IDLE)
                    self.setStatus.emit(AppStatus.INFERENCE_IDLE)
                else:
                    self._current_state.add(AppStatus.INFERENCE_DISABLED)
                    self.setStatus.emit(AppStatus.INFERENCE_DISABLED)
            elif AppStatus.BUFFER_WAITING in self._current_state:
                self.setStatus.emit(AppStatus.BUFFER_WAITING)

        elif status == AppStatus.NO_MODEL:  # start of application
            self._current_state.clear()
            self._current_state.add(AppStatus.NO_MODEL)
            self._current_state.add(AppStatus.INFERENCE_DISABLED)
            self._current_state.add(AppStatus.BUFFER_WAITING)
            self.setStatus.emit(status)
        else:
            raise NotImplementedError(f"Unknown status: {status}")

        self._current_state.add(status)
