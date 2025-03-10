"""
Miscellaneous idgets for the hysteresis prediction app.
"""

from __future__ import annotations

from enum import Enum

from qtpy import QtCore, QtWidgets


class ToggleButton(QtWidgets.QPushButton):
    class State(Enum):
        STATE1 = 0
        STATE2 = 1

    stateChanged = QtCore.Signal(State)
    """ Signal emitted when the state of the button changes. """

    state1Activated = QtCore.Signal()
    state2Activated = QtCore.Signal()

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self._state = self.State.STATE1
        self.clicked.connect(self._on_clicked)

    def initializeState(
        self,
        label_s1: str,
        label_s2: str | None = None,
        initial_state: State | None = None,
    ) -> None:
        """Set the labels for the two states of the button."""
        self._label_s1 = label_s1
        self._label_s2 = label_s2 or label_s1

        self._state = initial_state or self.State.STATE1

        if self._state == self.State.STATE1:
            self.setText(self._label_s1)
        else:
            self.setText(self._label_s2)

    def _on_clicked(self) -> None:
        if self._state == self.State.STATE1:
            self._state = self.State.STATE2
            self.setText(self._label_s2)
            self.state2Activated.emit()
        else:
            self._state = self.State.STATE1
            self.setText(self._label_s1)
            self.state1Activated.emit()

        self.stateChanged.emit(self._state)

    @QtCore.Slot(State)
    def setState(self, state: State) -> None:
        if state == self.State.STATE1:
            self._state = self.State.STATE1
            self.setText(self._label_s1)
        else:
            self._state = self.State.STATE2
            self.setText(self._label_s2)

    @property
    def state(self) -> State:
        return self._state
