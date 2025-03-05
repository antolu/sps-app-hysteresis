from __future__ import annotations

import random
from collections import deque

import pyqtgraph as pg
from qtpy import QtGui


class ColorPool:
    CM = pg.colormap.getFromMatplotlib("tab20")

    def __init__(self) -> None:
        assert self.CM is not None
        color_list = list(self.CM.getColors())
        random.shuffle(color_list)
        self._colors: deque[QtGui.QColor] = deque(
            [QtGui.QColor(*val) for val in color_list], maxlen=20
        )

    def get_color(self) -> QtGui.QColor:
        return self._colors.popleft()

    def return_color(self, color: QtGui.QColor) -> None:
        self._colors.append(color)
