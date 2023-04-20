from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from accwidgets.app_frame import ApplicationFrame
from accwidgets.graph import (
    CurveData,
    ScrollingPlotWidget,
    TimeSpan,
    UpdateSource,
)
from accwidgets.qt import exec_app_interruptable
from qtpy.QtCore import QTimer
from qtpy.QtGui import QCloseEvent, QPainter
from qtpy.QtWidgets import QApplication

from sps_apps.hysteresis_prediction.data import Acquisition, SingleCycleData

log = logging.getLogger()


class LocalTimerTimingSource(UpdateSource):
    def __init__(self, offset: float = 0.0):
        """
        Class for sending timing-update signals based on a QTimer instance.

        Args:
            offset: offset of the emitted time to the actual current time
        """
        super().__init__()
        self.timer = QTimer(self)
        self.offset = offset
        self.timer.timeout.connect(self._create_new_value)
        self.timer.start(int(1000 / 30))

    def _create_new_value(self) -> None:
        self.sig_new_timestamp.emit(datetime.now().timestamp() + self.offset)


class MainWindow(ApplicationFrame):
    def __init__(self, *args, **kwargs) -> None:  # type: ignore
        super().__init__(*args, **kwargs)

        self.acq = Acquisition(min_buffer_size=150000)
        self.acq.new_measured_data.connect(self.new_measured_handler)
        self.acq.buffer.new_programmed_cycle.connect(
            self.new_programmed_handler
        )

        self.plot = ScrollingPlotWidget(
            parent=self,
            timing_source=LocalTimerTimingSource(),
            time_span=TimeSpan(left=30),
            time_progress_line=True,
        )
        self.plot.add_layer(layer_id="current", unit="A")
        self.plot.add_layer(layer_id="field", unit="T")
        self.plot.hideAxis("left")
        self.plot.setRenderHint(QPainter.Antialiasing)

        self._measured_field_source = UpdateSource()
        self._measured_current_source = UpdateSource()

        self._programmed_field_source = UpdateSource()
        self._programmed_current_source = UpdateSource()

        self.plot.addLegend()
        self.plot.addCurve(
            data_source=self._measured_current_source,
            layer="current",
            pen=pg.mkPen(color="#96939B", width=2),
            unit="A",
            name="Measured I",
        )
        self.plot.addCurve(
            data_source=self._programmed_current_source,
            layer="current",
            pen=pg.mkPen(color="#E8E8E8", width=2),
            unit="A",
            name="Programmed I",
        )
        self.plot.addCurve(
            data_source=self._measured_field_source,
            layer="field",
            pen=pg.mkPen(color="#1F5673", width=2),
            unit="T",
            name="Measured B",
        )
        # predicted field color: #FC814A, #BFBFBF
        # self.plot.addCurve(
        #     data_source=self._programmed_field_source,
        #     layer="field",
        #     pen="y",
        #     name="Programmed B",
        # )
        # self.plot.addTimestampMarker(data_source=self.time_line)
        # self.plot.enableAutoRange()
        self.plot.setRange(field=(-0.1, 2.2), current=(-0.045 * 7000, 7000))
        self.setCentralWidget(self.plot)

        self.resize(1000, 400)

    def show(self) -> None:
        super().show()
        self.th = self.acq.run()

    def new_measured_handler(self, cycle_data: SingleCycleData) -> None:
        if cycle_data.field_meas is None or cycle_data.current_meas is None:
            log.error("Received field or data is None.")
            return

        measured_field = cycle_data.field_meas.flatten()
        measured_current = cycle_data.current_meas.flatten()

        time_range = (
            np.arange(cycle_data.num_samples) / 1e3
            + cycle_data.cycle_timestamp / 1e9
        )
        field_data = CurveData(x=time_range[::10], y=measured_field[::10])
        current_data = CurveData(x=time_range[::10], y=measured_current[::10])
        self._measured_field_source.send_data(field_data)
        self._measured_current_source.send_data(current_data)

    def new_programmed_handler(self, cycle_data: SingleCycleData) -> None:
        time_current, programmed_current = cycle_data.current_prog
        # field_time, programmed_field = cycle_data.field_prog

        time_current = time_current / 1e3 + cycle_data.cycle_timestamp / 1e9

        time_interp = (
            np.arange(cycle_data.num_samples) / 1e3
            + cycle_data.cycle_timestamp / 1e9
        )

        programmed_current_interp = np.interp(
            time_interp, time_current, programmed_current
        )

        # time_field = field_time / 1e3 + cycle_data.cycle_timestamp
        # field_data = CurveData(x=time_field, y=programmed_field[::10])
        current_data = CurveData(x=time_interp, y=programmed_current_interp)
        # self._programmed_field_source.send_data(field_data)
        self._programmed_current_source.send_data(current_data)

    def closeEvent(self, event: QCloseEvent) -> None:
        self.acq.stop()
        event.accept()


def buffer_handler(buffer: list[SingleCycleData]) -> None:
    msg = "\n".join(
        [f"{cycle} [-> {cycle.num_samples} ms]" for cycle in buffer]
    )
    log.info(f"Buffer handler called with {len(buffer)} elements.\n" + msg)


def setup_logging() -> None:
    log = logging.getLogger()
    log.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    log.addHandler(ch)


def main() -> None:
    setup_logging()

    app = QApplication([])
    win = MainWindow()
    win.show()

    exit(exec_app_interruptable(app))


if __name__ == "__main__":
    main()
