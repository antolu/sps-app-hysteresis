from __future__ import annotations

import logging
import time
from threading import Thread

from qtpy.QtWidgets import QApplication

from sps_apps.hysteresis_prediction.data import SingleCycleData
from sps_apps.hysteresis_prediction.widgets.prediction_analysis_widget import (
    PredictionAnalysisModel,
    PredictionAnalysisWidget,
)
from test_scripts.test_predict import OUTPUT_PATH, load_buffers, setup_logging

log = logging.getLogger()


SELECTOR = "SPS.USER.HIRADMT1"


def main() -> None:
    buffers = load_buffers(OUTPUT_PATH)

    application = QApplication([])

    model = PredictionAnalysisModel()
    model.selector = SELECTOR
    widget = PredictionAnalysisWidget(model=model)
    widget.show()

    def add_buffer_to_model(buffer_list: list[list[SingleCycleData]]) -> None:
        time.sleep(1)
        for buffer in buffer_list:
            if buffer[-1].field_pred is None:
                raise ValueError("Buffer does not contain predictions.")

            log.debug("Adding buffer for %s", buffer[-1].user)
            time.sleep(0.5)
            model.newData.emit(buffer[-1])

    thread = Thread(target=add_buffer_to_model, args=(buffers,))
    thread.start()

    application.exec_()


if __name__ == "__main__":
    setup_logging()
    main()
