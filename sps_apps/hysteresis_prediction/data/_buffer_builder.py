from __future__ import annotations

import datetime
import logging

import hystcomp_utils.cycle_data
import numpy as np
import pyda
import pyda.data
from qtpy import QtCore

from ._buffer import AcquisitionBuffer, InsufficientDataError
from ._event_builder_abc import EventBuilderAbc

log = logging.getLogger(__name__)


PARAM_I_PROG = "rmi://virtual_sps/MBI/IREF"
PARAM_B_PROG = "rmi://virtual_sps/SPSBEAM/B"
PARAM_BHYS_CORRECTION = "rda3://UCAP-NODE-SPS-HYSTCOMP-TEST/BHYS-CORRECTION/Acquisition"


class BufferEventbuilder(EventBuilderAbc):
    newBufferAvailable = QtCore.Signal(list[hystcomp_utils.cycle_data.CycleData])
    newEcoBufferAvailable = QtCore.Signal(hystcomp_utils.cycle_data.CycleData)

    def __init__(
        self,
        buffer_size: int = 60000,  # ms
        *,
        parent: QtCore.QObject | None = None,
    ):
        super().__init__(
            [],
            parent=parent,
        )

        self._buffer_size = buffer_size
        self._acquisition_buffer = AcquisitionBuffer(buffer_size)

    def _handle_acquisition_impl(
        self, fspv: pyda.data.PropertyRetrievalResponse
    ) -> None:
        # no need to handle acquisition
        msg = f"{self.__class__.__name__} does not subscribe to triggers."
        raise NotImplementedError(msg)

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewCycleData(self, cycle_data: hystcomp_utils.cycle_data.CycleData) -> None:
        msg = f"[{cycle_data}]: Received forewarning."
        log.debug(msg)

        try:
            self._acquisition_buffer.new_cycle(cycle_data)
        except:  # noqa: E722
            msg = f"[{cycle_data}]: Error appending cycle to buffer."
            log.exception(msg)

        try:
            buffer = self._acquisition_buffer.collate_samples()
        except InsufficientDataError:
            size = len(self._acquisition_buffer)
            total = self._buffer_size

            msg = f"[{cycle_data}]: Insufficient data in buffer: {size}/{total}."
            log.info(msg)

        self.newBufferAvailable.emit(buffer)

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewEcoCycleData(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        msg = f"[{cycle_data}]: Received ECO cycle data."
        log.debug(msg)

        now = datetime.datetime.now().timestamp() * 1e9
        delta_s = (now - cycle_data.cycle_timestamp) / 1e9
        msg = f"[{cycle_data}]: Received ECO cycle data {delta_s:.3f} s after start."
        log.info(msg)

        self._acquisition_buffer.update_cycle(cycle_data)
        buffer_l = self._acquisition_buffer.collate_samples()

        if not np.allclose(buffer_l[-1].cycle_timestamp, cycle_data.cycle_timestamp):
            msg = (
                f"{cycle_data} Last cycle in buffer has timestamp {buffer_l[-1].cycle_timestamp}, "
                f"but received ECO cycle {cycle_data.cycle_timestamp}"
            )
            log.error(msg)
            return

        msg = f"[{cycle_data}]: Appended ECO cycle to buffer and publishing."

        log.info(msg)

        self.newEcoBufferAvailable.emit(buffer_l)

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewMeasCycleData(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        msg = f"[{cycle_data}]: Received measurements."
        log.debug(msg)

        now = datetime.datetime.now().timestamp() * 1e9
        delta_s = (now - cycle_data.cycle_timestamp) / 1e9
        msg = f"[{cycle_data}]: Received measurements {delta_s:.3f} s after start."
        log.debug(msg)

        log.debug(f"[{cycle_data}]: Updating buffer with measurements.")

        self._acquisition_buffer.update_cycle(cycle_data)

    @QtCore.Slot(hystcomp_utils.cycle_data.CycleData)
    def onNewProgCycleData(
        self, cycle_data: hystcomp_utils.cycle_data.CycleData
    ) -> None:
        msg = f"[{cycle_data}]: Received cycle data with updated programs."
        log.debug(msg)

        now = datetime.datetime.now().timestamp() * 1e9
        delta_s = (now - cycle_data.cycle_timestamp) / 1e9
        msg = f"[{cycle_data}]: Received program {delta_s:.3f} s after start."
        log.debug(msg)

        log.debug(f"[{cycle_data}]: Updating buffer with program.")
        self._acquisition_buffer.update_cycle(cycle_data)
