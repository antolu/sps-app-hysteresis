"""
Two different implementations of writers, writing to
* plain text files (.txt) with comma separated values
* tensorboard files (.tfevents) with scalar summaries

The writers implement a method that takes a dictionary of metrics and writes them to the respective file format.
"""

from __future__ import annotations

import logging
import os
import pathlib

import tensorboardX

from ...contexts import app_context
from ...local.event_building._calculate_metrics import Metrics

log = logging.getLogger(__package__)


class WriterBase:
    def __init__(
        self,
        output_dir: pathlib.Path | os.PathLike | str = ".",
        prefix: str | None = None,
    ) -> None:
        self._output_dir = pathlib.Path(output_dir)

        self._prefix = (
            prefix
            if prefix is not None
            else app_context().TIMESTAMP.strftime("%Y%m%d_%H%M%S")
        )

        self._output_dir.mkdir(parents=True, exist_ok=True)

    def onNewMetrics(self, metrics: dict[str, Metrics]) -> None:
        raise NotImplementedError


class TextWriter(WriterBase):
    def __init__(
        self,
        output_dir: pathlib.Path | os.PathLike | str = ".",
        prefix: str | None = None,
    ) -> None:
        super().__init__(output_dir, prefix)

        self._relative_fname = self._output_dir / f"{self._prefix}_relative.txt"
        self._absolute_fname = self._output_dir / f"{self._prefix}_absolute.txt"

    def onNewMetrics(self, metrics: dict[str, Metrics]) -> None:
        self._write_relative_metrics(metrics["relative"])
        self._write_absolute_metrics(metrics["absolute"])

    def _write_relative_metrics(self, metrics: Metrics) -> None:
        cycle_name = metrics["lsaCycleName"]

        metrics_l = [metric for metric in metrics.values() if isinstance(metric, float)]
        metrics_str = ",".join(map(str, metrics_l))
        metrics_str = f"{cycle_name},{metrics_str}\n"

        log.debug(f"Writing relative metrics to {self._relative_fname}")

        with open(self._relative_fname, "a", encoding="utf8") as f:
            if f.tell() == 0:
                f.write(self._make_header())

            f.write(metrics_str)

    def _write_absolute_metrics(self, metrics: Metrics) -> None:
        cycle_name = metrics["lsaCycleName"]

        metrics_l = [metric for metric in metrics.values() if isinstance(metric, float)]
        metrics_str = ",".join(map(str, metrics_l))
        metrics_str = f"{cycle_name},{metrics_str}\n"

        log.debug(f"Writing avsolute metrics to {self._absolute_fname}")

        with open(self._absolute_fname, "a", encoding="utf8") as f:
            if f.tell() == 0:
                f.write(self._make_header())

            f.write(metrics_str)

    def _make_header(self) -> str:
        return ",".join([
            key for key in Metrics.__annotations__ if key != "lsaCycleName"
        ])


class TensorboardWriter(WriterBase):
    def __init__(
        self,
        output_dir: pathlib.Path | os.PathLike | str = ".",
        prefix: str | None = None,
    ) -> None:
        super().__init__(output_dir, prefix)

        self._relative_writer = tensorboardX.SummaryWriter(
            log_dir=self._output_dir / f"{self._prefix}_relative"
        )
        self._absolute_writer = tensorboardX.SummaryWriter(
            log_dir=self._output_dir / f"{self._prefix}_absolute"
        )

    def onNewMetrics(self, metrics: dict[str, Metrics]) -> None:
        cycle_name = metrics["lsaCycleName"]

        log.debug(f"[{cycle_name}] Writing metrics to tensorboard")

        for key, value in metrics["relative"].items():
            if isinstance(value, float):
                key = f"{cycle_name}/{key}"
                self._relative_writer.add_scalar(key, value)

        for key, value in metrics["absolute"].items():
            if isinstance(value, float):
                key = f"{cycle_name}/{key}"
                self._absolute_writer.add_scalar(key, value)
