from __future__ import annotations

import logging
from os import path
from typing import Optional

import torch
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QStandardItem
from qtpy.QtWidgets import QDialog, QFileDialog, QMessageBox, QWidget

from ...generated.model_load_dialog_ui import Ui_ModelLoadDialog

log = logging.getLogger(__name__)


class ModelLoadDialog(Ui_ModelLoadDialog, QDialog):
    load_checkpoint = Signal(str, str)  # ckpt path, device

    def __init__(
        self, default_device: str = "cpu", parent: Optional[QWidget] = None
    ):
        QDialog.__init__(self, parent)
        self.setupUi(self)

        gpu_item: QStandardItem = self.comboDevice.model().item(1)

        if torch.cuda.is_available():
            gpu_item.setFlags(gpu_item.flags() | Qt.ItemIsEnabled)
            if default_device == "gpu":
                self.comboDevice.setCurrentIndex(1)
        else:
            gpu_item.setFlags(gpu_item.flags() & ~Qt.ItemIsEnabled)
            self.comboDevice.setCurrentIndex(0)

        self.buttonBrowse.clicked.connect(self.on_browse_clicked)
        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.reject)

    def on_browse_clicked(self) -> None:
        log.debug("Opening up model load file dialog.")

        file_path, ok = QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            "",
            "PyTorch model (*.pt, *.pth, *.ckpt)",
        )

        if not ok:
            log.debug("Model load file dialog cancelled.")
            return

        log.debug(f"Selected model checkpoint: {file_path}.")

        self.textCkptPath.setText(file_path)

    def on_ok_clicked(self) -> None:
        ckpt_path = self.textCkptPath.text()

        if ckpt_path == "":
            QMessageBox.warning("No checkpoint path specified.")
            log.error("No checkpoint path specified.")
            return

        if not path.exists(ckpt_path):
            QMessageBox.warning("Checkpoint path does not exist.")
            log.error("Checkpoint path does not exist.")
            return

        device = self.comboDevice.currentText().lower()

        self.load_checkpoint.emit(ckpt_path, device)

        self.accept()
