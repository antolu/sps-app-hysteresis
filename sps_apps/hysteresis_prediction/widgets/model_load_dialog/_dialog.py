"""
A dialog for selecting a model checkpoint and the device to load it on.

author: Anton Lu
"""

from __future__ import annotations

import logging
from os import path

import torch
from op_app_context import settings
from qtpy import QtCore, QtGui, QtWidgets

from ...generated.model_load_dialog_ui import Ui_ModelLoadDialog

log = logging.getLogger(__name__)


class ModelLoadDialog(Ui_ModelLoadDialog, QtWidgets.QDialog):
    load_checkpoint = QtCore.Signal(str, str, str)
    """Signal emitted when the user has selected a model checkpoint and device to load it on.
    model name, checkpoint path, device.
    """
    last_selected_device: int = -1
    last_selected_model: str = ""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        gpu_item: QtGui.QStandardItem = self.comboDevice.model().item(1)

        if self.last_selected_device == -1:
            if torch.cuda.is_available():
                gpu_item.setFlags(gpu_item.flags() | QtCore.Qt.ItemIsEnabled)
                self.comboDevice.setCurrentIndex(1)
            else:
                gpu_item.setFlags(gpu_item.flags() & ~QtCore.Qt.ItemIsEnabled)
                self.comboDevice.setCurrentIndex(0)
            self.last_selected_device = self.comboDevice.currentIndex()
        else:
            self.comboDevice.setCurrentIndex(self.last_selected_device)

        self.last_selected_model = settings["last_selected_model", ""]
        if self.last_selected_model != "":
            self.comboBoxModel.setCurrentIndex(
                self.comboBoxModel.findText(self.last_selected_model)
            )
        else:
            self.last_selected_model = self.comboBoxModel.currentText()

        self.buttonBrowse.clicked.connect(self.on_browse_clicked)
        self.buttonBox.accepted.connect(self.on_ok_clicked)
        self.buttonBox.rejected.connect(self.reject)

    @QtCore.Slot()
    def on_browse_clicked(self) -> None:
        log.debug("Opening up model load file dialog.")

        file_path, ok = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select model checkpoint",
            settings["checkpoint_dir", "."],
            "PyTorch model (*.pt *.pth *.ckpt)",
        )

        if not ok:
            log.debug("Model load file dialog cancelled.")
            return

        log.debug(f"Selected model checkpoint: {file_path}.")

        self.textCkptPath.setText(file_path)
        settings["checkpoint_dir"] = path.dirname(file_path)

    @QtCore.Slot()
    def on_ok_clicked(self) -> None:
        ckpt_path = self.textCkptPath.text()

        if ckpt_path == "":
            QtWidgets.QMessageBox.warning(
                self, "Model load error", "No checkpoint path specified."
            )
            log.error("No checkpoint path specified.")
            return

        if not path.exists(ckpt_path):
            QtWidgets.QMessageBox.warning(
                self, "Model load error", "Checkpoint path does not exist."
            )
            log.error("Checkpoint path does not exist.")
            return

        device = self.comboDevice.currentText().lower()
        log.debug(f"Selected checkpoint at {ckpt_path} on device {device}.")
        self.last_selected_device = self.comboDevice.currentIndex()
        self.last_selected_model = self.comboBoxModel.currentText()

        settings["last_selected_model"] = self.last_selected_model

        self.load_checkpoint.emit(self.last_selected_model, ckpt_path, device)

        self.accept()

    @property
    def ckpt_path(self) -> str:
        return self.textCkptPath.text()

    @property
    def device(self) -> str:
        return self.comboDevice.currentText().lower()
