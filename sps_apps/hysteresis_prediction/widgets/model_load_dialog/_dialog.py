"""
A dialog for selecting a model checkpoint and the device to load it on.

author: Anton Lu
"""

from __future__ import annotations

import logging
import re
from os import path

import torch
from op_app_context import settings
from qtpy import QtCore, QtGui, QtWidgets

from ...generated.model_load_dialog_ui import Ui_ModelLoadDialog

log = logging.getLogger(__package__)


class ModelLoadDialog(Ui_ModelLoadDialog, QtWidgets.QDialog):
    loadLocalCheckpoint = QtCore.Signal(str, str, str)
    """Signal emitted when the user has selected a model checkpoint and device to load it on.
    model name, checkpoint path, device.
    """
    loadMlpCheckpoint = QtCore.Signal(str, str, str, str)
    """Signal emitted when the user has selected a model checkpoint and device to load it on.
    model_name, parameters_name, parameters_version, device
    """

    last_selected_device: int = -1
    last_selected_model: str = ""

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        QtWidgets.QDialog.__init__(self, parent)
        self.setupUi(self)

        gpu_item: QtGui.QStandardItem = self.comboDevice.model().item(1)

        self.textCkptPath.setText(settings["checkpoint_file", ""])

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
        if self.last_selected_model:
            log.debug(f"Setting last selected model to {self.last_selected_model}.")
            self.comboBoxModel.setCurrentIndex(
                self.comboBoxModel.findText(self.last_selected_model)
            )
        else:
            log.debug("No last selected model found. Setting to first model.")
            self.last_selected_model = self.comboBoxModel.currentText()

        self.lineModelName.setText(settings["model_name", ""])
        self.lineModelVersion.setText(settings["model_version", ""])

        self.buttonBrowse.clicked.connect(self.onBrowseClicked)
        self.buttonBox.accepted.connect(self.onOkClicked)
        self.buttonBox.rejected.connect(self.reject)

    @QtCore.Slot()
    def onBrowseClicked(self) -> None:
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
        settings["checkpoint_file"] = file_path

    @QtCore.Slot()
    def onOkClicked(self) -> None:
        ckpt_path = self.ckpt_path

        log.debug(f"Attempting to load model checkpoint at {ckpt_path}.")

        if self.tabWidget.currentWidget() == self.tabLocal:
            if not ckpt_path:
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

            self._save_shared_settings()
            log.debug(
                f"Selected checkpoint at {ckpt_path} on device {self.last_selected_device}."
            )

            self.loadLocalCheckpoint.emit(
                self.last_selected_model,
                ckpt_path,
                self.comboDevice.itemText(self.last_selected_device).lower(),
            )

        elif self.tabWidget.currentWidget() == self.tabMlp:
            model_name = self.lineModelName.text()

            if not model_name:
                QtWidgets.QMessageBox.warning(
                    self, "Model load error", "No model name specified."
                )
                log.error("No model name specified.")
                return

            model_version = self.lineModelVersion.text()
            if not model_version:
                QtWidgets.QMessageBox.warning(
                    self, "Model load error", "No model version specified."
                )
                log.error("No model version specified.")
                return

            # check that its d.d
            RE = r"^\d+\.\d+$"
            if not re.match(RE, model_version):
                QtWidgets.QMessageBox.warning(
                    self, "Model load error", "Invalid model version format."
                )
                log.error("Invalid model version format.")
                return

            self._save_shared_settings()

            settings["last_selected_param_name"] = model_name
            settings["last_selected_param_version"] = model_version

            log.debug(
                f"Selected MLP model {model_name} v{model_version} on device {self.last_selected_device}."
            )
            self.loadMlpCheckpoint.emit(
                model_name,
                model_version,
                self.last_selected_model,
                self.last_selected_device,
            )
        else:
            msg = "Unknown tab selected."
            log.critical(msg)
            return

        self.accept()

    def _save_shared_settings(self) -> None:
        self.comboDevice.currentText().lower()
        self.last_selected_device = self.comboDevice.currentIndex()
        self.last_selected_model = self.comboBoxModel.currentText()

        settings["last_selected_model"] = self.last_selected_model

    @property
    def ckpt_path(self) -> str:
        return self.textCkptPath.text()

    @property
    def device(self) -> str:
        return self.comboDevice.currentText().lower()
