"""
This module contains the AppSettings class, an abstraction of the
QSettings class.
"""
from os import path

from PyQt5.QtCore import QSettings


class AppSettings:
    """
    The AppSettings class wraps an instance of QSettings and provides easy
    access to some data that can be saved to the users computer.

    The settings must be loaded using the init method before accessing any
    other attribute.
    """

    def __init__(self):
        self.settings: QSettings = None

        self.is_init = False

        self._deploy_path = dict()

    def init(self) -> None:
        if not self.is_init:
            self.settings = QSettings()
            self.settings.sync()

            self.is_init = True

    def save_window_state(self, geometry, window_state) -> None:
        """Currently not used"""
        if not self.is_init:
            self.init()

        self.settings.setValue("geometry", geometry)
        self.settings.setValue("windowState", window_state)
        self.settings.sync()

    def geometry(self):
        """Currently not used"""
        if not self.is_init:
            self.init()

        return self.settings.value("geometry", None)

    def window_state(self):
        if not self.is_init:
            self.init()

        return self.settings.value("windowState", None)

    @property
    def current_dir(self) -> str:
        """Accesses the last used directory in a file browser."""
        if not self.is_init:
            self.init()

        return self.settings.value("current_directory", ".", str)

    @current_dir.setter
    def current_dir(self, pth: str):
        """Accesses the last used directory in a file browser."""
        if not self.is_init:
            self.init()

        if not isinstance(pth, str):
            raise ValueError(f"Path {pth} should be of type string.")

        if path.isfile(pth) and not path.isdir(pth):
            pth = path.split(pth)[0]

        self.settings.setValue("current_directory", pth)


settings = AppSettings()
