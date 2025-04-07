from __future__ import annotations

import logging

import pyrbac
from qtpy import QtCore

log = logging.getLogger(__name__)


class PyrbacAuthenticationListener(pyrbac.AuthenticationListener, QtCore.QObject):
    rbacTokenObtained = QtCore.Signal(pyrbac.Token)

    rbacTokenExpired = QtCore.Signal(pyrbac.Token)

    def __init__(self):
        pyrbac.AuthenticationListener.__init__(self)
        QtCore.QObject.__init__(self)

        self.token: pyrbac.Token | None = None

    def authentication_done(self, token: pyrbac.Token):
        log.info(f"RBAC token obtained: {token.get_serial_id()}")
        self.token = token
        self.rbacTokenObtained.emit(token)

    def authentication_error(self, error: pyrbac.AuthenticationError):
        log.error(f"RBAC authentication error: {error}")
        if self.token is not None:
            self.rbacTokenExpired.emit(self.token)

        self.token = None

    def token_expired(self, token: pyrbac.Token):
        log.info(f"RBAC token expired: {token.get_serial_id()}")
        self.rbacTokenExpired.emit(token)
