from __future__ import annotations

import logging
import typing

import pyrbac

log = logging.getLogger(__name__)


class PyrbacAuthenticationListener(pyrbac.AuthenticationListener):
    def __init__(self):
        super().__init__()
        self.token: pyrbac.Token | None = None
        self._token_obtained_callbacks: list[typing.Callable[[pyrbac.Token], None]] = []
        self._token_expired_callbacks: list[typing.Callable[[pyrbac.Token], None]] = []

    def register_token_obtained_callback(
        self, callback: typing.Callable[[pyrbac.Token], None]
    ) -> None:
        """Register a callback to be called when a token is obtained."""
        self._token_obtained_callbacks.append(callback)

    def register_token_expired_callback(
        self, callback: typing.Callable[[pyrbac.Token], None]
    ) -> None:
        """Register a callback to be called when a token expires."""
        self._token_expired_callbacks.append(callback)

    def authentication_done(self, token: pyrbac.Token) -> None:
        log.info(f"RBAC token obtained: {token.get_serial_id()}")
        self.token = token
        for callback in self._token_obtained_callbacks:
            try:
                callback(token)
            except:  # noqa: E722
                # Catch all exceptions to avoid crashing the application
                log.exception("Error in token obtained callback")

    def authentication_error(self, error: pyrbac.AuthenticationError) -> None:
        log.error(f"RBAC authentication error: {error}")
        if self.token is not None:
            for callback in self._token_expired_callbacks:
                try:
                    callback(self.token)
                except:  # noqa: E722
                    log.exception("Error in token expired callback")
        self.token = None

    def token_expired(self, token: pyrbac.Token) -> None:
        log.info(f"RBAC token expired: {token.get_serial_id()}")
        for callback in self._token_expired_callbacks:
            try:
                callback(token)
            except:  # noqa: E722
                log.exception("Error in token expired callback")
