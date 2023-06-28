"""
Application context that's accessible as a global object, providing
which machine, ring (for PSB) and UCAP node type the application is
running for.

Additionally provides pyjapc and pjlsa instances configured for the
machine in the context.
N.B. The Machine attribute must be set first, or default PyJapc
accelerator will be used.
"""
from __future__ import annotations

import logging
import typing

from pjlsa import LSAClient
from pyjapc import PyJapc

__all__ = ["context"]
log = logging.getLogger(__name__)


class ApplicationContext:
    def __init__(self) -> None:
        self.lsa_server: typing.Literal["sps", "next"] = "sps"
        self.save_dir = "output"

    @property
    def lsa(self) -> LSAClient:
        return LSAClient(server=self.lsa_server)

    @property
    def japc(self) -> PyJapc:
        return PyJapc(
            incaAcceleratorName=None,
            selector="SPS.USER.ALL",
            noSet=True,
            logLevel="INFO",
        )


context = ApplicationContext()
