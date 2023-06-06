"""
Code to make it a two liner to know the mapped PLS users for an accelerator.
Maps PLS user to LSA cycle and vice-versa.

This module uses pjlsa to interface with LSA.

The lsa_contexts object is a global object that can be imported in the app.
"""
import logging
import typing as t

from ..core.application_context import context

__all__ = ["LSAContexts"]

log = logging.getLogger(__name__)


class LSAContexts:
    """
    This class wraps pjlsa to find which LSA contexts are mapped to a played
    PLS user. The user needs to call the `update` method to fetch contexts
    after machine is set.
    """

    def __init__(self, machine: str):
        """
        Optionally pass a machine in the constructor, or set the machine
        property after initialization.

        Parameters
        ----------
        machine : str or Accelerator
            CERN machine to map contexts for.
        """
        self.lsa_to_pls: dict[str, str] = {}
        self.pls_to_lsa: dict[str, str] = {}

        self.machine = machine

    def update(self):
        """
        Fetches LSA contexts from LSA with pjlsa and queries if they are being
        played.

        This function can be run asynchronously.
        """
        if self.machine is None:
            raise ValueError(
                "Must set machine attribute before attempting to "
                "fetch contexts."
            )

        log.debug("Preparing to update LSA cycle to PLS mapping.")
        with context.lsa.java_api():
            from cern.accsoft.commons.domain import CernAccelerator
            from cern.lsa.client import ContextService, ServiceLocator
            from cern.lsa.domain.settings import (
                Contexts,
                DrivableContext,
                StandAloneCycle,
            )

            if self.machine != "SPS":
                raise NotImplementedError(
                    "LSAContexts is currently only implemented for SPS."
                )
            acc = CernAccelerator.SPS

            log.debug(f"Getting drivable contexts for " f"{self.machine}.")
            service = ServiceLocator.getService(ContextService)
            contexts: t.Collection[StandAloneCycle]
            drivable_contexts: t.Collection[DrivableContext]

            contexts = service.findStandAloneCycles(acc)
            drivable_contexts = Contexts.getDrivableContexts(contexts)

            mapped_contexts = [
                o for o in drivable_contexts if o.getUser() is not None
            ]

            lsa_to_pls = {}
            pls_to_lsa = {}

            log.debug("Mapping LSA cycles to PLS users.")
            for o in mapped_contexts:
                pls_user = o.getUser()
                lsa_cycle = o.getName()

                if pls_user == "":
                    pls_user = "non-multiplexed"

                lsa_to_pls[lsa_cycle] = pls_user
                pls_to_lsa[pls_user] = lsa_cycle

            log.debug(
                f"Found {len(lsa_to_pls)} resident contexts for "
                f"{self.machine}."
            )

        self.lsa_to_pls = lsa_to_pls
        self.pls_to_lsa = pls_to_lsa

    def is_mapped(self, lsa_cycle: str) -> bool:
        """
        Returns if an lsa cycle is mapped to a played user.

        Parameters
        ----------
        lsa_cycle : str
            Name of the LSA cycle

        Returns
        -------
        bool
            True if LSA cycle is mapped to PLS user, otherwise False.
        """
        return lsa_cycle in self.lsa_to_pls
