"""
Implementation of a PyQt signal/slot system with asyncio.
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from threading import Thread, current_thread
from typing import Any, Callable, Optional, Type

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Handle:
    slot: Callable[..., None]
    thread: Thread


class Signal:
    """
    Implementation of a PyQt signal/slot system with asyncio.

    The implementation is thread safe.
    However, the slots are called in a new thread to avoid blocking
    the event loop, therefore the slots should be thread safe.
    """

    _active_signals: set[Signal] = set()

    def __init__(self, *types: Type) -> None:
        self._types = types
        self._handles: list[Handle] = []

        self._q: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._wait_task: Optional[asyncio.Task] = None

        self._active_signals.add(self)

    def _schedule_signal(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> asyncio.Task:
        loop = loop if loop is not None else self._get_loop()
        self._loop = loop

        self._q = asyncio.Queue()
        self._wait_task = loop.create_task(self._wait())

        return self._wait_task

    @staticmethod
    def _get_loop() -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop

    def emit(self, *args: Any) -> None:
        """
        Send a signal to all connected slots.
        This operation is thread safe, and the slots executed in a new thread
        by the event loop.

        :param args: The arguments to send to the slots. The arguments can be
            of any type, but must be the same number as the signal. The slot
            may fail (silently) if incorrect arguments are passed.
        """
        if self._loop is None:
            raise RuntimeError(
                "Event loop has not yet been started. " "Cannot emit signal."
            )
        asyncio.run_coroutine_threadsafe(self._put(*args), loop=self._loop)

    def connect(self, handle: Callable[..., None]) -> None:
        """
        Connect a slot to the signal. The slot is executed when the signal is
        emitted. The slot must have the same number of arguments as the
        signal.

        :param handle: The slot to connect to the signal.

        :raises TypeError: If the slot is not a callable.
        :raises TypeError: If the slot has the wrong number of arguments.
        """
        if not isinstance(handle, Callable):
            raise TypeError(f"Expected a callable, got {type(handle)}")

        signature = inspect.signature(handle)
        if len(signature.parameters) != len(self._types):
            raise TypeError(
                f"Expected {len(self._types)} arguments, got "
                f"{len(signature.parameters)}."
            )

        log.debug(f"Connecting {handle.__name__}.")
        self._handles.append(Handle(handle, current_thread()))

    def disconnect(self, handle: Callable[..., None]) -> None:
        """
        Remove a slot from the signal. The slot will no longer be executed
        when the signal is emitted.

        :handle: The slot to disconnect from the signal.

        :raises ValueError: If the slot is not connected to the signal.
        """
        for handler in self._handles:
            if handler.slot == handle:
                log.debug(f"Disconnecting {handle.__name__}.")
                self._handles.remove(handler)
                return
        raise ValueError(f"Could not find handler {handle}.")

    async def _wait(self) -> None:
        """
        The event loop that waits for signals and executes the slots.

        The slots are executed in the same thread as the event loop.
        """
        if self._q is None:
            raise RuntimeError(
                "Event loop is not running. Did you forget "
                "to schedule Signal._schedule_signal?"
            )
        try:
            while True:
                value = await self._q.get()

                for handle in self._handles:
                    try:
                        handle.slot(*value)
                    except asyncio.CancelledError:
                        log.debug("Signal event loop cancelled.")
                        break
                    except:  # noqa
                        log.exception(
                            "An error occurred when executing slot "
                            f"{handle.slot}."
                        )
                        continue
        except asyncio.CancelledError:
            log.debug("Signal wait loop cancelled.")
            pass

    async def _put(self, *args: Any) -> None:
        """
        Coroutine that puts a signal on the queue. This method should be
        called in the same event loop as the signal to avoid race
        conditions.
        """
        if self._q is None:
            raise RuntimeError(
                "Event loop is not running. Did you forget "
                "to schedule Signal._schedule_signal?"
            )
        value = tuple(args) if not isinstance(*args, tuple) else args
        await self._q.put(value)

    @staticmethod
    def start_all(loop: asyncio.AbstractEventLoop) -> None:
        log.debug(f"Starting all signals in loop {loop}.")
        for signal in Signal._active_signals:
            if signal._loop is None:
                signal._schedule_signal(loop)

    @staticmethod
    def cancel_all() -> None:
        log.debug("Stopping all active signals...")
        for signal in Signal._active_signals:
            if signal._wait_task is not None:
                signal._wait_task.cancel()
