"""
A powerful, synchronous implementation of run_in_main_thread(...).
It allows you to receive results from the function invocation:

    @run_in_main_thread
    def return_2():
        return 2

    # Runs the above function in the main thread and prints '2':
    print(return_2())
"""

from __future__ import annotations

import typing
import typing as t
from functools import wraps
from threading import Event, Thread, get_ident

from qtpy import QtCore
from qtpy.QtWidgets import QApplication


class Signals(QtCore.QObject):
    started = QtCore.Signal()
    finished = QtCore.Signal()
    result = QtCore.Signal(object)
    exception = QtCore.Signal(Exception)


class ThreadWorker(QtCore.QRunnable):
    def __init__(self, function: t.Callable, *args: t.Any, **kwargs: t.Any):
        super().__init__()

        self.function = function
        self.args = args
        self.kwargs = kwargs

        self.signals = Signals()

        self.started = self.signals.started
        self.finished = self.signals.finished
        self.result = self.signals.result
        self.exception = self.signals.exception

    @QtCore.Slot()
    def run(self) -> None:
        self.started.emit()

        try:
            result = self.function(*self.args, **self.kwargs)
            self.result.emit(result)
        except Exception as e:
            self.exception.emit(e)
        finally:
            self.finished.emit()


def thread(function: t.Callable, *args: t.Any, **kwargs: t.Any) -> Thread:
    th = Thread(target=function, args=args, kwargs=kwargs)
    th.start()

    return th


def run_in_thread(thread_fn: t.Callable[[], QtCore.QThread]) -> t.Callable:
    def decorator(f: t.Callable) -> t.Any:
        @wraps(f)
        def result(*args: t.Any, **kwargs: t.Any) -> t.Callable:
            thread = thread_fn()
            return Executor.instance().run_in_thread(thread, f, args, kwargs)

        return result

    return decorator


def _main_thread() -> QtCore.QThread:
    app = QApplication.instance()
    if app:
        return app.thread()
    # We reach here in tests that don't (want to) create a QApplication.
    if int(QtCore.QThread.currentThreadId()) == get_ident():
        return QtCore.QThread.currentThread()
    raise RuntimeError("Could not determine main thread")


run_in_main_thread = run_in_thread(_main_thread)


def is_in_main_thread() -> bool:
    return QtCore.QThread.currentThread() == _main_thread()


class Executor:
    _INSTANCE = None

    @classmethod
    def instance(cls) -> "Executor":
        if cls._INSTANCE is None:
            cls._INSTANCE = cls(QApplication.instance())
        return cls._INSTANCE

    def __init__(self, app: QApplication):
        self._pending_tasks: list[Task] = []
        self._app_is_about_to_quit = False
        app.aboutToQuit.connect(self._about_to_quit)

    def _about_to_quit(self) -> None:
        self._app_is_about_to_quit = True
        for task in self._pending_tasks:
            task.set_exception(SystemExit())
            task.has_run.set()

    def run_in_thread(
        self,
        qthread: QtCore.QThread,
        f: t.Callable,
        args: t.Tuple,
        kwargs: t.Dict,
    ) -> t.Callable:
        if QtCore.QThread.currentThread() == qthread:
            return f(*args, **kwargs)
        elif self._app_is_about_to_quit:
            # In this case, the target thread's event loop most likely is not
            # running any more. This would mean that our task (which is
            # submitted to the event loop via events/slots) is never run.
            raise SystemExit()
        task = Task(f, args, kwargs)
        self._pending_tasks.append(task)
        try:
            receiver = Receiver(task)
            receiver.moveToThread(qthread)
            sender = Sender()
            sender.signal.connect(receiver.slot)
            sender.signal.emit()
            if not qthread.isRunning():
                qthread.start()
            task.has_run.wait()
            return task.result
        finally:
            self._pending_tasks.remove(task)


class Task:
    def __init__(self, fn: t.Callable, args: t.Any, kwargs: t.Any):
        self._fn = fn
        self._args = args
        self._kwargs = kwargs
        self.has_run = Event()
        self._exception: Exception | None = None
        self._result: typing.Any | None = None

    def __call__(self) -> None:
        try:
            self._result = self._fn(*self._args, **self._kwargs)
        except Exception as e:
            self._exception = e
        finally:
            self.has_run.set()

    def set_exception(self, exception: t.Any) -> None:
        self._exception = exception

    @property
    def result(self) -> t.Any:
        if not self.has_run.is_set():
            raise ValueError("Hasn't run.")
        if self._exception:
            raise self._exception
        return self._result


class Sender(QtCore.QObject):
    signal = QtCore.Signal()


class Receiver(QtCore.QObject):
    def __init__(self, callback: t.Callable, parent: t.Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.callback = callback

    def slot(self) -> None:
        self.callback()
