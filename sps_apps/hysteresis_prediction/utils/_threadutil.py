"""
A powerful, synchronous implementation of run_in_main_thread(...).
It allows you to receive results from the function invocation:

    @run_in_main_thread
    def return_2():
        return 2

    # Runs the above function in the main thread and prints '2':
    print(return_2())
"""

import typing as t
from functools import wraps
from threading import Event, Thread, get_ident

from qtpy.QtCore import QObject, QThread, pyqtSignal
from qtpy.QtWidgets import QApplication


def thread(function: t.Callable, *args: t.Any, **kwargs: t.Any) -> Thread:
    th = Thread(target=function, args=args, kwargs=kwargs)
    th.start()

    return th


def run_in_thread(thread_fn: t.Callable) -> t.Callable:
    def decorator(f: t.Callable) -> t.Any:
        @wraps(f)
        def result(*args: t.Any, **kwargs: t.Any) -> t.Callable:
            thread = thread_fn()
            return Executor.instance().run_in_thread(thread, f, args, kwargs)

        return result

    return decorator


def _main_thread() -> QThread:
    app = QApplication.instance()
    if app:
        return app.thread()
    # We reach here in tests that don't (want to) create a QApplication.
    if int(QThread.currentThreadId()) == get_ident():
        return QThread.currentThread()
    raise RuntimeError("Could not determine main thread")


run_in_main_thread = run_in_thread(_main_thread)


def is_in_main_thread() -> bool:
    return QThread.currentThread() == _main_thread()


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
        self, qthread: QThread, f: t.Callable, args: t.Tuple, kwargs: t.Dict
    ) -> t.Callable:
        if QThread.currentThread() == qthread:
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
        self._result = self._exception = None

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


class Sender(QObject):
    signal = pyqtSignal()


class Receiver(QObject):
    def __init__(
        self, callback: t.Callable, parent: t.Optional[QObject] = None
    ):
        super().__init__(parent)
        self.callback = callback

    def slot(self) -> None:
        self.callback()
