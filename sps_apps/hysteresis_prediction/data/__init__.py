from ._acquisition import Acquisition  # noqa: F401

for _mod in (Acquisition,):
    _mod.__module__ = __name__

del _mod
