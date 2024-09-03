from ._acquisition import Acquisition  # noqa: F401
from ._dataclass import CycleData  # noqa: F401

for _mod in (
    Acquisition,
    CycleData,
):
    _mod.__module__ = __name__

del _mod
