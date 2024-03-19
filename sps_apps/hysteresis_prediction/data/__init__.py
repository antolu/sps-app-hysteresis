from ._acquisition import Acquisition  # noqa: F401
from ._acquisition_buffer import AcquisitionBuffer  # noqa: F401
from ._acquisition_buffer import BufferData  # noqa: F401
from ._acquisition_buffer import InsufficientDataError  # noqa:  F401
from ._dataclass import CycleData  # noqa: F401

for _mod in (
    Acquisition,
    AcquisitionBuffer,
    BufferData,
    InsufficientDataError,
    CycleData,
):
    _mod.__module__ = __name__

del _mod
