from .._mod_replace import replace_modname
from ._local_trim import LocalTrim
from ._settings import LocalTrimSettings, OnlineTrimSettings, TrimSettings

for _mod in [LocalTrim, TrimSettings, LocalTrimSettings, OnlineTrimSettings]:
    replace_modname(_mod, __name__)


__all__ = [
    "LocalTrim",
    "LocalTrimSettings",
    "OnlineTrimSettings",
    "TrimSettings",
]
