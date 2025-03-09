from .._mod_replace import replace_modname
from ._settings import LocalTrimSettings, OnlineTrimSettings, TrimSettings

for _mod in [TrimSettings, LocalTrimSettings, OnlineTrimSettings]:
    replace_modname(_mod, __name__)


__all__ = [
    "LocalTrimSettings",
    "OnlineTrimSettings",
    "TrimSettings",
]
