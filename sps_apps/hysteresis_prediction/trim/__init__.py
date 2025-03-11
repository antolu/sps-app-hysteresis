from .._mod_replace import replace_modname
from ._cycle_metadata import cycle_metadata
from ._settings import LocalTrimSettings, OnlineTrimSettings, TrimSettings

for _mod in [
    TrimSettings,
    LocalTrimSettings,
    OnlineTrimSettings,
    cycle_metadata,
]:
    replace_modname(_mod, __name__)


__all__ = [
    "LocalTrimSettings",
    "OnlineTrimSettings",
    "TrimSettings",
    "cycle_metadata",
]
