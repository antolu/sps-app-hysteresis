import typing

from .._mod_replace import replace_modname
from ..local.trim import LocalTrimSettings, OnlineTrimSettings
from ._base_context import (
    ApplicationContext,
    EddyCurrentModel,
    ParameterNames,
    UcapParameterNames,
)
from ._params import MBI_EDDY_CURRENT_MODEL, MBI_PARAMS, MBI_UCAP_PARAMS

for _mod in [ApplicationContext, ParameterNames, MBI_PARAMS]:
    replace_modname(_mod, __name__)


class ContextRecipe(typing.TypedDict):
    device: typing.Literal["MBI", "QF", "QD"]
    param_names: ParameterNames
    trim_settings: type[LocalTrimSettings | OnlineTrimSettings]
    ucap_param_names: typing.NotRequired[UcapParameterNames]
    eddy_current_model: EddyCurrentModel


_context_recipes: dict[str, ContextRecipe] = {
    "MBI_local": {
        "device": "MBI",
        "param_names": MBI_PARAMS,
        "trim_settings": LocalTrimSettings,
        "eddy_current_model": MBI_EDDY_CURRENT_MODEL,
    },
    "MBI_online": {
        "device": "MBI",
        "param_names": MBI_PARAMS,
        "trim_settings": OnlineTrimSettings,
        "ucap_param_names": MBI_UCAP_PARAMS,
        "eddy_current_model": MBI_EDDY_CURRENT_MODEL,
    },
    # "QF": {
    #     "device": "QF",
    #     "param_names": MBI_PARAMS,
    #     "trim_settings": LocalTrimSettings,
    # },
    # "QD": {
    #     "device": "QD",
    #     "param_names": MBI_PARAMS,
    #     "trim_settings": OnlineTrimSettings,
    # },
}
_app_context: ApplicationContext | None = None


def set_context(
    device: typing.Literal["MBI", "QF", "QD"],
    *,
    online: bool = False,
) -> ApplicationContext:
    recipe = _context_recipes[f"{device}_{'online' if online else 'local'}"]

    trim_settings_cls = recipe["trim_settings"]
    if online:
        if trim_settings_cls != OnlineTrimSettings:
            msg = "Online context requested, but recipe is not for online context"
            raise ValueError(msg)
        trim_settings_cls = typing.cast(type[OnlineTrimSettings], trim_settings_cls)
        trim_settings = trim_settings_cls(
            device=recipe["param_names"].TRIM_SETTINGS or ""
        )
        ucap_params = recipe.get("ucap_param_names")
        if ucap_params is None:
            msg = "Online context requested, but no UCAP parameter names provided"
            raise ValueError(msg)
    else:
        if trim_settings_cls != LocalTrimSettings:
            msg = "Local context requested, but recipe is not for local context"
            raise ValueError(msg)
        trim_settings_cls = typing.cast(type[LocalTrimSettings], trim_settings_cls)
        trim_settings = trim_settings_cls(prefix=device)

        ucap_params = None

    context = ApplicationContext(
        device=device,
        param_names=recipe["param_names"],
        trim_settings=trim_settings,
        ucap_params=ucap_params,
        eddy_current_model=recipe["eddy_current_model"],
    )
    global _app_context  # noqa: PLW0603
    _app_context = context

    return context


def app_context() -> ApplicationContext:
    if _app_context is None:
        msg = "Context not set, call set_context before accessing app_context"
        raise AttributeError(msg)

    return _app_context


__all__ = [
    "app_context",
    "set_context",
]
