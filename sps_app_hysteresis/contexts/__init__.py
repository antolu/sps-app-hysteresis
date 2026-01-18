import typing

from .._mod_replace import replace_modname
from ..settings import OnlineTrimSettings, StandaloneTrimSettings
from ._base_context import (
    ApplicationContext,
    EddyCurrentModel,
    MeasurementEddyCurrentModel,
    ParameterNames,
    RemoteParameterNames,
)
from ._params import (
    MBI_EDDY_CURRENT_MODEL,
    MBI_MEASUREMENT_EDDY_CURRENT_MODEL,
    MBI_PARAMS,
    MBI_REMOTE_PARAMS,
)

for _mod in [ApplicationContext, ParameterNames, MBI_PARAMS]:
    replace_modname(_mod, __name__)


class ContextRecipe(typing.TypedDict):
    device: typing.Literal["MBI", "QF", "QD"]
    param_names: ParameterNames
    trim_settings: type[StandaloneTrimSettings | OnlineTrimSettings]
    remote_param_names: typing.NotRequired[RemoteParameterNames]
    eddy_current_model: EddyCurrentModel
    measurement_eddy_current_model: MeasurementEddyCurrentModel


_context_recipes: dict[str, ContextRecipe] = {
    "MBI_standalone": {
        "device": "MBI",
        "param_names": MBI_PARAMS,
        "trim_settings": StandaloneTrimSettings,
        "eddy_current_model": MBI_EDDY_CURRENT_MODEL,
        "measurement_eddy_current_model": MBI_MEASUREMENT_EDDY_CURRENT_MODEL,
    },
    "MBI_online": {
        "device": "MBI",
        "param_names": MBI_PARAMS,
        "trim_settings": OnlineTrimSettings,
        "remote_param_names": MBI_REMOTE_PARAMS,
        "eddy_current_model": MBI_EDDY_CURRENT_MODEL,
        "measurement_eddy_current_model": MBI_MEASUREMENT_EDDY_CURRENT_MODEL,
    },
    # "QF": {
    #     "device": "QF",
    #     "param_names": MBI_PARAMS,
    #     "trim_settings": StandaloneTrimSettings,
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
    recipe = _context_recipes[f"{device}_{'online' if online else 'standalone'}"]

    trim_settings_cls = recipe["trim_settings"]
    if online:
        if trim_settings_cls != OnlineTrimSettings:
            msg = "Online context requested, but recipe is not for online context"
            raise ValueError(msg)
        trim_settings_cls = typing.cast(type[OnlineTrimSettings], trim_settings_cls)
        trim_settings = trim_settings_cls(
            device=recipe["param_names"].TRIM_SETTINGS or ""
        )
        remote_params = recipe.get("remote_param_names")
        if remote_params is None:
            msg = "Online context requested, but no remote parameter names provided"
            raise ValueError(msg)
    else:
        if trim_settings_cls != StandaloneTrimSettings:
            msg = (
                "Standalone context requested, but recipe is not for standalone context"
            )
            raise ValueError(msg)
        trim_settings_cls = typing.cast(type[StandaloneTrimSettings], trim_settings_cls)
        trim_settings = trim_settings_cls(prefix=device)

        remote_params = None

    context = ApplicationContext(
        device=device,
        param_names=recipe["param_names"],
        trim_settings=trim_settings,
        eddy_current_model=recipe["eddy_current_model"],
        measurement_eddy_current_model=recipe["measurement_eddy_current_model"],
        remote_params=remote_params,
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
