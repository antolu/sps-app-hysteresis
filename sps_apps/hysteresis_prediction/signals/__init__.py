from .._mod_replace import replace_modname
from .track_precycle import TrackPrecycleEventBuilder

replace_modname(TrackPrecycleEventBuilder, __name__)


__all__ = ["TrackPrecycleEventBuilder"]
