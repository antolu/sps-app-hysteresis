from ..._mod_replace import replace_modname
from ._track_precycle import TrackPrecycleEventBuilder

replace_modname(TrackPrecycleEventBuilder, __name__)
