from ._add_measurement_reference import AddMeasurementReferencesEventBuilder
from ._add_measurements import AddMeasurementsEventBuilder
from ._add_measurements_cycle_stamped import CycleStampedAddMeasurementsEventBuilder
from ._add_programmed import AddProgrammedEventBuilder
from ._buffer_builder import BufferEventbuilder
from ._calculate_metrics import CalculateMetricsConverter
from ._create_cycle import CreateCycleEventBuilder
from ._event_builder_abc import (
    BufferedSubscription,
    BufferedSubscriptionEventBuilder,
    CycleStampGroupedTriggeredEventBuilder,
    EventBuilderAbc,
    Subscription,
)
from ._pyda import JapcEndpoint
from ._start_cycle import StartCycleEventBuilder
from ._track_dyneco import TrackDynEcoEventBuilder
from ._track_fulleco import TrackFullEcoEventBuilder
from ._track_reference_changed import TrackReferenceChangedEventBuilder

for _mod in (
    CreateCycleEventBuilder,
    AddMeasurementsEventBuilder,
    BufferEventbuilder,
    EventBuilderAbc,
    BufferedSubscriptionEventBuilder,
    Subscription,
    BufferedSubscription,
    AddProgrammedEventBuilder,
    AddMeasurementReferencesEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    CycleStampGroupedTriggeredEventBuilder,
    TrackFullEcoEventBuilder,
    TrackDynEcoEventBuilder,
    StartCycleEventBuilder,
    JapcEndpoint,
    TrackReferenceChangedEventBuilder,
    CalculateMetricsConverter,
):
    _mod.__module__ = __name__

del _mod
