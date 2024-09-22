from ._acquisition import Acquisition  # noqa: F401
from ._add_measurement_reference import AddMeasurementReferencesEventBuilder
from ._add_measurements import AddMeasurementsEventBuilder
from ._add_measurements_cycle_stamped import CycleStampedAddMeasurementsEventBuilder
from ._add_programmed import AddProgrammedEventBuilder
from ._buffer_builder import BufferEventbuilder
from ._create_cycle import CreateCycleEventBuilder
from ._event_builder_abc import (
    BufferedSubscription,
    BufferedSubscriptionEventBuilder,
    CycleStampGroupedTriggeredEventBuilder,
    CycleStampSubscriptionBuffer,
    EventBuilderAbc,
    Subscription,
)

for _mod in (
    Acquisition,
    CreateCycleEventBuilder,
    AddMeasurementsEventBuilder,
    BufferEventbuilder,
    EventBuilderAbc,
    BufferedSubscriptionEventBuilder,
    Subscription,
    BufferedSubscription,
    CycleStampSubscriptionBuffer,
    AddProgrammedEventBuilder,
    AddMeasurementReferencesEventBuilder,
    CycleStampedAddMeasurementsEventBuilder,
    CycleStampGroupedTriggeredEventBuilder,
):
    _mod.__module__ = __name__

del _mod
