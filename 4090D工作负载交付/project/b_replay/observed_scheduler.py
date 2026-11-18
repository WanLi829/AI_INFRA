"""Public experiment entrypoints; upstream method implementations are unchanged."""
from .observability import MetricsMixin
from kairos._runtime.compat.pipelined_prefix import PipelinedPrefixScheduler
from kairos.scheduler import KairosScheduler


class ObservedContinuumScheduler(MetricsMixin,PipelinedPrefixScheduler):
    pass


class ObservedKairosScheduler(MetricsMixin,KairosScheduler):
    pass
