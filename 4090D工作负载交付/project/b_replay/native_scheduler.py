"""Native asynchronous FCFS plus the common private fixed-work stop hook only.

No program priority, TTL, pinning or Kairos allocation logic is installed here.
"""
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus
from kairos.replay.execution import update_output
from .observability import MetricsMixin


class ReplayNativeScheduler(MetricsMixin,AsyncScheduler):
    def _update_request_with_output(self, request, new_token_ids, *args, **kwargs):
        return update_output(request, new_token_ids, super()._update_request_with_output,
                             RequestStatus.FINISHED_LENGTH_CAPPED, *args, **kwargs)
