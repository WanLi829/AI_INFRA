"""Fixed-TTL ablation on the same program scheduler as the Continuum path."""
import math
import os

from kairos._runtime.compat.pipelined_prefix import PipelinedPrefixScheduler


class StaticTTLScheduler(PipelinedPrefixScheduler):
    def _switch_experiment(self, name):
        if name == self.experiment:
            return
        super()._switch_experiment(name)
        if name.startswith("continuum-index"):
            seconds = float(os.environ["CONTINUUM_TTL_SECONDS"])
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("Fixed TTL must be positive and finite")
            self.policy_model.mode = "fixed"
            self.policy_model.fixed = seconds
            self._emit("fixed_ttl_active", seconds=seconds,
                       definition="Engineering fixed-TTL ablation on Continuum program scheduling")
