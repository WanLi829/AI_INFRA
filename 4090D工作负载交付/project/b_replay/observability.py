"""Same low-frequency, read-only queue/KV sampling for every scheduler."""
import json
import os
from pathlib import Path
import time


class MetricsMixin:
    def _experiment_emit(self,event,**fields):
        path=os.environ.get('B_REPLAY_METRICS_PATH')
        if path:
            with Path(path).open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(event=event,time=time.monotonic(),**fields))+'\n')

    def add_request(self,request,*args,**kwargs):
        self._experiment_emit('request_arrival',request=request.request_id)
        return super().add_request(request,*args,**kwargs)

    def schedule(self,*args,**kwargs):
        result=super().schedule(*args,**kwargs)
        now=time.monotonic()
        if not hasattr(self,'_experiment_seen'):
            self._experiment_seen=set()
        for request in self.running:
            if request.request_id not in self._experiment_seen:
                self._experiment_seen.add(request.request_id)
                self._experiment_emit('first_admit',request=request.request_id)
        if now-getattr(self,'_experiment_last_sample',-float('inf'))>=.25:
            self._experiment_last_sample=now
            pool=self.kv_cache_manager.block_pool
            pins=getattr(self,'pins',{})
            unique={b.block_id for p in pins.values() for b in p.blocks if not b.is_null}
            total=getattr(pool,'num_gpu_blocks',None)
            self._experiment_emit('sample',running=len(self.running),
                waiting=len(self.waiting)+len(getattr(self,'skipped_waiting',[])),
                free_blocks=pool.get_num_free_blocks(),total_blocks=total,
                unique_pin_blocks=len(unique),pin_programs=len(pins))
        return result
