"""Same low-frequency, read-only queue/KV sampling for every scheduler."""
import json
import os
from pathlib import Path
import time
from .common import stats


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
        began=time.perf_counter()
        before={r.request_id:getattr(r,'num_computed_tokens',None)
                for queue in (self.running,self.waiting,getattr(self,'skipped_waiting',[])) for r in queue}
        result=super().schedule(*args,**kwargs)
        costs=self.__dict__.setdefault('_experiment_schedule_costs',[])
        costs.append(time.perf_counter()-began)
        now=time.monotonic()
        for queue in (self.running,self.waiting,getattr(self,'skipped_waiting',[])):
            for r in queue:
                old,new=before.get(r.request_id),getattr(r,'num_computed_tokens',None)
                if old is not None and new is not None and new<old:
                    self._experiment_emit('computed_frontier_rewound',request=r.request_id,
                        logical_frontier_loss_tokens=old-new,actual_recompute_tokens_measured=False)
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
            native_ids=set()
            inventory_available=True
            try:
                for request in self.running:
                    native_ids.update(b.block_id for b in self.kv_cache_manager.get_blocks(request.request_id).blocks[0] if not b.is_null)
            except (AttributeError,KeyError,TypeError):
                inventory_available=False
            self._experiment_emit('sample',running=len(self.running),
                waiting=len(self.waiting)+len(getattr(self,'skipped_waiting',[])),
                free_blocks=pool.get_num_free_blocks(),total_blocks=total,
                unique_pin_blocks=len(unique),pin_programs=len(pins),
                unique_pin_ratio=len(unique)/total if total else None,
                shared_native_pin_blocks=len(unique & native_ids) if inventory_available else None,
                schedule_cost_s=stats(costs),schedule_cost_sum_s=sum(costs),
                controller_cost_s=stats(self.__dict__.get('_experiment_controller_costs',[])),
                controller_cost_sum_s=sum(self.__dict__.get('_experiment_controller_costs',[])))
            costs.clear()
            self.__dict__.setdefault('_experiment_controller_costs',[]).clear()
        return result

    def _online_control(self,*args,**kwargs):
        began=time.perf_counter()
        try:
            return super()._online_control(*args,**kwargs)
        finally:
            self.__dict__.setdefault('_experiment_controller_costs',[]).append(time.perf_counter()-began)
