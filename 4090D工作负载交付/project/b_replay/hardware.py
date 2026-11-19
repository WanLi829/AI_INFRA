"""The selected single-H800 host, independent of archived RTX profiles."""
import contextlib
import math
import os
from pathlib import Path
import signal
import time

HARDWARE = dict(gpu='NVIDIA H800 80GB', gpu_count=1, cpu_cores=27,
                advertised_ram_bytes=230*1024**3, engine_cores=23, helper_cores=4)


def validate_gpu(gpu):
    name = ''.join(gpu['name'].upper().split())
    if 'H800' not in name or not 76000 <= int(gpu['total']) <= 84000:
        raise RuntimeError('Expected the selected H800 80GB, observed '+str(gpu))
    return gpu


def cpu_inventory():
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    quota = None
    path = Path('/sys/fs/cgroup/cpu.max')
    if path.is_file():
        fields = path.read_text().split()
        if fields[0] != 'max':
            quota = int(fields[0])/int(fields[1])
    else:
        limit=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us')
        period=Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
        if limit.is_file() and period.is_file() and int(limit.read_text())>0:
            quota=int(limit.read_text())/int(period.read_text())
    return dict(allowed_cpu_ids=allowed, quota_cores=quota,
                effective_cores=min(len(allowed), quota) if allowed and quota else len(allowed) if allowed else None)


@contextlib.contextmanager
def deadline_guard(deadline_epoch=None):
    """Interrupt the owning launcher; its finally block stops only its workers."""
    if deadline_epoch is None:
        yield
        return
    remaining = deadline_epoch-time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError('The batch execution deadline has passed')
    if not hasattr(signal, 'setitimer'):
        raise RuntimeError('Timed GPU execution requires Linux signal timers')
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError('Batch execution deadline reached; preserving the failed attempt')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
