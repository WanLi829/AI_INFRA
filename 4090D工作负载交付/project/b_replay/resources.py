"""Container-aware RAM observations; swap is never counted as available RAM."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import re

GIB = 1024 ** 3


def _fields(path):
    result = {}
    for line in path.read_text().splitlines():
        parts = line.replace(":", "").split()
        if len(parts) >= 2:
            result[parts[0]] = int(parts[1])
    return result


def _limit(path):
    raw = path.read_text().strip()
    value = None if raw == "max" else int(raw)
    if value is not None and value < 0:
        raise ValueError(f"Negative memory limit: {path}")
    # v1 uses a value near LONG_MAX for an unlimited controller.
    return None if value is None or value >= 2 ** 60 else value


def _unescape(value):
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), value)


def memory_snapshot(proc=Path("/proc")):
    """Read both the leaf and visible ancestors, including namespaced mounts.

    Available RAM is bounded by host MemAvailable and each controller's
    limit-current+inactive_file. Only inactive file cache is reclaimable here;
    active file cache, swap and anonymous pages are not treated as free RAM.
    Unreadable advertised controller counters fail closed.
    """
    proc = Path(proc)
    host = _fields(proc / "meminfo")
    total = host["MemTotal"] * 1024
    available = host["MemAvailable"] * 1024
    if not 0 <= available <= total:
        raise ValueError("Invalid /proc/meminfo RAM counters")
    memberships = []
    for line in (proc / "self/cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            memberships.append((2 if not controllers else 1, PurePosixPath(path)))
    # On hybrid hosts the unified hierarchy can exist without carrying the
    # memory controller. An explicit v1 memory membership identifies its owner.
    if any(version == 1 for version, _ in memberships):
        memberships = [(version, path) for version, path in memberships if version == 1]
    mounts = []
    for line in (proc / "self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] == "cgroup2" or (fs[0] == "cgroup" and "memory" in fs[2].split(",")):
            mounts.append((2 if fs[0] == "cgroup2" else 1,
                           PurePosixPath(_unescape(fields[3])), Path(_unescape(fields[4]))))
    rows, seen = [], set()
    for version, membership in memberships:
        matched = False
        for mount_version, root, mount in mounts:
            if version != mount_version:
                continue
            # A cgroup namespace can report '/' while mountinfo retains the
            # original subtree root. For other paths require a real directory.
            relative = membership.relative_to(root) if membership.is_relative_to(root) else membership.relative_to("/")
            if ".." in relative.parts:
                raise ValueError("Cannot resolve cgroup namespace safely")
            leaf = mount.joinpath(*relative.parts)
            if not leaf.is_dir():
                continue
            matched = True
            node = leaf
            while True:
                key = (version, str(node))
                if key not in seen:
                    seen.add(key)
                    maximum = node / ("memory.max" if version == 2 else "memory.limit_in_bytes")
                    current_path = node / ("memory.current" if version == 2 else "memory.usage_in_bytes")
                    if maximum.exists() or current_path.exists():
                        hard = _limit(maximum)
                        high_path = node / "memory.high"
                        high = _limit(high_path) if version == 2 and high_path.exists() else None
                        current = int(current_path.read_text())
                        if current < 0:
                            raise ValueError("Negative cgroup memory usage")
                        stat_path = node / "memory.stat"
                        stats = _fields(stat_path) if stat_path.exists() else {}
                        inactive = max(0, stats.get("inactive_file", 0) if version == 2 else
                                       stats.get("total_inactive_file", stats.get("inactive_file", 0)))
                        inactive = min(current, inactive)
                        limits = [n for n in (hard, high) if n is not None]
                        limit = min(limits) if limits else None
                        row = dict(path=str(node), version=version, max_bytes=hard,
                                   high_bytes=high, limit_bytes=limit, current_bytes=current,
                                   inactive_file_bytes=inactive)
                        peak = node / ("memory.peak" if version == 2 else "memory.max_usage_in_bytes")
                        if peak.exists():
                            row["historical_peak_bytes"] = int(peak.read_text())
                        if limit is not None:
                            row["raw_headroom_bytes"] = max(0, limit - current)
                            row["available_bytes"] = max(0, limit - current + inactive)
                            available = min(available, row["available_bytes"])
                            total = min(total, limit)
                        events = node / ("memory.events" if version == 2 else "memory.failcnt")
                        if events.exists():
                            row["events"] = _fields(events) if version == 2 else {"failcnt": int(events.read_text())}
                        oom_control = node / "memory.oom_control"
                        if version == 1 and oom_control.exists():
                            row.setdefault("events", {}).update(_fields(oom_control))
                        rows.append(row)
                if node == mount:
                    break
                node = node.parent
        if not matched:
            raise RuntimeError("Cannot locate the process memory cgroup; refusing host-only RAM check")
    return dict(host_total_bytes=host["MemTotal"] * 1024,
                host_available_bytes=host["MemAvailable"] * 1024,
                effective_limit_bytes=total, available_bytes=min(available, total), cgroups=rows)


def require_memory(snapshot, minimum, *, phase):
    if snapshot["available_bytes"] < minimum:
        raise RuntimeError(f"{phase}: require {minimum / GIB:g} GiB available RAM after container limits; "
                           f"observed {snapshot['available_bytes'] / GIB:.2f} GiB")
