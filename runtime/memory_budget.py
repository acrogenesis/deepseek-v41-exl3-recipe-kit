"""Desktop memory guard: respect this process's cgroup, leave global swap alone."""
from pathlib import Path


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read available RAM")


def cgroup_path():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            relative = line.split("::", 1)[1].lstrip("/")
            if ".." in Path(relative).parts:
                raise RuntimeError("Unresolved cgroup namespace")
            return Path("/sys/fs/cgroup") / relative
    raise RuntimeError("cgroup v2 is required")


def guard_memory(reserve):
    if available_memory() < reserve:
        raise MemoryError("Host RAM reserve reached")
    group = cgroup_path()
    if (group / "memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("Run with equal Podman --memory and --memory-swap limits (no process swap)")
    if int((group / "memory.swap.current").read_text()) != 0:
        raise MemoryError("Model cgroup is swapping")
