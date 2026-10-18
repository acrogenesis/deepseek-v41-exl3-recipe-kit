"""Desktop memory guard: respect this process's cgroup, leave global swap alone."""
from pathlib import Path
import os


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


class PersistentMemoryGuard:
    """Check the same limits with reused descriptors inside one fixed cgroup.

    The recipe's container does not migrate between cgroups. pread keeps these
    descriptors safe for callers on different threads without shared offsets.
    Values are read again on every check; this caches no memory measurement.
    """
    def __init__(self, reserve, *, meminfo=Path('/proc/meminfo')):
        self.reserve = reserve
        self.fds = []
        group = cgroup_path()
        try:
            for path in (meminfo, group/'memory.swap.max', group/'memory.swap.current'):
                self.fds.append(os.open(path, os.O_RDONLY | os.O_CLOEXEC))
            self.check()
        except BaseException:
            self.close()
            raise

    def check(self):
        if len(self.fds) != 3:
            raise RuntimeError('Memory guard is closed')
        info = os.pread(self.fds[0], 4096, 0)
        start = info.find(b'MemAvailable:')
        if start < 0:
            raise RuntimeError('Cannot read available RAM')
        available = int(info[start:info.find(b'\n', start)].split()[1]) * 1024
        if available < self.reserve:
            raise MemoryError('Host RAM reserve reached')
        if os.pread(self.fds[1], 32, 0).strip() != b'0':
            raise RuntimeError('Run with equal Podman --memory and --memory-swap limits (no process swap)')
        if int(os.pread(self.fds[2], 32, 0)) != 0:
            raise MemoryError('Model cgroup is swapping')

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()
