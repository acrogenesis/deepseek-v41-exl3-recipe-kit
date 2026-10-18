"""Bounded RAM LRU over original EXL3 safetensors; no duplicate expert bank.

Metadata is CPU-only. Payload reads use aligned O_DIRECT on Linux by default,
so the 64 GiB explicit cache does not also populate 64 GiB of file page cache.
"""
from collections import OrderedDict, Counter
from contextlib import contextmanager
import json
import math
import mmap
import os
from pathlib import Path
import re
import struct
import threading

from .vendor.expert_store import ALIGN, WIDTH, SUFFIXES, aligned, validate_tensors

EXPERT = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(.+)")
MARKER = -2082680531


def read_header(path, file_size=None):
    with Path(path).open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("Missing safetensors header length")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= 64 * 2**20:
            raise ValueError("Invalid safetensors header size")
        raw = f.read(length)
        if len(raw) != length:
            raise ValueError("Truncated safetensors header")
    header = json.loads(raw)
    if not isinstance(header, dict):
        raise ValueError("Expected header object")
    if file_size is not None and length + 8 > file_size:
        raise ValueError("Header exceeds shard size")
    return length + 8, header


def make_records(weight_map, headers, *, layers=40, experts=384, hidden=5120, intermediate=2304):
    """Validate complete routed ownership and physical tensor geometry."""
    groups = {}
    for name, filename in weight_map.items():
        match = EXPERT.fullmatch(name)
        if not match:
            continue
        layer, expert, projection, suffix = match.groups()
        if not (0 <= int(layer) < layers and 0 <= int(expert) < experts):
            raise ValueError(f"Unexpected expert: {name}")
        if suffix not in SUFFIXES:
            raise ValueError(f"Unsupported expert tensor: {name}")
        base, header, size = headers[filename]
        spec = header[name]
        shape = spec["shape"]
        if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
            raise ValueError(f"Invalid shape: {name}")
        nbytes = math.prod(shape) * WIDTH[spec["dtype"]]
        start, end = spec["data_offsets"]
        if type(start) is not int or type(end) is not int or end - start != nbytes or not 0 <= start < end <= size - base:
            raise ValueError(f"Invalid source extent: {name}")
        key = f"{int(layer)}:{int(expert)}"
        group = groups.setdefault(key, {})
        group[f"{projection}.{suffix}"] = dict(dtype=spec["dtype"], shape=shape,
            bytes=nbytes, source_file=filename, source_offset=base + start)
    expected = {f"{l}:{e}" for l in range(layers) for e in range(experts)}
    if set(groups) != expected:
        raise ValueError(f"Incomplete expert ownership: {len(groups)} of {len(expected)}")
    records = {}
    for key, tensors in groups.items():
        validate_tensors(tensors)
        if tensors["w1.suh"]["shape"] != [hidden] or tensors["w1.svh"]["shape"] != [intermediate]:
            raise ValueError(f"Wrong model geometry: {key}")
        offset = 0
        for p in ("w1", "w2", "w3"):
            for suffix in SUFFIXES:
                spec = tensors[f"{p}.{suffix}"]
                offset = aligned(offset, 256)
                spec["offset"] = offset
                offset += spec["bytes"]
        records[key] = dict(bytes=aligned(offset), tensors=tensors)
    return records


def inventory(root, *, cached_headers=False):
    root = Path(root)
    lock = json.loads((root / "models.lock.json").read_text())
    model = next(m for m in lock["models"] if m["id"] == "exl3")
    directory = root / "work/models/exl3"
    weight_map = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    headers = {}
    for spec in model["files"]:
        if spec["name"].endswith(".safetensors"):
            path = root / "work/headers/exl3" / (spec["name"] + ".header") if cached_headers else directory / spec["name"]
            base, header = read_header(path, spec["bytes"])
            headers[spec["name"]] = (base, header, spec["bytes"])
    records = make_records(weight_map, headers)
    counts = Counter(str(t["shape"][2] // 16) for r in records.values() for n, t in r["tensors"].items() if n.endswith(".trellis"))
    return records, {"model": model["repo"], "revision": model["revision"], "experts": len(records),
        "tensor_count": sum(len(r["tensors"]) for r in records.values()), "projection_K_counts": dict(counts),
        "packed_record_bytes": sum(r["bytes"] for r in records.values()),
        "largest_record_bytes": max(r["bytes"] for r in records.values()),
        "payload_verified": False, "header_source": "range_download" if cached_headers else "local_shards", "gpu_used": False}


class IndexedExpertStore:
    """C1/eager store. A read lease holds the lock until its consumer is done.

    Cached bytearrays are private and must not escape read(). RAM is allocated
    only on misses. One aligned I/O staging buffer lives alongside the LRU.
    """
    def __init__(self, directory, records, capacity_bytes, *, direct=True, reserve_bytes=12 * 2**30):
        self.directory = Path(directory)
        self.records = records
        self.capacity = capacity_bytes
        if not records or capacity_bytes < max(r["bytes"] for r in records.values()):
            raise ValueError("RAM cache must fit the largest expert")
        self.direct = direct
        self.reserve = reserve_bytes
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.resident = 0
        self.fds = {}
        self.stats = dict(hits=0, misses=0, evictions=0, read_bytes=0, read_calls=0, peak_resident_bytes=0)
        # The checkpoint stores most experts as twelve adjacent tensors. Read
        # each contiguous source extent once, retaining the validated packed
        # layout and padding used by the GPU cache. Never read across a gap.
        self.extents = {}
        for key, record in records.items():
            groups = []
            for name, spec in sorted(record['tensors'].items(),
                    key=lambda item: (item[1]['source_file'], item[1]['source_offset'])):
                if (groups and groups[-1][-1][1]['source_file'] == spec['source_file']
                        and groups[-1][-1][1]['source_offset'] + groups[-1][-1][1]['bytes'] == spec['source_offset']):
                    groups[-1].append((name, spec))
                else:
                    groups.append([(name, spec)])
            self.extents[key] = groups

    def _fd(self, name):
        path = (self.directory / name).resolve()
        if path.parent != self.directory.resolve():
            raise ValueError("Shard path escapes model directory")
        if name not in self.fds:
            fd = os.open(path, os.O_RDONLY | (os.O_DIRECT if self.direct else 0))
            st = os.fstat(fd)
            self.fds[name] = fd, (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
        fd, identity = self.fds[name]
        st = os.fstat(fd)
        current = path.stat()
        if (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) != identity or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != identity:
            raise RuntimeError("Source shard changed after opening")
        return fd

    def _read_extent(self, group, target, cancel):
        first, last = group[0][1], group[-1][1]
        fd = self._fd(first['source_file'])
        start = first['source_offset']
        length = last['source_offset'] + last['bytes'] - start
        base = start // ALIGN * ALIGN if self.direct else start
        delta = start - base
        size = aligned(delta + length) if self.direct else length
        with mmap.mmap(-1, size) as staging:
            n = os.preadv(fd, [staging], base)
            self.stats['read_calls'] += 1
            if n < delta + length:
                raise OSError("Truncated expert tensor")
            self.stats['read_bytes'] += n
            with memoryview(staging) as source:
                for name, spec in group:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError('Read cancelled')
                    begin = spec['source_offset'] - base
                    if name.endswith('.mul1') and struct.unpack_from('<i', source, begin)[0] != MARKER:
                        raise ValueError('Wrong MUL1 codebook marker')
                    target[spec['offset']:spec['offset'] + spec['bytes']] = source[begin:begin + spec['bytes']]

    def _evict(self):
        _, value = self.cache.popitem(last=False)
        self.resident -= len(value)
        self.stats["evictions"] += 1

    @contextmanager
    def read(self, key, cancel=None):
        with self.lock:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Read cancelled")
            if key in self.cache:
                self.stats["hits"] += 1
                self.cache.move_to_end(key)
            else:
                record = self.records[key]
                self.stats["misses"] += 1
                while self.resident + record["bytes"] > self.capacity:
                    self._evict()
                if self.reserve:
                    from .memory_budget import available_memory
                    while self.cache and available_memory() < self.reserve + 2 * record["bytes"]:
                        self._evict()
                    if available_memory() < self.reserve + 2 * record["bytes"]:
                        raise MemoryError("RAM reserve reached")
                data = bytearray(record["bytes"])
                view = memoryview(data)
                try:
                    for group in self.extents[key]:
                        if cancel is not None and cancel.is_set():
                            raise InterruptedError("Read cancelled")
                        self._read_extent(group, view, cancel)
                finally:
                    view.release()
                self.cache[key] = data
                self.resident += len(data)
                self.stats["peak_resident_bytes"] = max(self.stats["peak_resident_bytes"], self.resident)
            leased = memoryview(self.cache[key])
            try:
                yield leased
            finally:
                leased.release()

    def close(self):
        with self.lock:
            for fd, _ in self.fds.values():
                os.close(fd)
            self.fds.clear()
            self.cache.clear()
            self.resident = 0
