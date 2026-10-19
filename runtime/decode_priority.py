"""Online RAM admission metadata that distinguishes decode from prefill scans.

Only observed decode accesses increase frequency. All records remain eligible
for execution; this changes which exact bytes are retained, never routing.
"""
import heapq


class DecodePriority:
    def __init__(self):
        self.counts = {}
        self.versions = {}
        self.heap = []
        self.clock = 0

    def _push(self, key):
        self.clock += 1
        self.versions[key] = self.clock
        heapq.heappush(self.heap, (self.counts.get(key, 0), self.clock, key))
        if len(self.heap) > 3 * len(self.versions) + 1024:
            self.heap = [(self.counts.get(k, 0), v, k) for k, v in self.versions.items()]
            heapq.heapify(self.heap)

    def admit(self, key):
        self._push(key)

    def observe_decode(self, key):
        self.counts[key] = self.counts.get(key, 0) + 1
        if key in self.versions:
            self._push(key)

    def forget_ram(self, key):
        self.versions.pop(key, None)

    def candidate(self, resident, gpu_owned):
        while self.heap:
            count, version, key = self.heap[0]
            if (key not in resident or key in gpu_owned or self.versions.get(key) != version
                    or self.counts.get(key, 0) != count):
                heapq.heappop(self.heap)
                continue
            return key
        raise RuntimeError('No eligible RAM eviction candidate; ownership metadata is incomplete')

    def clear(self):
        self.counts.clear()
        self.versions.clear()
        self.heap.clear()
