"""CPU I/O and ownership bookkeeping. CUDA registration is tested separately."""
from pathlib import Path
import tempfile
import unittest

from runtime.phase_store import PhaseExpertStore
from test_indexed_store import fixture


class CPUPhaseStore(PhaseExpertStore):
    def _allocate(self, size):
        return bytearray(size)

    def _release(self, data):
        pass


class PhaseStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _, _, self.records, self.values = fixture(self.root / 'model.safetensors')
        largest = max(r['bytes'] for r in self.records.values())
        self.store = CPUPhaseStore(self.root, self.records, largest * 2,
            direct=False, reserve_bytes=0, pinned=True, gpu_only_limit_bytes=largest * 2)

    def tearDown(self):
        self.store.close()
        self.temp.cleanup()

    def check_bytes(self, key):
        with self.store.read(key) as data:
            for name, spec in self.records[key]['tensors'].items():
                expected = self.values[f"layers.0.ffn.experts.{key.split(':')[1]}.{name}"]
                self.assertEqual(bytes(data[spec['offset']:spec['offset'] + spec['bytes']]), expected)
        self.assertLessEqual(self.store.resident, self.store.capacity)

    def test_prefill_cannot_train_frequency_or_displace_hot_ram(self):
        self.check_bytes('0:0')
        self.store.set_phase('decode')
        for _ in range(10):
            self.store.touch_gpu('0:0')
        self.store.set_phase('prefill')
        for _ in range(20):
            for key in ('0:1', '0:2'):
                self.store.touch_gpu(key)
                self.check_bytes(key)
                self.assertIn('0:0', self.store.cache)
        self.assertEqual(self.store.priority.counts, {'0:0': 10})
        self.assertEqual(self.store.stats['decode_observations'], 10)
        self.assertGreater(self.store.stats['priority_evictions'], 0)

    def test_duplicate_eviction_preserves_gpu_ownership(self):
        self.check_bytes('0:0')
        self.store.set_phase('decode')
        self.store.touch_gpu('0:0')
        self.store.mark_gpu('0:0')
        self.store.set_phase('prefill')
        for key in ('0:1', '0:2'):
            self.check_bytes(key)
        self.assertNotIn('0:0', self.store.cache)
        self.assertIn('0:0', self.store.gpu_owned)
        self.assertEqual(self.store.gpu_only_bytes, self.records['0:0']['bytes'])
        # A RAM reload becomes a duplicate again without losing ownership.
        self.check_bytes('0:0')
        self.assertIn('0:0', self.store.duplicates)
        self.assertEqual(self.store.gpu_only_bytes, 0)

    def test_transition_is_not_a_decode_access(self):
        self.store.set_phase('transition')
        for key in self.records:
            self.store.touch_gpu(key)
            self.check_bytes(key)
        self.assertFalse(self.store.priority.counts)
        self.assertFalse(self.store.frequencies)

    def test_hot_heap_rebuild_accepts_untrained_prefill_duplicates(self):
        for key in ('0:0', '0:1'):
            self.check_bytes(key)
            self.store.touch_gpu(key)
            self.store.mark_gpu(key)
        self.assertFalse(self.store.frequencies)
        self.store.set_phase('decode')
        for _ in range(50):
            self.store.touch_gpu('0:0')
        self.assertEqual(self.store.frequencies, {'0:0': 50})
        self.assertIn((0, '0:1'), self.store.hot_duplicates)
        self.assertEqual(set(self.store.duplicates), {'0:0', '0:1'})
        self.assertEqual(self.store.gpu_only_bytes, 0)
        self.check_bytes('0:1')


if __name__ == '__main__':
    unittest.main()
