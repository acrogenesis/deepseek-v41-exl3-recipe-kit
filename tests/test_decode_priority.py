import random
import unittest

from runtime.decode_priority import DecodePriority


class DecodePriorityTests(unittest.TestCase):
    def test_prefill_scan_retains_observed_decode_working_set(self):
        policy = DecodePriority()
        resident = {}
        for key in range(8):
            resident[key] = None
            policy.admit(key)
        for key in (0, 1, 2):
            policy.observe_decode(key)
        for key in range(8, 200):
            victim = policy.candidate(resident, set())
            resident.pop(victim)
            policy.forget_ram(victim)
            resident[key] = None
            policy.admit(key)
        self.assertTrue({0, 1, 2}.issubset(resident))
        self.assertEqual(sum(policy.counts.values()), 3)

    def test_gpu_ownership_and_return_to_ram(self):
        policy = DecodePriority()
        resident = {'a': None, 'b': None}
        for key in resident:
            policy.admit(key)
        self.assertEqual(policy.candidate(resident, {'a'}), 'b')
        policy.admit('a')  # demoted: it is eligible again
        policy.observe_decode('b')
        self.assertEqual(policy.candidate(resident, set()), 'a')

    def test_recycled_key_cannot_use_stale_priority(self):
        policy = DecodePriority()
        resident = {'a': None, 'b': None}
        policy.admit('a')
        policy.admit('b')
        policy.forget_ram('a')
        policy.observe_decode('a')
        policy.admit('a')
        self.assertEqual(policy.candidate(resident, set()), 'b')

    def test_random_accesses_match_explicit_priority_order(self):
        rng = random.Random(1017)
        policy = DecodePriority()
        resident, gpu = {}, set()
        for _ in range(5000):
            key = rng.randrange(100)
            op = rng.randrange(5)
            if op == 0:
                resident[key] = None
                policy.admit(key)
            elif op == 1:
                policy.observe_decode(key)
            elif op == 2:
                resident.pop(key, None)
                policy.forget_ram(key)
            elif op == 3:
                gpu.add(key)
            else:
                gpu.discard(key)
                if key in resident:
                    policy.admit(key)
            eligible = set(resident) - gpu
            if eligible:
                expected = min(eligible, key=lambda k: (policy.counts.get(k, 0), policy.versions[k]))
                self.assertEqual(policy.candidate(resident, gpu), expected)
            else:
                with self.assertRaises(RuntimeError):
                    policy.candidate(resident, gpu)
            self.assertLessEqual(len(policy.heap), 3 * len(policy.versions) + 1024)


if __name__ == '__main__':
    unittest.main()
