from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from runtime.memory_budget import guard_memory


class BudgetTests(unittest.TestCase):
    def test_zero_process_swap_and_reserve(self):
        with tempfile.TemporaryDirectory() as temp:
            group = Path(temp)
            (group / "memory.swap.max").write_text("0\n")
            (group / "memory.swap.current").write_text("0\n")
            with patch("runtime.memory_budget.cgroup_path", return_value=group), \
                 patch("runtime.memory_budget.available_memory", return_value=16 * 2**30):
                guard_memory(12 * 2**30)
                (group / "memory.swap.max").write_text("max\n")
                with self.assertRaisesRegex(RuntimeError, "equal Podman"):
                    guard_memory(12 * 2**30)
                (group / "memory.swap.max").write_text("0\n")
                (group / "memory.swap.current").write_text("4096\n")
                with self.assertRaisesRegex(MemoryError, "swapping"):
                    guard_memory(12 * 2**30)

    def test_host_reserve(self):
        with patch("runtime.memory_budget.available_memory", return_value=8 * 2**30):
            with self.assertRaisesRegex(MemoryError, "reserve"):
                guard_memory(12 * 2**30)
