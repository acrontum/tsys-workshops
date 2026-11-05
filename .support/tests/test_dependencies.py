"""Dependency policy regression tests without GPU or NVIDIA imports."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from check_dependencies import check_dependencies


def dist(name, version, *requires):
    return SimpleNamespace(metadata={"Name": name}, version=version, requires=requires)


class DependencyTests(unittest.TestCase):
    def test_known_nvidia_override_is_reported(self):
        failures, applied = check_dependencies([
            dist("isaacsim-kernel", "6.2.0.0", "urllib3==2.7.0"),
            dist("urllib3", "2.8.0"),
        ])
        self.assertEqual(failures, [])
        self.assertEqual(len(applied), 1)

    def test_bad_override_version_fails(self):
        failures, _ = check_dependencies([
            dist("isaacsim-kernel", "6.2.0.0", "urllib3==2.7.0"),
            dist("urllib3", "2.6.0"),
        ])
        self.assertTrue(failures)

    def test_numba_conflict_is_never_exempted(self):
        failures, _ = check_dependencies([
            dist("numba", "0.64.0", "numpy>=1.22,<2.5"),
            dist("numpy", "2.5.4"),
        ])
        self.assertTrue(any("numba" in failure for failure in failures))

    def test_missing_typeguard_fails(self):
        failures, _ = check_dependencies([
            dist("generate-parameter-library-py", "0.5.0", "typeguard"),
        ])
        self.assertTrue(any("typeguard" in failure for failure in failures))

    def test_future_release_is_not_exempted(self):
        failures, _ = check_dependencies([
            dist("isaacsim-kernel", "6.3.0.0", "urllib3==2.7.0"),
            dist("urllib3", "2.8.0"),
        ])
        self.assertTrue(failures)

    def test_repaired_numpy_and_typeguard(self):
        failures, _ = check_dependencies([
            dist("isaacsim-kernel", "6.2.0.0", "numpy==2.3.1"),
            dist("numba", "0.64.0", "numpy>=1.22,<2.5"),
            dist("numpy", "2.3.1"),
            dist("generate-parameter-library-py", "0.5.0", "typeguard"),
            dist("typeguard", "4.4.4"),
            dist("example", "1.0", 'missing; extra == "optional"'),
        ])
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
