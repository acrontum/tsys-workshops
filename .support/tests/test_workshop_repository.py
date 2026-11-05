"""Portable entrypoint and repository-boundary checks after the cleanup."""

from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[2]


class RepositoryTests(unittest.TestCase):
    def test_rl_help_uses_the_narrated_command_sequence(self):
        result = subprocess.run(
            ["bash", "example2_parallel_rl/workshop.sh", "help"], cwd=ROOT,
            capture_output=True, text=True, check=True,
        )
        expected = [
            "doctor", "setup", "randomize --resets 2", "data",
            "train", "test", "policy", "deploy",
        ]
        commands = [
            line.removeprefix("./workshop.sh ")
            for line in result.stdout.splitlines()
            if line.startswith("./workshop.sh ")
        ]
        self.assertEqual(commands, expected)

    def test_presentation_and_external_artifacts_are_ignored(self):
        for path in [
            "Isaac_Sorting_Workshop/video.mp4",
            "Isaac_Sorting_Workshop/result/report.json",
            "example2_parallel_rl/video.mp4",
            "example2_parallel_rl/assets/gear_base.usd",
            "example2_parallel_rl/checkpoints/trained.pth",
            "example2_parallel_rl/output/replicator/rgb.png",
            ".runtime/example",
        ]:
            result = subprocess.run(
                ["git", "check-ignore", path], cwd=ROOT, capture_output=True,
            )
            self.assertEqual(result.returncode, 0, path)

    def test_only_the_two_active_demo_folders_remain(self):
        self.assertTrue((ROOT / "Isaac_Sorting_Workshop/workshop.py").is_file())
        self.assertEqual(len(list((ROOT / "Isaac_Sorting_Workshop/cells").glob("*.py"))), 14)
        self.assertTrue((ROOT / "example2_parallel_rl/scripts/parallel_workshop.py").is_file())
        for name in [
            "demo", "sorting_demo", "example2_franka_rl",
            "physics_sorting_demo", "gear_insertion_rl",
            "local_materials", "client_summary", "workshop-runtime",
        ]:
            self.assertFalse((ROOT / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
