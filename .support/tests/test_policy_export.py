"""Interrupted training must not block the verified recorded checkpoint."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "workshop_policy_export", ROOT / "example2_parallel_rl/scripts/export_policy.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class PolicyExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        folder = Path(self.temporary.name)
        self.source = folder / "trained.pth"
        self.source.write_bytes(b"recorded test checkpoint")
        self.report = folder / "training_report.json"
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_verified_checkpoint_works_without_a_training_report(self):
        with patch.object(export, "VERIFIED_RECORDED_SHA256", self.digest):
            self.assertEqual(
                export.checkpoint_provenance(self.source, self.report),
                "verified_recorded_policy",
            )

    def test_verified_checkpoint_survives_an_unrelated_training_report(self):
        self.report.write_text(json.dumps({"checkpoint_sha256": "other-checkpoint"}))
        with patch.object(export, "VERIFIED_RECORDED_SHA256", self.digest):
            self.assertEqual(
                export.checkpoint_provenance(self.source, self.report),
                "verified_recorded_policy",
            )

    def test_new_checkpoint_requires_a_matching_completed_report(self):
        self.report.write_text(json.dumps({"checkpoint_sha256": self.digest}))
        self.assertEqual(
            export.checkpoint_provenance(self.source, self.report),
            "completed_training_report",
        )

    def test_unknown_checkpoint_without_report_is_rejected(self):
        with self.assertRaises(SystemExit):
            export.checkpoint_provenance(self.source, self.report)

    def test_checkpoint_with_a_mismatched_report_is_rejected(self):
        self.report.write_text(json.dumps({"checkpoint_sha256": "other-checkpoint"}))
        with self.assertRaises(SystemExit):
            export.checkpoint_provenance(self.source, self.report)


if __name__ == "__main__":
    unittest.main()
