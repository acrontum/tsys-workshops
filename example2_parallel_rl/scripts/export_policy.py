"""Export a completed training checkpoint or the verified recorded policy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Recorded 12-update checkpoint: eight evaluation successes and two deployment
# successes. An interrupted new run may remove its old output report, but cannot
# invalidate these exact saved weights.
VERIFIED_RECORDED_SHA256 = (
    "577582c89d643fbc080d4ecbdf05d4b1ac1f7f0568714fab200f5a64ecbac948"
)


def checkpoint_provenance(source: Path, training_report: Path) -> str:
    if not source.is_file():
        raise SystemExit(
            "No trained checkpoint found. If checkpoints/franka_parallel_inference.pth "
            "is present, use ./workshop.sh deploy directly. Otherwise run "
            "./workshop.sh setup and ./workshop.sh train."
        )
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest == VERIFIED_RECORDED_SHA256:
        return "verified_recorded_policy"
    if training_report.is_file():
        try:
            training = json.loads(training_report.read_text())
        except (ValueError, OSError):
            training = {}
        if isinstance(training, dict) and digest == training.get("checkpoint_sha256"):
            return "completed_training_report"
    raise SystemExit(
        "This checkpoint matches neither a completed training report nor the "
        "verified recorded policy. To use the existing exported policy, run "
        "./workshop.sh deploy. Complete training before exporting new weights."
    )


def main() -> None:
    source = ROOT / "checkpoints/franka_parallel_trained.pth"
    destination = ROOT / "checkpoints/franka_parallel_inference.pth"
    provenance = checkpoint_provenance(source, ROOT / "output/training_report.json")

    import torch

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    torch.save({"model": checkpoint["model"], "env_state": None}, destination)
    with destination.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    spec = {
        "task": "IsaacContrib-Factory-GearMesh-Direct",
        "robot": "Franka Panda",
        "simulation_only": True,
        "policy": destination.name,
        "policy_sha256": digest,
        "observations": 19,
        "actions": 6,
        "control_rate_hz": 15,
        "orientation_at_network_input": "WXYZ (adapter converts from Isaac Lab 3 XYZW)",
        "action_meaning": "Normalized incremental hand translation and axis-angle rotation; task controller applies torques.",
        "recurrent_policy": "Two-layer LSTM; initialize once and reset at episode boundaries.",
        "normalizer": "Included in model state; do not remove.",
        "starting_state": "Medium gear already grasped above the fixture; neighbouring gears already present.",
        "tested_runtime": "Isaac Sim 6.2 / Isaac Lab 3.0.0 / pinned RL Games 1.6.1 fork",
        "source_training_checkpoint": source.name,
        "source_validation": provenance,
    }
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "deployment_spec.json").write_text(json.dumps(spec, indent=2))
    print("POLICY SOURCE:", provenance, flush=True)
    print("POLICY EXPORTED", destination.name, destination.stat().st_size, "bytes", flush=True)
    print(
        "Normalizer and recurrent model included. "
        "Deploy on the simulated Franka with ./workshop.sh deploy",
        flush=True,
    )


if __name__ == "__main__":
    main()
