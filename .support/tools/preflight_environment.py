#!/usr/bin/env python3
"""Fail-fast health check for a portable T-System Isaac workshop environment."""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version
from check_dependencies import check_dependencies

REQUIRED_PACKAGES = (
    "isaacsim",
    "isaacsim-core",
    "isaaclab",
    "rsl-rl-lib",
    "torch",
    "torchvision",
    "torchaudio",
    "packaging",
    "Pillow",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        default="auto",
        help="Gymnasium task ID that must be registered, or 'auto'.",
    )
    args = parser.parse_args()

    failures: list[str] = []
    if sys.version_info[:2] != (3, 12):
        failures.append(
            f"Python 3.12 is required for Isaac Sim 6; found {sys.version.split()[0]}"
        )

    print("Workshop environment versions")
    installed: dict[str, str] = {}
    for package in REQUIRED_PACKAGES:
        try:
            actual = version(package)
        except PackageNotFoundError:
            actual = "MISSING"
        installed[package] = actual
        status = "OK" if actual != "MISSING" else "FAIL"
        print(f"  {package:<20} {actual:<18} {status}")
        if actual == "MISSING":
            failures.append(f"required package is missing: {package}")

    if installed["isaacsim"] != "MISSING" and Version(installed["isaacsim"]).major != 6:
        failures.append(f"Isaac Sim 6.x is required; found {installed['isaacsim']}")
    if installed["torch"] != "MISSING":
        torch_version = Version(installed["torch"])
        if torch_version < Version("2.11") or torch_version >= Version("2.12"):
            failures.append(f"PyTorch 2.11.x is required; found {installed['torch']}")

    dependency_failures, overrides = check_dependencies()
    failures.extend(dependency_failures)
    for override in overrides:
        print(f"  dependency override  {override}")
    if not dependency_failures:
        print("  dependency graph     valid with workshop overrides OK")

    try:
        import isaaclab_rl  # noqa: F401  # bundled in the published isaaclab wheel

        print("  Isaac Lab RL import  ready              OK")
    except Exception as exc:
        failures.append(f"Isaac Lab RL import failed: {exc}")

    try:
        import torch

        if not torch.cuda.is_available():
            failures.append("PyTorch cannot access CUDA")
        else:
            print(f"  CUDA GPU             {torch.cuda.get_device_name(0)}")
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(f"PyTorch/CUDA import failed: {exc}")

    try:
        import rsl_rl  # noqa: F401

        print("  RSL-RL import        ready              OK")
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(f"RSL-RL import failed: {exc}")

    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  # registers the task catalog

        if args.task == "auto":
            from resolve_task_id import resolve_task_id

            args.task = resolve_task_id()

        spec = gym.spec(args.task)
        print(f"  task registration    {spec.id} OK")
    except Exception as exc:  # pragma: no cover - diagnostic path
        failures.append(f"Task registration failed: {exc}")

    if failures:
        print("\nPREFLIGHT FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nPREFLIGHT PASSED — the workshop runtime is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
