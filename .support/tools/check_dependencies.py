#!/usr/bin/env python3
"""Check installed base dependencies with narrowly scoped Isaac Lab overrides.

Like pip check, this validates base requirements, not unrecorded selected extras.
Only known metadata conflicts in the supported NVIDIA releases are overridden.
"""

from importlib.metadata import distributions
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

# (distribution, version, original requirement). Do not exempt whole packages:
# unrelated requirements and future releases must still be checked normally.
OVERRIDDEN_REQUIREMENTS = {
    ("isaacsim-kernel", "6.2.0.0", "numpy==2.3.1"),
    ("isaacsim-kernel", "6.2.0.0", "urllib3==2.7.0"),
    ("isaacsim-core", "6.2.0.0", "packaging<=26.0,>=23.2"),
    ("isaacsim-core", "6.2.0.0", "newton-usd-schemas==0.5.0"),
    ("isaacsim-core", "6.2.0.0", "fsspec==2026.2.0"),
    ("ovphysx", "0.6.4.72118723", "packaging<24,>=20"),
}


def check_dependencies(installed_distributions=None) -> tuple[list[str], list[str]]:
    if installed_distributions is None:
        installed_distributions = distributions()
    installed = {
        canonicalize_name(dist.metadata["Name"]): dist
        for dist in installed_distributions
    }
    override_path = Path(__file__).resolve().parents[1] / "runtime-overrides.txt"
    overrides = {}
    for line in override_path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            req = Requirement(line)
            overrides[canonicalize_name(req.name)] = req

    failures = []
    applied = []
    # Validate the override bounds as well as every original base requirement.
    for name, req in overrides.items():
        if name in installed and not req.specifier.contains(installed[name].version):
            failures.append(f"workshop requires {req}; found {installed[name].version}")
    for name, dist in installed.items():
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            target_name = canonicalize_name(req.name)
            target = installed.get(target_name)
            if target is None:
                failures.append(f"{name} {dist.version} requires {req}, which is not installed")
                continue
            if req.specifier.contains(target.version):
                continue
            key = (name, dist.version, str(req))
            replacement = overrides.get(target_name)
            if key in OVERRIDDEN_REQUIREMENTS and replacement is not None:
                if replacement.specifier.contains(target.version):
                    applied.append(f"{name}: {req} -> {replacement} (installed {target.version})")
                    continue
            failures.append(f"{name} {dist.version} requires {req}; found {target.version}")
    return failures, applied


if __name__ == "__main__":
    failures, applied = check_dependencies()
    for message in applied:
        print(f"Documented override: {message}")
    for message in failures:
        print(f"FAIL: {message}")
    if not failures:
        print("Dependencies satisfy workshop requirements and documented overrides.")
    raise SystemExit(bool(failures))
