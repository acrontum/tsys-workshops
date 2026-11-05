#!/usr/bin/env bash
set -Eeuo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${T_SYSTEM_RUNTIME_ROOT:-$project_dir/.runtime}"
tools_dir="$runtime_dir/tools"
env_dir="$runtime_dir/env_isaac"
isaac_lab_dir="$runtime_dir/IsaacLab"

# shellcheck source=.support/runtime-versions.env
source "$project_dir/.support/runtime-versions.env"

accept_eula=0
install_system_packages=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [options]

Install the complete Isaac workshop runtime inside this repository.

Options:
  --accept-eula              Confirm that you reviewed and accept NVIDIA's
                             Omniverse/Isaac Sim EULA (required).
  --install-system-packages  Install required Ubuntu/Debian packages with sudo.
  -h, --help                 Show this help.

Example:
  ./install.sh --accept-eula --install-system-packages
EOF
}

while (($#)); do
  case "$1" in
    --accept-eula) accept_eula=1 ;;
    --install-system-packages) install_system_packages=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ((accept_eula == 0)); then
  cat >&2 <<'EOF'
Installation stopped: explicit EULA acceptance is required.

Review NVIDIA's Omniverse/Isaac Sim license terms, then run:
  ./install.sh --accept-eula --install-system-packages
EOF
  exit 2
fi

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "This installer supports x86_64 Linux only (Ubuntu 22.04/24.04 recommended)." >&2
  exit 1
fi

version_at_least() {
  [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" == "$2" ]]
}

glibc_version="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [[ -z "$glibc_version" ]] || ! version_at_least "$glibc_version" "2.35"; then
  echo "Isaac Sim 6 requires GLIBC 2.35 or newer; found ${glibc_version:-unknown}." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  cat >&2 <<'EOF'
No NVIDIA driver was detected. Install a current NVIDIA production driver for
your GPU, reboot, confirm that `nvidia-smi` works, and rerun this installer.
The script deliberately does not replace GPU drivers automatically.
EOF
  exit 1
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
printf 'Detected GPU: %s (driver %s)\n' "$gpu_name" "$driver_version"

available_kb="$(df -Pk "$project_dir" | awk 'NR==2 {print $4}')"
minimum_kb=$((45 * 1024 * 1024))
if ((available_kb < minimum_kb)); then
  printf 'At least 45 GiB free is required; only %.1f GiB is available.\n' \
    "$(awk -v kb="$available_kb" 'BEGIN {print kb/1024/1024}')" >&2
  exit 1
fi

base_packages=(
  ca-certificates curl git build-essential
  libgl1-mesa-dev libx11-dev libxcursor-dev libxi-dev
  libxinerama-dev libxrandr-dev
  python3-tk python3-pil
)

if ((install_system_packages == 1)); then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "--install-system-packages currently supports Ubuntu/Debian (apt) only." >&2
    exit 1
  fi
  sudo apt-get update
  sudo apt-get install -y "${base_packages[@]}"
fi

for command_name in curl git; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing command: $command_name" >&2
    echo "Rerun with --install-system-packages or install the prerequisite manually." >&2
    exit 1
  fi
done

mkdir -p "$tools_dir/bin" "$runtime_dir"
uv_bin="$tools_dir/bin/uv"
if [[ ! -x "$uv_bin" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
  else
    uv_installer="$runtime_dir/uv-installer.sh"
    if ! curl -LsSf https://astral.sh/uv/install.sh -o "$uv_installer"; then
      rm -f "$uv_installer"
      cat >&2 <<'EOF'
Could not download uv from https://astral.sh/uv/install.sh.
If curl reports "Could not resolve host", check your DNS, network, and proxy settings.
Alternatively, install uv through an accessible package source, ensure `uv --version`
works on your PATH, and rerun this installer.
EOF
      exit 1
    fi
    env UV_INSTALL_DIR="$tools_dir/bin" sh "$uv_installer"
    rm -f "$uv_installer"
  fi
fi

printf 'Using uv: %s\n' "$($uv_bin --version)"

if [[ ! -d "$isaac_lab_dir/.git" ]]; then
  git clone --branch "$ISAAC_LAB_GIT_REF" --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git "$isaac_lab_dir"
else
  current_ref="$(git -C "$isaac_lab_dir" describe --tags --always 2>/dev/null || true)"
  printf 'Keeping existing Isaac Lab checkout: %s (%s)\n' "$isaac_lab_dir" "$current_ref"
fi

if [[ ! -x "$env_dir/bin/python" ]]; then
  "$uv_bin" venv --python "$PYTHON_VERSION" --seed "$env_dir"
fi

python_bin="$env_dir/bin/python"

# NVIDIA wheels are large; allow slow connections while respecting user settings.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-300}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-5}"

"$uv_bin" pip install \
  --python "$python_bin" \
  "isaaclab[isaacsim,all]==$ISAAC_LAB_VERSION" \
  typeguard \
  --overrides "$project_dir/.support/$ISAAC_LAB_OVERRIDES_FILE" \
  --extra-index-url https://pypi.nvidia.com \
  --torch-backend "$TORCH_BACKEND" \
  --index-strategy unsafe-best-match

touch "$runtime_dir/eula.accepted"
"$python_bin" "$project_dir/.support/tools/check_dependencies.py"
"$python_bin" "$project_dir/.support/tools/resolve_task_id.py" > "$runtime_dir/task-id"
"$python_bin" "$project_dir/.support/tools/preflight_environment.py"

cat <<EOF

Installation complete.

Runtime:   $runtime_dir
Python:    $python_bin
Isaac Lab: $isaac_lab_dir

Next:
  Open README.md and follow either example's narrated video.
  Sorting: cd Isaac_Sorting_Workshop && bash open_isaac.sh
  RL:      cd example2_parallel_rl && ./workshop.sh doctor
EOF
