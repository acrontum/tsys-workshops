#!/usr/bin/env bash
set -euo pipefail
workshop_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime_dir="${T_SYSTEM_RUNTIME_ROOT:-$(dirname "$workshop_dir")/.runtime}"
isaac_python="${ISAAC_PYTHON:-$runtime_dir/env_isaac/bin/python}"
if [[ ! -x "$isaac_python" ]]; then
  echo 'Install the root runtime with ./install.sh, or set ISAAC_PYTHON.'
  echo 'Binary installation example: export ISAAC_PYTHON=/path/to/isaac-sim/python.sh'
  echo 'Configured Python environment example: export ISAAC_PYTHON=/path/to/venv/bin/python'
  exit 1
fi
if [[ -f "$runtime_dir/eula.accepted" ]]; then
  export OMNI_KIT_ACCEPT_EULA=YES
fi
"$isaac_python" -u "$workshop_dir/start_isaac.py" > "$workshop_dir/isaac.log" 2>&1 &
echo 'Isaac is opening. Wait for the empty scene.'
echo 'The application log is saved in isaac.log in this folder.'
