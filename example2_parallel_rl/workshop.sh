#!/usr/bin/env bash
set -euo pipefail
workshop_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
command_name="${1:-help}"
if (($#)); then shift; fi
case "$command_name" in
 help|-h|--help)
  echo "Follow video.mp4. Run each command separately and wait until it finishes:"
  echo "./workshop.sh doctor"
  echo "./workshop.sh setup"
  echo "./workshop.sh randomize --resets 2"
  echo "./workshop.sh data"
  echo "./workshop.sh train"
  echo "./workshop.sh test"
  echo "./workshop.sh policy"
  echo "./workshop.sh deploy"
  echo "Live commands use fast rendering. Add --render_quality standard for the original view."
  echo "Deploy also reuses GPU controller work; add --playback_mode reference for the original loop."
  echo "For training without a viewport: workshop.sh train-headless --num_envs 8"
  exit 0 ;;
 doctor|setup|randomize|snapshot|train|train-headless|test|policy|deploy|data) ;;
 *) echo "Unknown command: $command_name. Use ./workshop.sh help." >&2; exit 2 ;;
esac
runtime_dir="${T_SYSTEM_RUNTIME_ROOT:-$(dirname "$workshop_dir")/.runtime}"
py="$runtime_dir/env_isaac/bin/python"
[[ -x "$py" ]] || { echo "Install the root Isaac runtime with ./install.sh first."; exit 1; }
mkdir -p "$workshop_dir/output" "$workshop_dir/checkpoints"
export OMNI_KIT_ACCEPT_EULA=YES TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
case "$command_name" in
 doctor) "$py" "$workshop_dir/scripts/doctor.py" "$@" ;;
 setup) "$py" "$workshop_dir/scripts/setup_workshop.py" "$@" ;;
 randomize) "$py" -u "$workshop_dir/scripts/parallel_workshop.py" --mode resets --num_envs 8 --resets 3 --hold_seconds 12 --visualizer kit "$@" 2>&1 | tee "$workshop_dir/output/randomize_live.log" ;;
 snapshot) "$py" -u "$workshop_dir/scripts/parallel_workshop.py" --mode snapshot --num_envs 8 --resets 1 --hold_seconds 0 --visualizer none "$@" ;;
 train) "$py" -u "$workshop_dir/scripts/parallel_workshop.py" --mode train --num_envs 4 --iterations 12 --visualizer kit "$@" 2>&1 | tee "$workshop_dir/output/training_live.log" ;;
 train-headless) "$py" -u "$workshop_dir/scripts/parallel_workshop.py" --mode train --num_envs 8 --iterations 12 --visualizer none "$@" 2>&1 | tee "$workshop_dir/output/training_live.log" ;;
 test) "$py" -u "$workshop_dir/scripts/parallel_workshop.py" --mode evaluate --num_envs 4 --episodes 2 --seed 2026 --visualizer none --checkpoint "$workshop_dir/checkpoints/franka_parallel_trained.pth" "$@" 2>&1 | tee "$workshop_dir/output/test_live.log" ;;
 policy) "$py" "$workshop_dir/scripts/export_policy.py" "$@" ;;
 deploy) "$py" -u "$workshop_dir/scripts/parallel_workshop.py" --mode deploy --num_envs 1 --episodes 2 --seed 2026 --visualizer kit --checkpoint "$workshop_dir/checkpoints/franka_parallel_inference.pth" "$@" 2>&1 | tee "$workshop_dir/output/deployment_live.log" ;;
 data) "$py" -u "$workshop_dir/scripts/generate_sdg.py" "$@" 2>&1 | tee "$workshop_dir/output/data_live.log" ;;
esac
