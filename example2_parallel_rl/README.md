# Example 2 — follow `video.mp4`

Following commands will help initialize your project. They reproduce the narrated eight-Franka workshop. Run one command at a
time in this folder and wait until it finishes.

```bash
./workshop.sh doctor
./workshop.sh setup
./workshop.sh randomize --resets 2
./workshop.sh data
./workshop.sh train
./workshop.sh test
./workshop.sh policy
./workshop.sh deploy
```
| Step | What you will see |
|---|---|
| `doctor` | Check the installed runtime and NVIDIA GPU. Prints missing dependencies and how to install them.|
| `setup` | Verify the NIST reference assets and published starting policy. |
| `randomize` | Eight workcells with independently randomized insertion starts. |
| `data` | Generate 16 RGB, label and depth captures from the eight starts. |
| `train` | Four active Frankas collect 1,536 transitions and perform 12 PPO updates. Add `--num_envs 8` for the recorded eight-Franka, 3,072-transition run. Default value is 4. |
| `test` | Evaluate the resulting policy on eight fresh trials. |
| `policy` | Export the actor with its observation normalizer and recurrent memory. |
| `deploy` | Load that policy onto one simulated Franka and run two insertion trials. |

To use the saved working policy immediately, run this command in this folder:

```bash
./workshop.sh deploy
```

This opens one simulated Franka and runs two insertion trials. It uses the
already exported policy; training and export are optional for this replay.
`./workshop.sh policy` can also re-export the verified recorded checkpoint,
even if an interrupted training run removed `output/training_report.json`.

Deployment now reuses the GPU work for arm targets and torque control, skips PPO
reward calculations, and treats the rear display board as a visual reference.
The saved policy, insertion contact geometry, physics timestep, controller rate,
192 solver iterations and eight consecutive success checks are preserved.
Training and evaluation use the original Factory control and reward calculations.

Live commands use RTX Minimal textured shading and a 960 × 600 render image.
Deployment displays every policy step; other fast live commands display every
two steps. The camera still shows the gear insertion closely.

On this machine, the same single GUI insertion took 61.5 seconds instead of
100.4 seconds, with identical final geometry. Both recorded randomized trials
also passed. These times exclude startup; precise contact simulation still runs
slower than real time.

```bash
./workshop.sh deploy                         # Fast live view.
./workshop.sh train --num_envs 8              # Eight visible Frankas, fast view.
./workshop.sh train-headless --num_envs 8     # PPO without a 3D viewport.
./workshop.sh deploy --render_quality standard  # Original display settings.
./workshop.sh deploy --playback_mode reference  # Original execution loop.
```

Use `--render_every 1` if you prefer a display frame after every policy step.
The standalone `data` command uses its own camera-rendering settings.


The same optional viewers shown in the recording are included. Open them in
another terminal in this folder:

```bash
python3 lesson/live_dashboard.py
python3 lesson/data_preview.py
python3 lesson/view_files.py
```

Keep the live dashboard open during randomization/training, and the camera
viewer open during `data`. Each viewer is a separate window; use separate
terminals if you want more than one open. They need Tkinter and Pillow.

Your generated files appear in `output/`, and policies in `checkpoints/`.
The recorded 12-update policy is already included locally for a quick repeat
of `test`, `policy` and `deploy`; its training report describes the filmed run.
Running `train` replaces it with your new policy and report. The recorded run
passed 8/8 test trials and 2/2 deployment trials; check your own run's reports.

The gear starts held in the gripper. This lesson learns alignment and insertion,
using a shared state-based policy and contact physics. Synthetic camera data is
a separate exercise. Deployment is to Isaac's simulated Franka.

The front Factory fixture is the validated insertion task; the yellow/white
NIST assembly at the rear is a visual reference. Close one simulation command
before starting the next. Allow about 13 minutes for training on the recording
machine; startup, data generation and evaluation also take time.
