# Workshop instructions
These are the numbered source files used in the narrated sorting workshop.
The teaching images are already in the video. No PDF is needed.

1. Open a Terminal at the repository root.
2. Type: cd Isaac_Sorting_Workshop
3. Type: ls cells
4. The video selects a recording-machine shortcut, /tmp/IsaacRuntime/python.sh.
   On THIS machine, use the included runtime instead:
     export ISAAC_PYTHON="$(cd .. && pwd)/.runtime/env_isaac/bin/python"
   The launcher also finds that runtime automatically if ISAAC_PYTHON is unset.
   On another machine, use your installed Isaac Python executable.
5. Type: bash open_isaac.sh
   Wait for the empty Isaac window.
6. Type: python3 browse_files.py
   This is the same source viewer shown in the video.
7. In Isaac, choose Window > Script Editor.
8. In the viewer, select one file and click COPY ENTIRE FILE.
   In Script Editor, click the code area, press Ctrl+A, then Ctrl+V, then Run.
   Check STEP_OK before continuing. Keep the same Script Editor session.

Run these cells in this exact order:
  01_new_scene.py
  02_conveyor.py
  03_receiving_station.py
  04_robot.py
  05_environment.py
  06_physics.py          — wait for INITIALIZATION_OK
  07_detection.py
  08_destinations.py
  09_controller.py
  09a_state_machine.py
  09b_safe_placement.py  — run ONCE before starting
  10_controls.py

9. Click START / RESUME in SORTING CONTROLS.
   Let all six containers finish; Isaac saves the result and pauses.
10. Run 11_inspect_sensing.py, then 12_validate.py.
    The validation should print ACCEPTED: six containers, three A and three B.
11. Type: ls result
    Your run creates report.json and physical_sorting.usda.
12. Close Isaac before the RL example. To repeat sorting, launch again and
    repeat the numbered cells from the beginning.

The conveyor, suction attachment, release and settling use contact physics.
Detection reads simulator positions and stored A/B class IDs, not camera images.
The saved USD is a scene; the Python controller is required to run the sorter.
