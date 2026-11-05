# Disclaimer
The following files are to be used in the context of the workshop held by Acrontum and Blu for T-Systems on the 03.11.2026. 
The source files are not prepared for use in a productive environment and will have correspondingly dependencies and coding patterns that would otherwise not be found.

# Prerequsites
1. Installed Isaac SIM https://docs.isaacsim.omniverse.nvidia.com/6.1.0/installation/install_python.html
2. Access to the workshop video files: workshop_video_1.mp4, workshop_video_2.1.mp4, workshop_video_2.2.mp4

# Two Isaac workshops — follow the narrated videos

| Example | Step-by-step files |
|---|---|
| 1. Physically sort six boxes with a UR10 |  [Start here](Isaac_Sorting_Workshop/README.txt) |
| 2. Eight Frankas learn randomized gear insertion with PPO | [Start here](example2_parallel_rl/README.md) |

Pause the video and repeat each action. Both folders use the same filenames,
numbered steps and command sequence as their recordings.

From a terminal in this repository:

```bash
# Example 1: open an empty Isaac scene and the source-file viewer.
cd Isaac_Sorting_Workshop
bash open_isaac.sh
python3 browse_files.py
```

Close Isaac before starting Example 2. In a new terminal at the repository root:

```bash
cd example2_parallel_rl
./workshop.sh help
```

The runtime is already installed on this machine. On a new compatible NVIDIA
Linux machine, review NVIDIA's license and install once from the repository root:

```bash
./install.sh --accept-eula --install-system-packages
```

The examples use Isaac Sim 6.2 / Isaac Lab 3.0.0. The recordings use Linux.
Internet access is needed for referenced NVIDIA assets. The source viewer needs
Tkinter; the RL camera viewer also needs Pillow.

Videos, downloaded assets, checkpoints, generated results and `.runtime/` are
ignored by Git. A source-only clone follows the same steps; RL `setup` downloads
the starting policy and `train` creates the trained policy. Runtime checks,
tests and license notices are kept in `.support/`.
