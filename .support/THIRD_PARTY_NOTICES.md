# Third-party notices

This is an independent workshop demo and is not an NVIDIA-authored or NVIDIA-supported training product.

The installer downloads NVIDIA Isaac Sim and Isaac Lab packages from official distribution locations. NVIDIA software, example tasks, robot models, USD assets, product names, and marks remain subject to NVIDIA's applicable licenses and trademark terms.

The physical sorter reuses NVIDIA's installed UR10 palletizing example, UR10 robot,
SurfaceGripper, conveyor, KLT containers and pallet. Its motion controller remains
NVIDIA's installed implementation.

The Franka RL demo uses the installed Isaac Lab Factory GearMesh task and a
published NVIDIA starting policy. Setup downloads and verifies the NIST reference
gear USDs and policy into ignored `example2_parallel_rl/assets/` and `checkpoints/`.
Generated scenes and sensor data remain in ignored output directories. NVIDIA
assets and policy weights are not included in the Git repository.

The supplied M1–M12 presentation, local presenter PDFs, recordings and narrated
backups are kept in ignored local directories. They are not repository assets.

Other dependencies—including PyTorch, RSL-RL, Gymnasium, Pillow, and uv—retain their respective upstream licenses.

