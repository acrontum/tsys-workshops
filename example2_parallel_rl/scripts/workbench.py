"""NVIDIA NIST taskboard on the validated Factory bench.

Assembly contact geometry and robot coordinates remain the official Factory task.
The NIST board sits to the left as a visible companion station.
"""
def add_nist_bench(cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
    cfg.scene.ground.spawn = sim_utils.GroundPlaneCfg(color=(0.16,0.20,0.24),size=(20.,20.))
    cfg.scene.light.spawn.intensity=1000.
    cfg.scene.table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/Mounts/UWPatVention/pat_vention.usd",
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.4,0.,-0.868),rot=(0.,0.,-0.70710678,0.70710678)
        ),
    )
    # Clear yellow/white contrast for the audience. Only visual materials change.
    yellow=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0,0.65,0.025),roughness=0.38)
    white=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.91,0.94,0.97),roughness=0.45)
    cfg.task.held_asset.spawn.visual_material=yellow
    cfg.task.small_gear_cfg.spawn.visual_material=yellow
    cfg.task.large_gear_cfg.spawn.visual_material=yellow
    cfg.task.fixed_asset.spawn.visual_material=white
    cfg.scene.nist_board = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/NIST_Taskboard",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/NIST/Taskboard/nistboard.usd",
            scale=(1.0,1.0,0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.35,0.08,0.015),rot=(1.,0.,0.,0.)
        ),
    )
    # Genuine NIST reference assembly on the rear board; visual inventory only.
    # The front Factory fixture supplies the proven RL collision task.
    from pathlib import Path
    asset_dir=Path(__file__).resolve().parents[1]/"assets"
    for name,filename,material in [
        ("NIST_ReferenceBase","gear_base.usd",white),
        ("NIST_ReferenceMedium","gear_medium.usd",yellow),
        ("NIST_ReferenceSmall","gear_small.usd",yellow),
        ("NIST_ReferenceLarge","gear_large.usd",yellow),
    ]:
        setattr(cfg.scene,name.lower(),AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/"+name,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(asset_dir/filename),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=False),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=material,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.52,0.27,0.015),rot=(0.,0.,0.,1.)
            ),
        ))
