# STEP 04: add the real UR10 asset and its short suction-tool variant.
nv._add_scene_reference(assets.ur10_mount_usd, env + '/ur10_mount',
    translation=(0.0, 0.0, 0.0))
nv._add_scene_reference(assets.sortbot_housing_usd, env + '/sortbot_housing',
    translation=(-0.018, -0.282, -1.1749999737))
nv._add_scene_reference(assets.ur10_usd, env + '/ur10',
    translation=(0.0, 0.0, 0.0), variants=assets.ur10_variants)
robot = nv.Ur10(prim_path=env + '/ur10', use_test_gripper=False)
print('STEP_OK=04: articulated robot; real SurfaceGripper selected')
