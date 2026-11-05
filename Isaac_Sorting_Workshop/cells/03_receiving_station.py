# STEP 03: reuse the receiving equipment at the tested positions.
nv._add_scene_reference(assets.flip_stack_usd, env + '/pallet_holder',
    translation=(0.9087183897, 0.6628936652, -0.9452558689),
    orientation=(0.7071068, 0.0, 0.0, -0.7071068))
nv._add_scene_reference(assets.pallet_usd, env + '/pallet',
    translation=(0.8826279803, -0.3281049927, -0.8097789819),
    orientation=(0.7071068, 0.0, 0.0, 0.7071068))
nv._add_scene_reference(assets.dolly_usd, env + '/dolly',
    translation=(0.8875089802, -0.3328929926, -1.1810699736))
print('STEP_OK=03: receiving pallet and supports added')
