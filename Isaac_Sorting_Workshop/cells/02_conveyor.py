# STEP 02: reference a complete conveyor asset, including its contact geometry.
stage.DefinePrim(env + '/conveyor', 'Xform')
with Sdf.ChangeBlock():
    nv._add_scene_reference(
        assets.conveyor_usd, env + '/conveyor',
        translation=(-0.0544, 2.409, -1.4),
        orientation=(0.7071068, 0.0, 0.0, -0.7071068))
    nv._author_conveyor_kinematic_override(stage, env + '/conveyor/Rollers')
view_station()
print('STEP_OK=02: conveyor added; physics is not running yet')
