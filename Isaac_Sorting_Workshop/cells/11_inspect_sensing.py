# STEP 11: after all six finish and physics pauses, inspect actual feedback.
body = sim.spawner.bins[0]
path = str(body.paths[0])
x, y, z = body.get_world_poses()[0].numpy()[0]
lane = stage.GetPrimAtPath(path).GetAttribute('sorting:class').Get()
print('CLASS FROM STORED ATTRIBUTE:', lane)
print('POSITION FROM PHYSICS:', [round(float(v), 3) for v in (x,y,z)])
print('INSIDE DETECTION ZONE:', -0.4 < x < 0.4 and 0.0 < y < 0.7)
print('CAMERA USED BY SORTER: none')
gripper = robot.suction_gripper
print('GRIPPER ATTACHMENTS:', gripper._iface.get_gripped_objects(gripper.surface_gripper_path))
print('RECORDED ATTACHMENTS:', len(sim.context.grasps))
print('COMPLETED GROUP COUNTS:', sim.context.counts)
print('ROBOT JOINT POSITIONS:', robot.articulation.get_dof_positions().numpy().round(3))
# Bring the operating buttons above the viewport after inspecting objects.
sim.windows[-1].focus()
print('STEP_OK=11: completed feedback inspected; run step 12 to validate')
