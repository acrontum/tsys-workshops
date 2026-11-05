# STEP 09b: release 10 cm above the resting height to avoid pallet contact.
# Keep NVIDIA's installed controller. PhysX settles each released bin.
# Run this file ONCE, before clicking START / RESUME.
resting_destination = destination_for
def destination_with_clearance(lane, slot):
    target = resting_destination(lane, slot).copy()
    target[2] += 0.10
    return target
sim.destination_for = destination_with_clearance
# The arm's release goal and the bin's resting centre have different heights.
# Confirm the physical bin on the pallet, rather than at the raised tool goal.
def bin_at_pallet_place():
    current = sim.context.active_bin
    target = sim.context.monitor_target
    if current is None or target is None:
        return False
    position = current.bin_obj.get_world_poses()[0].numpy()[0]
    return bool(np.linalg.norm(position[:2] - target[:2]) < 0.12
                and abs(float(position[2]) - (-0.59374)) < 0.03)
sim.context.active_bin_at_stack_target = bin_at_pallet_place
print('Release target is 10 centimetres above the original resting target.')
print('The installed NVIDIA controller still drives the physical arm.')
print('Gravity settles each bin after the physical gripper releases it.')
print('Completion checks the measured bin centre on the pallet.')
print('STEP_OK=09b: release clearance configured; controller retained')
