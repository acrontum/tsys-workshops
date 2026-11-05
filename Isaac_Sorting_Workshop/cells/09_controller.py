# STEP 09: this callback really runs on each completed physics step.
# The NVIDIA PalletizerController contains the wait/pick/place state machine.
# Its UR10 motion backend converts motion goals into joint-drive targets.
def control_step(dt, *_):
    now = SimulationManager.get_simulation_time()
    if sim.controller.state != 'done':
        sim.spawner.step()                 # add one dynamic KLT bin if needed
        observation = sim.context.read(now) # detection, robot/gripper feedback
        command = sim.controller.step(observation) # decide the next action
        sim.context.apply(command, dt)     # belt, suction, robot joint drives
    sim.observe(now)                       # track travel, settling and success
sim.install_control_step(control_step)
print('STEP_OK=09: the physical control loop is installed, still paused')
