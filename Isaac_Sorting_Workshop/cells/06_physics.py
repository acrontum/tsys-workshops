# STEP 06: initialize physics, spawning and the robot motion backend.
# This asynchronous setup keeps Isaac responsive while assets load.
sim = Workshop(robot, assets, count=6)
async def initialize_workcell():
    await sim.initialize()
    if sim.error:
        raise RuntimeError(sim.error)
    print('STEP_OK=06: INITIALIZATION_OK; controller paused')
asyncio.ensure_future(initialize_workcell())
# Wait for INITIALIZATION_OK before continuing. No sorting callback exists yet.
print('Physics initialization requested; please wait...')
