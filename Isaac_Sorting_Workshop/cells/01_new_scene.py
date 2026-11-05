# STEP 01: begin with a new scene and the installed NVIDIA tools.
# start_isaac.py already added this workshop folder to Python's search path.
from workshop import *
new_stage()
assets = nv.Ur10Assets()
env = '/World/PhysicalSorting'
stage = omni.usd.get_context().get_stage()
stage.DefinePrim(env, 'Xform')

print('Conveyor asset:', assets.conveyor_usd)
print('Robot asset:', assets.ur10_usd)
print('STEP_OK=01: empty scene; NVIDIA asset catalog is ready')
