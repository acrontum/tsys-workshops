# STEP 12: after all six bins settle, the workshop pauses and saves the result.
from validate_report import validate
validate(sim.output / 'report.json')
print('SAVED REPORT: result/report.json inside the workshop folder')
print('SAVED SCENE: result/physical_sorting.usda inside the workshop folder')
print('STEP_OK=12: all physical acceptance checks passed')
