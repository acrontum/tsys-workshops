# Inspect the real state-machine table used by NVIDIA's controller.
# This prints its actual runtime rules; it does not replace the controller.
print('ACTUAL NVIDIA CONTROLLER TRANSITIONS:')
for rule in nv.PalletizerFSM._TRANSITIONS:
    condition = rule.get('conditions') or rule.get('unless') or ''
    print(rule['source'], '--', rule['trigger'], '-->', rule['dest'], condition)
print('In this demo: idle -> picking -> placing -> idle, then done.')
print('STEP_OK=09a: inspected the controller transition rules')
