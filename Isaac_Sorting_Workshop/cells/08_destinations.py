# STEP 08: this function chooses the real placement target for each bin.
# lane comes from the bin's stored sorting:class attribute, NOT its image.
def destination_for(lane, slot):
    x = 0.63 if lane == 'A' else 1.05
    y = -0.62 + slot * 0.31
    return np.array([x, y, -0.51874])
sim.destination_for = destination_for
sim.destination_installed = True
print('STEP_OK=08: A and B each have three receiving positions')
