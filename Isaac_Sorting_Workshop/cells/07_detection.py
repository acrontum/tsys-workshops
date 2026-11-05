# STEP 07: this is the actual detection logic used by the controller.
# It reads perfect simulator positions; it is NOT a camera or proximity sensor.
def detect_bins(ctx):
    if ctx.active_bin is not None:
        return
    for body in ctx.spawner.bins[len(ctx.bins):]:
        ctx.bins.append(nv.BinState(body))
    candidates = []
    for item in ctx.bins:
        x, y, z = item.bin_obj.get_world_poses()[0].numpy()[0]
        if -0.4 < x < 0.4 and 0.0 < y < 0.7:
            candidates.append((float(y), item))
    if candidates:
        ctx.active_bin = min(candidates, key=lambda pair: pair[0])[1]
sim.context.monitors[0] = detect_bins
sim.detector_installed = True
print('STEP_OK=07: simulator-state detection zone connected')
