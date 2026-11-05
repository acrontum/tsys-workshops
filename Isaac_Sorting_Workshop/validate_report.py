#!/usr/bin/env python3
"""Independently check physical acceptance evidence saved by a completed run."""
import argparse
import itertools
import json
import math
from pathlib import Path


def validate(path):
    report = json.loads(path.read_text())
    assert report['success'], report.get('error', 'Simulation did not report success')
    assert report['physics_engine'] == 'physx'
    assert report['test_gripper'] is False
    assert report['animation_time_samples'] is False
    total = report['requested_bins']
    assert report['sorted_bins'] == total
    assert report['lanes'] == {'A': total//2, 'B': total//2}
    placements = report['placements']
    assert len({p['path'] for p in placements}) == total
    for p in placements:
        assert p['grasp_confirmed'], p['path']
        x, y, z = p['settled_position']
        # Independent receiving-lane footprint and actual pallet-top height.
        assert (.49 < x < .77) if p['lane'] == 'A' else (.91 < x < 1.19), p
        assert -.76 < y < .14, p
        assert abs(z - (-.59374)) < .03, p
        assert p['settled_linear_speed'] < .02, p
        travel = report['conveyor_travel'][p['path']]
        assert travel['initial_position'][1] - travel['arrival_y'] > .7, travel
    for a, b in itertools.combinations(placements, 2):
        assert math.dist(a['settled_position'], b['settled_position']) > .19
    # Inspect the root layer without fetching/composing external asset references.
    from pxr import Sdf
    layer = Sdf.Layer.FindOrOpen(str(path.parent/'physical_sorting.usda'))
    assert layer is not None
    time_samples = []
    def inspect(spec_path):
        spec = layer.GetObjectAtPath(spec_path)
        if isinstance(spec, Sdf.AttributeSpec) and layer.ListTimeSamplesForPath(spec_path):
            time_samples.append(str(spec_path))
    layer.Traverse('/', inspect)
    assert not time_samples, time_samples
    print(f'ACCEPTED: {total} physically transported, gripped, sorted and settled bins; no authored animation samples.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    validate(parser.parse_args().report.resolve())
