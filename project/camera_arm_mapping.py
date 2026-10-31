"""Measured local planar camera/joint mapping. Never opens a serial port.

Capture each point with the gripper centered at the same working height and
orientation. This is local interpolation, NOT a full robot IK or pick planner.
Outside measured coverage is rejected. A separate approach/lift is still needed.
"""
import argparse
import itertools
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')


def interpolate(samples, pixel):
    if len(samples) < 4:
        raise ValueError('Need at least four measured points at the same working height')
    point = np.asarray(pixel, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError('Invalid image point')
    options = []
    for triple in itertools.combinations(samples, 3):
        xy = np.array([s['pixel'] for s in triple], dtype=float)
        matrix = np.vstack((xy.T, np.ones(3)))
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 100:
            continue
        weights = np.linalg.solve(matrix, np.append(point, 1))
        if np.min(weights) < -1e-8:
            continue
        raw = np.array([[s['raw'][m] for m in BODY] for s in triple], dtype=float)
        if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 4095):
            raise ValueError('Invalid measured joint position')
        # Do not blend different IK branches or widely separated wrist poses.
        if np.any(np.ptp(raw, axis=0) > 512):
            continue
        options.append((np.max(np.linalg.norm(xy - point, axis=1)), weights @ raw))
    if not options:
        raise ValueError('Outside measured local coverage, degenerate points or incompatible joint poses')
    _, result = min(options, key=lambda item: item[0])
    return {m: int(round(value)) for m, value in zip(BODY, result)}


def capture_sample(pixel, status, health):
    if status.get('phase') not in ('holding','paused','ready_start','released','complete','cycle_complete'):
        raise ValueError('Arm must be stationary in a monitored hold')
    if status.get('error') or time.time() - status.get('last_sample', 0) > 2:
        raise ValueError('Arm telemetry is stale or reports an error')
    if not health.get('live') or health.get('frame_age_seconds', 99) > 1:
        raise ValueError('Camera is not live')
    rows = status['readings']
    if any(rows[m]['Status'] or rows[m]['Moving'] or rows[m]['Torque_Enable'] != 1 for m in BODY):
        raise ValueError('Arm must be held, stationary and without servo alarms')
    if len(pixel) != 2 or not np.isfinite(pixel).all():
        raise ValueError('Invalid pixel point')
    return {'pixel': list(pixel), 'raw': {m:int(rows[m]['Present_Position']) for m in BODY},
            'captured_at': time.time(), 'camera_generation': health['camera_generation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True)
    parser.add_argument('--capture', nargs=2, type=float, metavar=('PIXEL_X','PIXEL_Y'))
    parser.add_argument('--query', nargs=2, type=float)
    args = parser.parse_args()
    data = json.loads(args.file.read_text()) if args.file.exists() else {
        'schema':'local_camera_joint_mapping_v1', 'samples':[], 'motion_validated':False}
    if data.get('schema') != 'local_camera_joint_mapping_v1':
        raise ValueError('Unknown mapping schema')
    if args.capture:
        status = json.load(urllib.request.urlopen('http://127.0.0.1:8768/status', timeout=3))
        health = json.load(urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=3))
        data['samples'].append(capture_sample(args.capture, status, health))
        args.file.parent.mkdir(parents=True, exist_ok=True)
        with args.file.open('w') as output:
            json.dump(data, output, indent=2)
        print('Measured samples:', len(data['samples']))
    if args.query:
        print(json.dumps({'candidate_raw':interpolate(data['samples'], args.query),
                          'motion_validated':False, 'motor_commands_sent':0}))


if __name__ == '__main__': main()
