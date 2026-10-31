"""Follower-only opening/return test. No body-joint writes; opt-in motion."""
import argparse
import json
import time

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from test_so100 import stop_motor


def probe(robot, move=False, ticks=80, torque_limit=None):
    bus, motor = robot.bus, 'gripper'
    def read(register, name=motor):
        return int(bus.read(register, name, normalize=False, num_retry=2))
    def write(register, value):
        bus.write(register, motor, value, normalize=False, num_retry=2)
        if read(register) != value:
            raise RuntimeError('Readback mismatch: ' + register)
    origin = read('Present_Position')
    low, high = read('Min_Position_Limit'), read('Max_Position_Limit')
    result = {'origin': origin, 'limits': [low, high], 'motion_requested': move}
    if not move:
        return result
    if not 1 <= ticks <= 80:
        raise ValueError('Opening probe must be 1..80 raw ticks')
    if not robot.is_calibrated:
        raise RuntimeError('Calibration mismatch')
    if any(read('Torque_Enable', m) for m in bus.motors):
        raise RuntimeError('Another motor is enabled; probe did not start')
    # Allow starting at the recorded closed end; the first leg opens away
    # from that end and the return never commands below the starting encoder.
    if not low <= origin <= origin + ticks <= high - 3:
        raise RuntimeError('Probe would leave calibrated range')
    if read('Operating_Mode') != 0 or read('Goal_Time') != 0:
        raise RuntimeError('Unexpected mode or Goal_Time')
    vlow, vhigh = read('Min_Voltage_Limit'), read('Max_Voltage_Limit')
    health_samples = []
    def health():
        snapshot = {k: read(k) for k in ('Status', 'Present_Temperature', 'Present_Voltage', 'Present_Current', 'Present_Load')}
        health_samples.append(snapshot)
        if snapshot['Status'] or snapshot['Present_Temperature'] >= 55:
            raise RuntimeError('Gripper status/temperature fault: ' + json.dumps(snapshot))
        if not vlow < snapshot['Present_Voltage'] < vhigh:
            raise RuntimeError('Voltage outside configured range: ' + json.dumps(snapshot))
    health()
    old_speed = read('Goal_Velocity')
    old_torque = read('Torque_Limit')
    if torque_limit is not None:
        if (type(torque_limit) is not int or not old_torque <= torque_limit <= min(600, old_torque + 100)
                or read('Lock') != 1):
            raise ValueError('Temporary torque limit must be at most +100, capped at 600; Lock must stay 1')
        result['torque_limit_before'] = old_torque
    samples = []
    try:
        if torque_limit is not None:
            write('Torque_Limit', torque_limit)
            result['torque_limit_tested'] = read('Torque_Limit')
        write('Goal_Velocity', 2)
        write('Goal_Position', origin)
        if abs(read('Present_Position') - origin) > 5:
            raise RuntimeError('Gripper moved during preparation')
        write('Torque_Enable', 1)
        command = origin
        for label, target in [('open', origin + ticks), ('return', origin)]:
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                health()
                if read('Torque_Enable') != 1:
                    raise RuntimeError('Gripper torque unexpectedly off')
                actual = read('Present_Position')
                if not low <= actual <= high or abs(actual - command) > 35:
                    raise RuntimeError('Position outside range or following error')
                if command == target and abs(actual - target) <= 12:
                    samples.append({'stage': label, 'target': target, 'actual': actual})
                    break
                if abs(actual - command) <= 12:
                    command += max(-4, min(4, target - command))
                    write('Goal_Position', command)
                time.sleep(.05)
            else:
                raise RuntimeError(f'Gripper timeout at {label}: target={target}, actual={actual}')
        result.update(samples=samples, passed=True,
                      max_temperature=max(s['Present_Temperature'] for s in health_samples),
                      peak_current_raw=max(s['Present_Current'] for s in health_samples),
                      object_pickup_verified=False)
    finally:
        result['torque_off_verified'] = stop_motor(robot, motor)
        if not result['torque_off_verified']:
            raise RuntimeError('Torque-off unconfirmed; disconnect motor power')
        write('Goal_Velocity', old_speed)
        if torque_limit is not None:
            write('Torque_Limit', old_torque)
            result['torque_limit_restored'] = read('Torque_Limit')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--torque-limit', type=int)
    args = parser.parse_args()
    robot = SO100Follower(SO100FollowerConfig(
        port='/dev/cu.usbmodem5B421348831', id='pawacha_so100_follower'))
    try:
        robot.bus.connect()
        print(json.dumps(probe(robot, args.move, torque_limit=args.torque_limit), ensure_ascii=False), flush=True)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
