"""Hold, bounded manual jogging, and explicitly selected experimental routes."""

import json
import queue
import subprocess
import sys
from pathlib import Path
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from test_so100 import stop_motor
from bounded_trim import BoundedTrim
from route_stepper import RouteStepper, make_transfer_route, make_source_route, make_return_route

PORT = 8766
BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ENABLE_ORDER = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "shoulder_pan")
stop = threading.Event()
lock = threading.Lock()
state = {"phase": "idle", "error": None, "torque_off_verified": None}
operation_lock = threading.Lock()
worker = None
manual_commands = queue.Queue(maxsize=8)


def jog_target(target, motor, ticks, limits):
    maximum = 80 if motor == 'gripper' else 22
    if motor not in (*BODY, 'gripper') or type(ticks) is not int or not 0 < abs(ticks) <= maximum:
        raise ValueError('Invalid bounded jog')
    updated = dict(target)
    updated[motor] += ticks
    low, high = limits[motor]
    margin = 0 if motor == 'gripper' else 3
    if not low + margin <= updated[motor] <= high - margin:
        raise ValueError('Jog outside joint range: ' + motor)
    return updated


def update(**values):
    with lock:
        state.update(values)


def probe_result(origin, target, outbound, returned, tolerance=12):
    return {
        'outbound_error_ticks': abs(outbound - target),
        'return_error_ticks': abs(returned - origin),
        'passed': abs(outbound - target) <= tolerance and abs(returned - origin) <= tolerance,
    }


def route_entry_limit(path):
    if path in ('/transfer24', '/source24'):
        # Explicitly authorized recovery from the observed ~37-degree elbow
        # offset. Other joints retain the small entry limit.
        return {m: (450 if m == 'elbow_flex' else 90) for m in BODY}
    if path == '/approach':
        return 180
    return 90 if path in ('/full_dryrun', '/transfer') else 35


def guide():
    """Single-owner, read-only feedback for manually matching the start pose."""
    robot = SO100Follower(SO100FollowerConfig(
        port='/dev/cu.usbmodem5B421348831', id='pawacha_so100_follower'))
    bus = robot.bus
    try:
        target = json.loads(Path(__file__).with_name('pick_place_poses.json').read_text())['poses']['bottom_left']['retreat']['raw_position']
        bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError('Calibration mismatch')
        while not stop.is_set():
            torque = {m: bus.read('Torque_Enable', m, normalize=False, num_retry=2) for m in bus.motors}
            if any(torque.values()):
                raise RuntimeError('Read-only guide requires torque OFF on every motor')
            actual = {m: bus.read('Present_Position', m, normalize=False, num_retry=2) for m in BODY}
            update(phase='guiding', guide={
                'difference_degrees': {m: round((actual[m] - target[m]) * 360 / 4095, 1) for m in BODY},
                'ready': all(abs(actual[m] - target[m]) <= 25 for m in BODY),
            })
            stop.wait(.3)
    except Exception as exc:
        update(error=str(exc))
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)
        with lock:
            state['phase'] = 'fault' if state.get('error') else 'stopped'


def hold(route=None, probe=False, gain_probe=False, baseline_only=False, trim_probe=False, entry_limit=35, route_p=None, manual=False):
    robot = SO100Follower(SO100FollowerConfig(
        port="/dev/cu.usbmodem5B421348831", id="pawacha_so100_follower",
        use_degrees=True, max_relative_target=2.0,
    ))
    bus = robot.bus
    armed = False
    speeds = {}
    original_p = None
    gain_changed = False
    gripper_armed = False
    grip_command = grip_target = grip_started = None

    def read(register, motor):
        return bus.read(register, motor, normalize=False, num_retry=2)

    def write_verified(register, motor, value):
        try:
            bus.write(register, motor, value, normalize=False, num_retry=2)
        except ConnectionError:
            # A lost acknowledgement does not prove that an idempotent SRAM
            # write failed. Accept it only with a matching readback and no alarm.
            if read(register, motor) != value or read("Status", motor) != 0:
                raise
            print("Write acknowledgement missing; register verified:", motor, register, flush=True)
        if read(register, motor) != value:
            raise RuntimeError(motor + ": " + register + " readback mismatch.")

    try:
        bus.connect()
        if not robot.is_calibrated:
            raise RuntimeError("Calibration mismatch; hold cancelled.")
        if any(read("Torque_Enable", motor) != 0 for motor in bus.motors):
            raise RuntimeError("All motors must initially have torque off.")
        positions = {motor: read("Present_Position", motor) for motor in BODY}
        limits, baseline, minimum_voltage, maximum_voltage = {}, {}, {}, {}
        for motor in BODY:
            limits[motor] = (read("Min_Position_Limit", motor), read("Max_Position_Limit", motor))
            if not limits[motor][0] + 3 <= positions[motor] <= limits[motor][1] - 3:
                raise RuntimeError(f"{motor}: current raw={positions[motor]}, permitted start "
                                   f"{limits[motor][0] + 3}..{limits[motor][1] - 3}; "
                                   "support and move away from the end of travel with torque off.")
            if read("Operating_Mode", motor) != 0 or read("Status", motor) != 0:
                raise RuntimeError(motor + ": unexpected mode or alarm.")
            if read("Present_Temperature", motor) >= 55 or read("Goal_Time", motor) != 0:
                raise RuntimeError(motor + ": temperature/timing check failed.")
            baseline[motor] = read("Present_Voltage", motor)
            minimum_voltage[motor] = read("Min_Voltage_Limit", motor)
            maximum_voltage[motor] = read("Max_Voltage_Limit", motor)
            if not minimum_voltage[motor] < baseline[motor] < maximum_voltage[motor]:
                raise RuntimeError(motor + ": supply voltage is outside configured limits.")
            speeds[motor] = read("Goal_Velocity", motor)
        if any(abs(read("Present_Position", motor) - positions[motor]) > 4 for motor in BODY):
            raise RuntimeError("Arm moved during preflight; support it steadily before retrying.")
        if route:
            if any(abs(positions[m] - route[0][1][m]) >
                   (entry_limit[m] if isinstance(entry_limit, dict) else entry_limit) for m in BODY):
                raise RuntimeError("Dry run must start at the taught outside clearance pose.")
            for label, target in route:
                if any(not limits[m][0] + 3 <= target[m] <= limits[m][1] - 3 for m in BODY):
                    raise RuntimeError("Route target outside limits: " + label)
        probe_origin = positions['elbow_flex']
        # Larger diagnostic, explicitly requested: +45 ticks (~3.96 degrees).
        probe_target = probe_origin + 45
        if probe_target > limits['elbow_flex'][1] - 3:
            probe_target = probe_origin - 45
        if probe and not limits['elbow_flex'][0] + 3 <= probe_target <= limits['elbow_flex'][1] - 3:
            raise RuntimeError('Elbow probe target outside limits')
        if gain_probe or route_p is not None:
            original_p = read('P_Coefficient', 'elbow_flex')
            if original_p != 16 or read('Lock', 'elbow_flex') != 1:
                raise RuntimeError('Gain comparison requires P=16 and non-persistent Lock=1.')
            if route_p is not None and (route_p != 24 or not route):
                raise RuntimeError('Only the bounded experimental P=24 route profile is allowed.')
        if stop.is_set():
            return
        armed = True
        update(phase="enabling")
        # A goal write can itself engage some firmware: cover every write with
        # cleanup, seed only the in-range present position, and verify readback.
        for motor in BODY:
            write_verified("Goal_Velocity", motor, 2)
            if read("Goal_Velocity", motor) != 2:
                raise RuntimeError(motor + ": low-speed readback failed.")
            write_verified("Goal_Position", motor, positions[motor])
            if read("Goal_Position", motor) != positions[motor]:
                raise RuntimeError(motor + ": seeded goal readback failed.")
        if any(abs(read("Present_Position", motor) - positions[motor]) > 5 for motor in BODY):
            raise RuntimeError("Posture drifted while goals were being prepared.")
        if route_p is not None:
            if read('Lock', 'elbow_flex') != 1:
                raise RuntimeError('EEPROM lock changed; route cancelled.')
            gain_changed = True
            write_verified('P_Coefficient', 'elbow_flex', route_p)
            update(route_gain_active=route_p, gain_restored=False)
        for motor in ENABLE_ORDER:
            if stop.is_set():
                return
            write_verified("Torque_Enable", motor, 1)
            if read("Torque_Enable", motor) != 1:
                raise RuntimeError(motor + ": torque enable was not confirmed.")
        started = time.monotonic()
        route_index = 0
        route_started = started
        last_route_progress = started
        probe_phase = 0
        trimmer = None
        trim_started = None
        trim_stable = 0
        route_stepper = RouteStepper(positions, limits) if route else None
        manual_stepper = RouteStepper(positions, limits) if manual else None
        manual_target = dict(positions)
        manual_origin = dict(positions)
        route_stable = 0
        update(phase="holding", target_raw=dict(positions), gripper_enabled=False, manual_mode=manual)
        print("Holding the five body joints at their current posture; gripper torque remains off.", flush=True)
        while not stop.is_set():
            readings = {}
            for motor in BODY:
                readings[motor] = {register: read(register, motor) for register in (
                    "Present_Position", "Present_Voltage", "Present_Temperature",
                    "Torque_Enable", "Status",
                )}
                row = readings[motor]
                if row["Status"] != 0 or row["Torque_Enable"] != 1:
                    update(last_fault={"motor": motor, "reading": dict(row), "time": time.time()})
                    raise RuntimeError(motor + ": alarm or torque shutdown.")
                low_voltage = minimum_voltage[motor]
                high_voltage = maximum_voltage[motor]
                if row["Present_Voltage"] <= low_voltage or row["Present_Voltage"] >= high_voltage:
                    first = row["Present_Voltage"]
                    # Confirm an isolated anomalous register value immediately.
                    # A confirmed fault still shuts down; no firmware protection
                    # thresholds are changed or disabled.
                    repeated = read("Present_Voltage", motor)
                    event = {"motor": motor, "first_raw": first, "repeated_raw": repeated}
                    update(last_voltage_recheck=event)
                    print("Voltage recheck:", json.dumps(event), flush=True)
                    if repeated <= low_voltage or repeated >= high_voltage:
                        update(last_fault={"motor": motor, "reading": dict(row), "confirmed_voltage_raw": repeated})
                        raise RuntimeError(f"{motor}: voltage fault confirmed ({first / 10:.1f} -> {repeated / 10:.1f} V).")
                    if read("Status", motor) != 0 or read("Torque_Enable", motor) != 1:
                        raise RuntimeError(motor + ": servo alarm/shutdown during voltage recheck.")
                    row["Present_Voltage"] = repeated
                if row["Present_Temperature"] >= 55:
                    raise RuntimeError(motor + ": temperature limit reached.")
                tracking_limit = 60 if probe and motor == 'elbow_flex' else 35
                if probe and motor == 'elbow_flex' and abs(row['Present_Position'] - probe_origin) > 60:
                    raise RuntimeError('Elbow exceeded bounded probe travel.')
                if abs(row["Present_Position"] - positions[motor]) > tracking_limit:
                    raise RuntimeError(motor + ": arm is not maintaining the held posture.")
                if not limits[motor][0] - 2 <= row["Present_Position"] <= limits[motor][1] + 2:
                    raise RuntimeError(motor + ": encoder position passed a calibrated limit.")
            if gripper_armed:
                grip_row = {k: read(k, 'gripper') for k in (
                    'Present_Position', 'Present_Voltage', 'Present_Temperature', 'Torque_Enable', 'Status')}
                readings['gripper'] = grip_row
                if (grip_row['Status'] or grip_row['Torque_Enable'] != 1
                        or grip_row['Present_Temperature'] >= 55
                        or not minimum_voltage['gripper'] < grip_row['Present_Voltage'] < maximum_voltage['gripper']
                        or not limits['gripper'][0] <= grip_row['Present_Position'] <= limits['gripper'][1]
                        or abs(grip_row['Present_Position'] - grip_command) > 35):
                    update(last_fault={'motor': 'gripper', 'reading': dict(grip_row)})
                    raise RuntimeError('Gripper alarm, temperature, voltage or tracking fault')
            elif read("Torque_Enable", "gripper") != 0:
                raise RuntimeError("Gripper torque unexpectedly enabled.")
            update(
                readings=readings, target_raw=dict(positions), seconds=round(time.monotonic() - started, 1),
                max_error_degrees=round(max(abs(readings[m]["Present_Position"] - positions[m]) for m in BODY) * 360 / 4095, 2),
                voltage_min=round(min(row["Present_Voltage"] for row in readings.values()) / 10, 1),
                temperature_max=max(row["Present_Temperature"] for row in readings.values()),
            )
            if stop.is_set():
                break
            if manual:
                try:
                    request = manual_commands.get_nowait()
                except queue.Empty:
                    request = None
                if request:
                    motor, ticks = request['motor'], request['ticks']
                    if motor == 'gripper':
                        if not gripper_armed:
                            actual_grip = read('Present_Position', 'gripper')
                            limits['gripper'] = (read('Min_Position_Limit', 'gripper'), read('Max_Position_Limit', 'gripper'))
                            # Validate the requested step before any gripper write.
                            try:
                                proposed = jog_target({'gripper': actual_grip}, motor, ticks, limits)['gripper']
                            except ValueError as exc:
                                update(manual_error=str(exc))
                                continue
                            if not limits['gripper'][0] <= actual_grip <= limits['gripper'][1]:
                                raise RuntimeError('Gripper encoder outside calibrated range')
                            if (read('Operating_Mode', motor) != 0 or read('Goal_Time', motor) != 0
                                    or read('Status', motor) or read('Present_Temperature', motor) >= 55):
                                raise RuntimeError('Gripper not ready')
                            minimum_voltage[motor] = read('Min_Voltage_Limit', motor)
                            maximum_voltage[motor] = read('Max_Voltage_Limit', motor)
                            if not minimum_voltage[motor] < read('Present_Voltage', motor) < maximum_voltage[motor]:
                                raise RuntimeError('Gripper voltage outside configured range')
                            speeds[motor] = read('Goal_Velocity', motor)
                            gripper_armed = True  # Cleanup also covers partial initialization.
                            write_verified('Goal_Velocity', motor, 2)
                            write_verified('Goal_Position', motor, actual_grip)
                            if abs(read('Present_Position', motor) - actual_grip) > 5:
                                raise RuntimeError('Gripper drifted during preparation')
                            write_verified('Torque_Enable', motor, 1)
                            grip_command, grip_target = actual_grip, proposed
                            update(gripper_enabled=True)
                        else:
                            try:
                                grip_target = jog_target({'gripper': grip_target}, motor, ticks, limits)['gripper']
                            except ValueError as exc:
                                update(manual_error=str(exc))
                        grip_started = time.monotonic()
                    else:
                        try:
                            proposed = jog_target(manual_target, motor, ticks, limits)
                            if any(abs(proposed[m] - manual_origin[m]) > 180 for m in BODY):
                                raise ValueError('Manual session travel exceeded 180 ticks')
                            manual_target = proposed
                            update(manual_error=None)
                        except ValueError as exc:
                            update(manual_error=str(exc))
                actual = {m: readings[m]['Present_Position'] for m in BODY}
                try:
                    next_goals, body_reached = manual_stepper.step(manual_target, actual)
                    for motor in BODY:
                        if stop.is_set(): break
                        if next_goals[motor] != positions[motor]:
                            write_verified('Goal_Position', motor, next_goals[motor])
                            positions[motor] = next_goals[motor]
                except ValueError as exc:
                    # A rejected jog freezes commands, not torque support.
                    manual_target = dict(manual_stepper.nominal)
                    body_reached = False
                    update(manual_error=str(exc))
                grip_reached = not gripper_armed
                if gripper_armed:
                    grip_actual = read('Present_Position', 'gripper')
                    grip_reached = grip_command == grip_target and abs(grip_actual - grip_target) <= 12
                    if not grip_reached and time.monotonic() - grip_started > 12:
                        # Relax a stalled jaw command to the measured position.
                        # This is not proof that an object was grasped.
                        grip_command = grip_target = grip_actual
                        write_verified('Goal_Position', 'gripper', grip_actual)
                        update(manual_error='Gripper timeout; holding measured opening, pickup not verified')
                    elif not grip_reached and abs(grip_actual - grip_command) <= 12:
                        grip_command += max(-4, min(4, grip_target - grip_command))
                        write_verified('Goal_Position', 'gripper', grip_command)
                update(manual_target_raw=dict(manual_target), gripper_target_raw=grip_target,
                       manual_move_complete=body_reached and grip_reached)
            if probe:
                sample = {k: read(k, 'elbow_flex') for k in (
                    'Goal_Position', 'Present_Position', 'Goal_Velocity',
                    'Present_Velocity', 'Present_Load', 'Moving', 'Status', 'P_Coefficient')}
                sample['elapsed'] = round(time.monotonic() - started, 2)
                print('Elbow probe:', json.dumps(sample), flush=True)
                update(probe_sample=sample)
                if probe_phase == 0:
                    write_verified('Goal_Position', 'elbow_flex', probe_target)
                    positions['elbow_flex'] = probe_target
                    probe_phase = 1
                    update(route_stage='probe: elbow +3.96 degrees; no gripper')
                elif probe_phase == 1 and time.monotonic() - started >= 6:
                    write_verified('Goal_Position', 'elbow_flex', probe_origin)
                    positions['elbow_flex'] = probe_origin
                    probe_phase = 2
                    update(probe_outbound=sample, route_stage='probe: returning elbow')
                elif probe_phase == 2 and time.monotonic() - started >= 12:
                    result = probe_result(probe_origin, probe_target,
                                          state['probe_outbound']['Present_Position'],
                                          sample['Present_Position'])
                    if trim_probe:
                        trimmer = BoundedTrim(probe_origin, limits['elbow_flex'][0] + 3,
                                              limits['elbow_flex'][1] - 3)
                        trim_started = time.monotonic()
                        probe_phase = 5
                        update(probe_return=sample, probe_result=result,
                               route_stage='bounded trim: at most 32 ticks; P/I unchanged')
                        continue
                    if baseline_only:
                        update(probe_return=sample, probe_result=result,
                               route_stage='baseline test finished; holding posture')
                        probe = False
                        continue
                    update(probe_return=sample, probe_result=result,
                           speed_comparison={'baseline': {'return': sample, 'result': result}},
                           route_stage=('comparison: P=32 outbound' if gain_probe else 'comparison: speed 4 outbound'))
                    if gain_probe:
                        if read('Lock', 'elbow_flex') != 1:
                            raise RuntimeError('EEPROM lock changed; gain test cancelled.')
                        gain_changed = True
                        write_verified('P_Coefficient', 'elbow_flex', 32)
                    else:
                        write_verified('Goal_Velocity', 'elbow_flex', 4)
                    write_verified('Goal_Position', 'elbow_flex', probe_target)
                    positions['elbow_flex'] = probe_target
                    probe_phase = 3
                elif probe_phase == 3 and time.monotonic() - started >= 18:
                    update(probe_fast_outbound=sample, route_stage='comparison: returning')
                    write_verified('Goal_Position', 'elbow_flex', probe_origin)
                    positions['elbow_flex'] = probe_origin
                    probe_phase = 4
                elif probe_phase == 4 and time.monotonic() - started >= 24:
                    result = probe_result(probe_origin, probe_target,
                                          state['probe_fast_outbound']['Present_Position'],
                                          sample['Present_Position'])
                    with lock:
                        comparison = dict(state['speed_comparison'])
                    comparison['P_32' if gain_probe else 'speed_4'] = {'return': sample, 'result': result}
                    if gain_probe:
                        write_verified('P_Coefficient', 'elbow_flex', original_p)
                        gain_changed = False
                        update(gain_restored=True)
                    else:
                        write_verified('Goal_Velocity', 'elbow_flex', 2)
                    probe = False
                    update(probe_fast_return=sample, speed_comparison=comparison,
                           route_stage='comparison finished; original settings restored; holding posture')
                elif probe_phase == 5:
                    command, reached = trimmer.step(sample['Present_Position'])
                    trim_stable = trim_stable + 1 if reached else 0
                    if command != positions['elbow_flex']:
                        write_verified('Goal_Position', 'elbow_flex', command)
                        positions['elbow_flex'] = command
                    update(trim_result={'nominal': probe_origin, 'actual': sample['Present_Position'],
                                        'command': command, 'trim_ticks': trimmer.trim,
                                        'passed': False, 'finished': False})
                    if trim_stable >= 3 or time.monotonic() - trim_started >= 12:
                        probe = False
                        update(trim_result={'nominal': probe_origin, 'actual': sample['Present_Position'],
                                            'command': command, 'trim_ticks': trimmer.trim,
                                            'passed': trim_stable >= 3, 'finished': True},
                               route_stage='bounded trim finished; holding last command; support before stop')
            if route and route_index < len(route):
                if (time.monotonic() - started > 600 or time.monotonic() - route_started > 180
                        or time.monotonic() - last_route_progress > 5):
                    # A stalled planner is not a servo fault. Freeze at the
                    # last commanded posture rather than dropping support.
                    update(route_stage="paused: no tracking progress; support before stop", route_complete=False)
                    route = None
                    continue
                label, target = route[route_index]
                update(route_stage=label, route_complete=False)
                actual = {m: readings[m]['Present_Position'] for m in BODY}
                try:
                    commands, reached = route_stepper.step(target, actual)
                except ValueError as exc:
                    # Planner rejection pauses commands without releasing the arm.
                    update(route_stage='paused: ' + str(exc), route_complete=False)
                    route = None
                    continue
                update(nominal_raw=dict(route_stepper.nominal),
                       route_trim_ticks=route_stepper.elbow.trim,
                       route_trim_raw={m: t.trim for m, t in route_stepper.trimmers.items()})
                for motor in BODY:
                    if stop.is_set():
                        break
                    if commands[motor] != positions[motor]:
                        write_verified('Goal_Position', motor, commands[motor])
                        positions[motor] = commands[motor]
                        last_route_progress = time.monotonic()
                route_stable = route_stable + 1 if reached else 0
                if route_stable >= 3:
                    route_index += 1
                    route_stable = 0
                    last_route_progress = time.monotonic()
                    route_started = last_route_progress
                    print('Dry-run waypoint reached:', label, flush=True)
                    if route_index == len(route):
                        update(route_complete=True, route_stage='complete; holding outside clearance')
            stop.wait(0.2)
    except Exception as exc:
        update(phase="fault", error=str(exc))
        print("Hold stopped:", exc, flush=True)
    finally:
        if bus.is_connected:
            try:
                if armed:
                    update(phase="stopping")
                    enabled_motors = (*BODY, 'gripper') if gripper_armed else BODY
                    off = {motor: stop_motor(robot, motor) for motor in enabled_motors}
                    if gain_changed:
                        try:
                            if read('Lock', 'elbow_flex') != 1:
                                raise RuntimeError('Lock is not 1; refusing persistent gain write')
                            write_verified('P_Coefficient', 'elbow_flex', original_p)
                            update(gain_restored=True)
                        except Exception as exc:
                            update(error='Gain restore not verified; support arm and power-cycle before reuse: ' + str(exc), gain_restored=False)
                    for motor in enabled_motors:
                        if off[motor]:
                            try:
                                bus.write("Goal_Velocity", motor, speeds[motor], normalize=False, num_retry=2)
                            except Exception as exc:
                                print("Speed restore response:", motor, exc, flush=True)
                    update(torque_off_verified=all(off.values()))
                    if not all(off.values()):
                        update(error="Cannot verify all motors are off. Support the arm and disconnect motor power.")
            finally:
                bus.disconnect(disable_torque=False)
        with lock:
            state["phase"] = "fault" if state["error"] else "stopped"
        print("Hold worker finished; serial port closed.", flush=True)


PAGE = """<!doctype html><html lang="th"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>แรงพยุงแขน</title><style>
body{max-width:850px;margin:24px auto;padding:0 18px;background:#15201b;color:#edf5ef;font:17px/1.7 system-ui}
button{padding:14px 20px;background:#a43e32;color:white;border:0;border-radius:8px;font:inherit;cursor:pointer}
button:disabled{opacity:.4}img{width:100%;margin-top:18px}#error{color:#ffd2a5}
</style><h1>แรงพยุงแขน</h1><div id="status">กำลังตรวจสถานะ…</div>
<p>โหมดพยุงปกติเปิดเฉพาะ 5 แกนแขน กริปเปอร์ยังไม่มีแรงคีบ<br>
โหมดระยะสั้นจะเปิดแรงกริปเปอร์เมื่อกดคำสั่งเปิดหรือปิดเท่านั้น<br>
ขณะเปิดแรง อย่าฝืนจัดแขนด้วยมือ<br>ก่อนปิดแรง ต้องประคองหรือรองรับแขนให้เรียบร้อย</p>
<p>จัดท่าด้วยมือได้เฉพาะตอนปิดแรงพยุงและรองรับแขน ปุ่มบันทึกอ่านตำแหน่งเท่านั้น ไม่ขยับและไม่เขียนทับท่าเดิม</p>
<section style="border:1px solid #668c73;padding:16px;border-radius:10px">
<h2>ควบคุมระยะสั้นจากท่าปัจจุบัน</h2>
<button id="manual">เปิดพยุงพร้อมควบคุมระยะสั้น</button>
<p>ไม่เรียกท่าเก่า · จำกัดข้อต่อครั้งละประมาณ 1° · กริปเปอร์ครั้งละ 80 หน่วยดิบ</p>
<select id="jogMotor" style="font:inherit"><option value="shoulder_pan">ฐาน</option><option value="shoulder_lift">ไหล่</option><option value="elbow_flex">ศอก</option><option value="wrist_flex">ข้อมือพับ</option><option value="wrist_roll">ข้อมือหมุน</option></select>
<button id="minus">−1°</button><button id="plus">+1°</button>
<button id="gripOpen">เปิดกริปเปอร์เล็กน้อย</button><button id="gripClose">ปิดกริปเปอร์เล็กน้อย</button>
<p id="manualState">ยังไม่เปิดโหมดระยะสั้น · ก่อนหยุดต้องรองแขน</p>
</section>
<p><button id="guide">ช่วยจัดท่าเริ่ม · อ่านค่าเท่านั้น ไม่เปิดแรง</button></p><div id="guideReadings"></div>
<p><button id="approach">เข้าท่าเริ่มช้า ๆ · ใช้เมื่อทางโล่ง · ไม่คีบ</button></p>
<p><button id="full_dryrun">เข้าท่าเริ่ม → ลงจุดวาง → ขึ้นกลับ · ไม่คีบ</button></p>
<p><button id="transfer">ไปเหนือช่องขวดแล้วกลับด้านนอก · ไม่ลดจับ · ไม่คีบ</button></p>
<p><button id="transfer24">ทดสอบเส้นทางเหนือช่องขวด · P24 พร้อมชดเชย · ไม่คีบ</button></p>
<p><button id="source24">ทดสอบลงตำแหน่งจับแล้วกลับด้านนอก · มือเปล่า ไม่คีบ</button></p>
<p><button id="return24">ยกกลับจากท่าจับไปด้านนอก · ชดเชยหัวไหล่และศอก · ไม่คีบ</button></p>
<button id="lift">บันทึกท่ายกพ้นตาราง</button>
<button id="place">บันทึกท่าวางนอกตาราง</button>
<button id="retreat">บันทึกท่าเหนือจุดวางด้านนอก</button>
<p><button id="dryrun">ทดสอบมือเปล่า: ลงจุดวางด้านนอกแล้วขึ้นกลับ</button></p>
<p><button id="probe">เปรียบเทียบความเร็วข้อศอก 4° ไปกลับ 2 รอบ — ไม่คีบ</button></p>
<p><button id="gainprobe">ตรวจ P ข้อศอก 16 เทียบ 32 · คืนค่าเดิม · ไม่คีบ</button></p>
<p><button id="baseline">ตรวจข้อศอก 4° รอบเดียว · ใช้ค่าเดิม</button></p>
<p><button id="trimprobe">ทดลองชดเชยข้อศอกแบบจำกัดระยะ · ไม่เปลี่ยน P/I</button></p>
<p><button id="start">ประคองแขนนิ่งแล้ว · เปิดแรงพยุงท่าปัจจุบัน</button></p>
<button id="stop" disabled>รองรับแขนแล้ว · ปิดแรงพยุง</button><div id="error"></div><div id="capture"></div>
<div id="readings"></div><img src="http://127.0.0.1:8765/stream.mjpg" alt="ภาพสดจากกล้อง Pi">
<script>
const button=document.getElementById('stop');
async function poll(){try{
const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
const labels={checking:'กำลังตรวจตำแหน่ง',enabling:'กำลังเปิดแรงพยุง',holding:'เปิดแรงพยุงอยู่ — แขนค้างท่าเดิม',stopping:'กำลังปิดแรงพยุง',stopped:'ปิดแรงพยุงแล้ว ต้องรองรับแขนไว้',fault:'การทดสอบหยุด กรุณารองรับแขน'};
document.getElementById('status').textContent=labels[s.phase]||s.phase;
if(s.phase==='idle')document.getElementById('status').textContent='รอสอนท่า — ยังไม่เปิดแรงพยุง';
const busy=['checking','enabling','holding','stopping','capturing','guiding'].includes(s.phase);
for(const id of ['lift','place','retreat','start','dryrun','probe','gainprobe','baseline','trimprobe','guide','approach','full_dryrun','transfer','transfer24','source24','return24'])document.getElementById(id).disabled=busy;
button.textContent=s.phase==='guiding'?'รองแขนไว้แล้ว · หยุดอ่านตำแหน่ง':'รองรับแขนแล้ว · ปิดแรงพยุง';
if(s.phase==='guiding')document.getElementById('status').textContent='ช่วยจัดท่า — มอเตอร์ปิดแรงอยู่ ให้ประคองน้ำหนักแขน';
const names={shoulder_pan:'ฐานหมุน',shoulder_lift:'หัวไหล่',elbow_flex:'ข้อศอก',wrist_flex:'ข้อมือพับ',wrist_roll:'ข้อมือหมุน'};
document.getElementById('guideReadings').textContent=s.guide ? (s.guide.ready?'ใกล้ท่าเริ่มแล้ว — รองแขนไว้ กดหยุดอ่าน แล้วค่อยทดสอบ':'ค่อย ๆ จัดท่าให้ตัวเลขห่างลดลง โดยไม่ฝืนข้อต่อ')+' · '+Object.entries(s.guide.difference_degrees).map(([m,v])=>names[m]+': ห่าง '+Math.abs(v).toFixed(1)+'°').join(' · ') : '';
if(s.route_stage)document.getElementById('status').textContent+=' · '+s.route_stage;
document.getElementById('error').textContent=s.error||'';
button.disabled=!['checking','enabling','holding','guiding'].includes(s.phase);
document.getElementById('readings').textContent=s.voltage_min ? (s.phase==='holding'?'':'ข้อมูลก่อนหยุด ไม่ใช่ค่าปัจจุบัน · ')+'แรงดัน '+s.voltage_min+' V · อุณหภูมิสูงสุด '+s.temperature_max+' °C · คลาดจากท่าที่ค้าง '+s.max_error_degrees+'°' : '';
let shoulderInfo=document.getElementById('shoulderInfo');
if(!shoulderInfo){shoulderInfo=document.createElement('div');shoulderInfo.id='shoulderInfo';document.getElementById('readings').after(shoulderInfo);}
shoulderInfo.textContent=s.nominal_raw?.shoulder_lift!==undefined && s.readings?.shoulder_lift ? (s.phase==='holding'?'':'ข้อมูลก่อนหยุด · ')+'หัวไหล่ห่างจุดหมาย '+(Math.abs(s.readings.shoulder_lift.Present_Position-s.nominal_raw.shoulder_lift)*360/4095).toFixed(2)+'° · ค่าชดเชย '+((s.route_trim_raw?.shoulder_lift??0)*360/4095).toFixed(2)+'°' : '';
const actual=s.readings?.elbow_flex?.Present_Position;
const nominal=s.nominal_raw?.elbow_flex ?? s.trim_result?.nominal;
if(actual!==undefined && nominal!==undefined){
document.getElementById('readings').textContent=(s.phase==='holding'?'':'ข้อมูลก่อนหยุด · ')+'ศอกห่างจุดหมายจริง '+(Math.abs(actual-nominal)*360/4095).toFixed(2)+'° · ค่าชดเชย '+((s.route_trim_ticks??s.trim_result?.trim_ticks??0)*360/4095).toFixed(2)+'° · แรงดัน '+s.voltage_min+' V · อุณหภูมิสูงสุด '+s.temperature_max+' °C';}
}catch(e){document.getElementById('error').textContent='การเชื่อมต่อขาด กรุณารองรับแขนและตรวจแหล่งจ่ายไฟ';}}
async function sendJog(motor,ticks){const r=await fetch('/jog',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({motor,ticks})}),v=await r.json();if(!r.ok)document.getElementById('manualState').textContent=v.error;else await poll();}
document.getElementById('manual').onclick=async()=>{const r=await fetch('/manual',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}),v=await r.json();if(!r.ok)document.getElementById('manualState').textContent=v.error;else await poll();};
document.getElementById('plus').onclick=()=>sendJog(document.getElementById('jogMotor').value,11);
document.getElementById('minus').onclick=()=>sendJog(document.getElementById('jogMotor').value,-11);
document.getElementById('gripOpen').onclick=()=>sendJog('gripper',80);
document.getElementById('gripClose').onclick=()=>sendJog('gripper',-80);
async function pollManual(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();document.getElementById('manualState').textContent=s.manual_error||(s.manual_mode&&s.phase==='holding'?(s.manual_move_complete?'ถึงเป้าหมายระยะสั้นแล้ว · ยังเปิดแรง':'กำลังขยับระยะสั้น · ยังเปิดแรง'):'ยังไม่เปิดโหมดระยะสั้น');for(const id of ['minus','plus','gripOpen','gripClose'])document.getElementById(id).disabled=!(s.manual_mode&&s.phase==='holding');}catch(e){for(const id of ['minus','plus','gripOpen','gripClose'])document.getElementById(id).disabled=true;document.getElementById('manualState').textContent='การเชื่อมต่อขาด — รองรับแขน';}}
pollManual();setInterval(pollManual,500);
button.onclick=async()=>{button.disabled=true;await fetch('/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('start').onclick=async()=>{await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('dryrun').onclick=async()=>{const r=await fetch('/dryrun',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});const v=await r.json();if(v.error)document.getElementById('capture').textContent=v.error;await poll();};
document.getElementById('probe').onclick=async()=>{await fetch('/probe',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('gainprobe').onclick=async()=>{await fetch('/gainprobe',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('baseline').onclick=async()=>{await fetch('/baseline',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('trimprobe').onclick=async()=>{await fetch('/trimprobe',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('guide').onclick=async()=>{await fetch('/guide',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('approach').onclick=async()=>{await fetch('/approach',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('full_dryrun').onclick=async()=>{await fetch('/full_dryrun',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('transfer').onclick=async()=>{await fetch('/transfer',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('transfer24').onclick=async()=>{await fetch('/transfer24',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('source24').onclick=async()=>{await fetch('/source24',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
document.getElementById('return24').onclick=async()=>{await fetch('/return24',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});await poll();};
for(const stage of ['lift','place','retreat'])document.getElementById(stage).onclick=async()=>{
document.getElementById('capture').textContent='กำลังบันทึก — ประคองให้นิ่ง';
try{const r=await fetch('/capture/'+stage,{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'}),v=await r.json();document.getElementById('capture').textContent=v.error||'บันทึกท่าแล้ว รองรับแขนก่อนปล่อยมือ';}catch(e){document.getElementById('capture').textContent='บันทึกไม่สำเร็จ';}await poll();};
poll();setInterval(poll,500);
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    def send(self, data, content_type="application/json", status=200):
        body = json.dumps(data).encode() if isinstance(data, dict) else data.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(PAGE, "text/html; charset=utf-8")
        elif self.path == "/status":
            with lock:
                snapshot = dict(state)
            self.send(snapshot)
        else:
            self.send({"error": "Not found"}, status=404)

    def do_POST(self):
        global worker
        origin = self.headers.get("Origin")
        if origin and origin not in ("http://127.0.0.1:8766", "http://localhost:8766"):
            self.send({"error": "Invalid origin"}, status=403)
        elif self.path not in ("/stop", "/start", "/manual", "/jog", "/dryrun", "/probe", "/gainprobe", "/baseline", "/trimprobe", "/guide", "/approach", "/full_dryrun", "/transfer", "/transfer24", "/source24", "/return24", "/capture/lift", "/capture/place", "/capture/retreat"):
            self.send({"error": "Not found"}, status=404)
        elif self.headers.get_content_type() != "application/json":
            self.send({"error": "Expected JSON"}, status=415)
        else:
            if self.path == '/jog':
                try:
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 0 < length <= 256:
                        raise ValueError('Invalid request length')
                    request = json.loads(self.rfile.read(length))
                    motor, ticks = request['motor'], request['ticks']
                    maximum = 80 if motor == 'gripper' else 22
                    if motor not in (*BODY, 'gripper') or type(ticks) is not int or not 0 < abs(ticks) <= maximum:
                        raise ValueError('Invalid bounded jog')
                    with lock:
                        available = state.get('phase') == 'holding' and state.get('manual_mode')
                    if not available or stop.is_set():
                        self.send({'error': 'ต้องเปิดโหมดควบคุมระยะสั้นก่อน'}, status=409)
                        return
                    manual_commands.put_nowait({'motor': motor, 'ticks': ticks})
                    self.send({'queued': True})
                except queue.Full:
                    self.send({'error': 'คำสั่งค้างมากเกินไป รอการขยับก่อน'}, status=409)
                except (ValueError, TypeError, KeyError):
                    self.send({'error': 'คำสั่งระยะสั้นไม่ถูกต้อง'}, status=400)
                return
            if self.path == "/stop":
                stop.set()
                self.send({"stopping": True})
                return
            with operation_lock:
                if worker is not None and worker.is_alive():
                    self.send({"error": "ปิดแรงพยุงและรอหยุดก่อนสอนท่า"}, status=409)
                    return
                if self.path in ("/start", "/manual", "/dryrun", "/probe", "/gainprobe", "/baseline", "/trimprobe", "/guide", "/approach", "/full_dryrun", "/transfer", "/transfer24", "/source24", "/return24"):
                    route = None
                    if self.path in ('/dryrun', '/approach', '/full_dryrun', '/transfer', '/transfer24', '/source24', '/return24'):
                        try:
                            data = json.loads(Path(__file__).with_name("pick_place_poses.json").read_text())
                            if data['robot_id'] != 'pawacha_so100_follower':
                                raise ValueError('Robot ID mismatch')
                            poses = data['poses']['bottom_left']
                            route = [(name, {m: int(poses[name]['raw_position'][m]) for m in BODY})
                                     for name in (('retreat',) if self.path == '/approach' else ('retreat', 'place', 'retreat'))]
                            if self.path in ('/transfer', '/transfer24'):
                                route = make_transfer_route(poses, BODY)
                            elif self.path == '/source24':
                                route = make_source_route(poses, BODY)
                            elif self.path == '/return24':
                                route = make_return_route(poses, BODY)
                        except Exception as exc:
                            self.send({'error': str(exc)}, status=409)
                            return
                    stop.clear()
                    while not manual_commands.empty(): manual_commands.get_nowait()
                    with lock:
                        state.clear()
                        state.update(phase="checking", error=None, torque_off_verified=None)
                    worker = (threading.Thread(target=guide) if self.path == '/guide' else
                              threading.Thread(target=hold, args=(route, self.path in ('/probe','/gainprobe','/baseline','/trimprobe'), self.path == '/gainprobe', self.path == '/baseline', self.path == '/trimprobe', route_entry_limit(self.path), 24 if self.path in ('/transfer24','/source24','/return24') else None, self.path == '/manual')))
                    worker.start()
                    self.send({"started": True})
                    return
                update(phase="capturing")
                try:
                    result = subprocess.run([
                        sys.executable, str(Path(__file__).with_name("save_arm_pose.py")),
                        "--name", "bottom_left", "--stage", self.path.rsplit("/", 1)[1],
                    ], capture_output=True, text=True, timeout=20)
                    if result.returncode:
                        self.send({"error": result.stderr[-1800:] or result.stdout[-1800:]}, status=409)
                    else:
                        self.send({"saved": True})
                except Exception as exc:
                    self.send({"error": str(exc)}, status=500)
                finally:
                    update(phase="idle")

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    # Bind first: if the port is busy, no motor connection or writes occur.
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    print("Hold controls: http://127.0.0.1:8766", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=20)
        server.server_close()
