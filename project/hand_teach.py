"""Kinesthetic follower teaching and bounded replay. Startup never opens USB.

Body torque stays OFF during recording. Optional gripper buttons are executed
by that same worker; both measured positions and actual jaw commands are saved.
Replay is opt-in, checks the taught start, and never invents Cartesian poses.
"""
import argparse
import json
import os
import queue
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from route_stepper import RouteStepper
from test_so100 import stop_motor

ROOT = Path(__file__).resolve().parent
BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
MOTORS = (*BODY, 'gripper')
SERIAL = '/dev/cu.usbmodem5B421348831'
ROBOT_ID = 'pawacha_so100_follower'


class RecordingJournal:
    """Flush every observed frame; fsync at least once a second and on close."""
    def __init__(self, path, metadata):
        self.path = Path(path)
        self.stream = self.path.open('x')
        self.count = 0
        self.last_sync = time.monotonic()
        try:
            self.write({'type': 'header', 'metadata': metadata}, sync=True)
        except Exception:
            self.stream.close()
            raise

    def write(self, event, sync=False):
        self.stream.write(json.dumps(event, ensure_ascii=False) + '\n')
        self.stream.flush()
        if sync or time.monotonic() - self.last_sync >= 1:
            os.fsync(self.stream.fileno())
            self.last_sync = time.monotonic()

    def frame(self, frame):
        self.write({'type': 'frame', 'frame': frame})
        self.count += 1

    def close(self, complete, error=None):
        if self.stream.closed: return
        try:
            self.write({'type': 'end', 'recording_complete': complete, 'error': error}, sync=True)
        finally:
            self.stream.close()


def recover_journal(path):
    """Recover only real complete JSON lines; interrupted paths are not replay-ready."""
    data, frames, ended = None, [], False
    with Path(path).open() as source:
        for line in source:
            try: event = json.loads(line)
            except json.JSONDecodeError: break  # Possible torn final write.
            if event.get('type') == 'header': data = dict(event['metadata'])
            elif event.get('type') == 'frame': frames.append(event['frame'])
            elif event.get('type') == 'marker' and data is not None:
                data.setdefault('markers', []).append(event['marker'])
            elif event.get('type') == 'end':
                ended = event.get('recording_complete') is True
                if data is not None: data['interruption'] = event.get('error')
    if data is None: raise ValueError('Journal header missing')
    data.update(frames=frames, recording_complete=ended)
    return data


def validate_session(data, limits):
    if data.get('schema') != 'hand_teach_v1' or data.get('robot_id') != ROBOT_ID:
        raise ValueError('ไฟล์สอนไม่ตรงกับแขนนี้')
    if data.get('recording_complete') is False:
        raise ValueError('รอบสอนขาดช่วง เก็บไว้เป็นหลักฐานแต่ไม่เล่นซ้ำอัตโนมัติ')
    frames = data.get('frames', [])
    if not 2 <= len(frames) <= 6000:
        raise ValueError('ต้องมีข้อมูลอย่างน้อย 2 เฟรม และไม่เกิน 6000 เฟรม')
    previous = -1
    for frame in frames:
        if type(frame.get('t')) not in (float, int) or not previous <= frame['t']:
            raise ValueError('เวลาในไฟล์ไม่ถูกต้อง')
        previous = frame['t']
        for m in MOTORS:
            value = frame['raw_position'].get(m)
            low, high = limits[m]
            margin = 3 if m in BODY else 0
            if type(value) is not int or not low + margin <= value <= high - margin:
                raise ValueError('ท่าสอนนอกช่วงข้อต่อ: ' + m)
        grip = frame.get('gripper_goal_raw', frame['raw_position']['gripper'])
        if type(grip) is not int or not limits['gripper'][0] <= grip <= limits['gripper'][1]:
            raise ValueError('เป้ากริปเปอร์นอกช่วง')
    if not any(abs(f['raw_position'][m] - frames[0]['raw_position'][m]) > 5
               for f in frames for m in MOTORS):
        raise ValueError('ข้อมูลยังไม่มีการเคลื่อนที่ให้เล่นซ้ำ')
    return frames


def keyframes(frames):
    """Retain the observed path and jaw commands; never interpolate a new pose."""
    result = [frames[0]]
    for frame in frames[1:-1]:
        last = result[-1]
        if (max(abs(frame['raw_position'][m] - last['raw_position'][m]) for m in BODY) >= 8
                or abs(frame['gripper_goal_raw'] - last['gripper_goal_raw']) >= 4
                or frame['t'] - last['t'] >= .6):
            result.append(frame)
    result.append(frames[-1])
    return result


def gripper_step(command, target, actual, limits):
    low, high = limits
    if not low <= target <= high or not low <= actual <= high:
        raise ValueError('Gripper outside range')
    requested = command + max(-4, min(4, target - command))
    lo, hi = max(low, actual - 30), min(high, actual + 30)
    return max(lo, min(hi, requested))


def return_frames(frames):
    """Reverse the observed body path, without replaying any jaw commands."""
    result = [frames[-1]]
    for frame in reversed(frames[1:-1]):
        if max(abs(frame['raw_position'][m] - result[-1]['raw_position'][m]) for m in BODY) >= 8:
            result.append(frame)
    result.append(frames[0])
    return result


def return_entry(frames, initial):
    """Explicit return only: reverse from the end, or align a nearby start.

    Nearby alignment is limited to 30 degrees per body joint, staged with
    shoulder/elbow before base/wrist. The normal replay entry guard is unchanged.
    """
    if all(abs(initial[m] - frames[-1]['raw_position'][m]) <= 90 for m in BODY):
        return return_frames(frames), 'observed_reverse'
    first = frames[0]['raw_position']
    if any(abs(initial[m] - first[m]) > 341 for m in BODY):
        raise RuntimeError('ไกลทั้งต้นและปลายเส้นทางเกินช่วงกลับอัตโนมัติ 30 องศา')
    middle = dict(initial)
    for m in ('shoulder_lift', 'elbow_flex'): middle[m] = first[m]
    return [{'raw_position': dict(initial)}, {'raw_position': middle}, frames[0]], 'near_start_alignment'


def confirm_temperature(read, motor, reported, previous, pause=time.sleep):
    """Validate an isolated impossible-looking jump while issuing NO motion.

    Keep the 55C cutoff: sustained/elevated readings or any status alarm fail.
    Only a 55..64C singleton after <=50C may be rejected, and only when three
    immediate rereads all show <=50C and status zero. >=65C fails immediately.
    """
    if reported < 55: return reported, []
    if reported >= 65 or previous is None or previous > 50:
        raise RuntimeError('Temperature fault: ' + motor + ' ' + str(reported))
    confirmations = []
    for _ in range(3):
        pause(.02)
        status = read('Status', motor)
        temperature = read('Present_Temperature', motor)
        confirmations.append({'temperature': temperature, 'status': status})
        if status or temperature > 50:
            raise RuntimeError('Temperature/status confirmed: ' + motor + ' ' + json.dumps(confirmations))
    return max(s['temperature'] for s in confirmations), confirmations


def cycle_entry(frames, actual):
    """One taught source only: return if necessary, then replay and release."""
    if all(abs(actual[m] - frames[0]['raw_position'][m]) <= 90 for m in BODY):
        return keyframes(frames), False, 'at_taught_start'
    route, mode = return_entry(frames, actual)
    return route, True, mode


def decode_health_block(payload):
    """STS3215 read-only registers56..70, from one checksum-validated reply."""
    if len(payload) != 15 or any(type(b) is not int or not 0 <= b <= 255 for b in payload):
        raise ValueError('Invalid STS health packet length/data')
    def word(i): return payload[i] | payload[i+1] << 8
    def signed(value, bit): return -(value & ~(1 << bit)) if value & (1 << bit) else value
    return {'Present_Position': signed(word(0),15), 'Present_Velocity': signed(word(2),15),
            'Present_Load': signed(word(4),10), 'Present_Voltage': payload[6],
            'Present_Temperature': payload[7], 'Status': payload[9], 'Moving': payload[10],
            'Present_Current': word(13)}


def read_health(bus, motor):
    if not hasattr(bus, 'packet_handler'):  # Simulated buses in unit tests.
        return {k:int(bus.read(k,motor,normalize=False,num_retry=1)) for k in
                ('Present_Position','Torque_Enable','Status','Present_Voltage','Present_Temperature',
                 'Present_Current','Present_Load')}
    if bus.motors[motor].model != 'sts3215':
        raise RuntimeError('Health block only supports verified STS3215 motors')
    motor_id = bus.motors[motor].id
    for _ in range(2):
        data, communication, error = bus.packet_handler.readTxRx(bus.port_handler,motor_id,56,15)
        if error: raise RuntimeError('Servo packet alarm: ' + motor + ' ' + str(error))
        if communication == 0 and len(data) == 15:
            row = decode_health_block(data)
            row['Torque_Enable'] = int(bus.read('Torque_Enable',motor,normalize=False,num_retry=1))
            return row
        bus.port_handler.clearPort()
    raise RuntimeError('Incomplete/corrupt health packet: ' + motor)


def calibration_jog_target(actual, motor, ticks, limits, origin, local_limit=180):
    if motor not in BODY or type(ticks) is not int or not 0 < abs(ticks) <= 11:
        raise ValueError('Invalid calibration jog')
    target = {m:actual[m] for m in BODY}
    target[motor] += ticks
    lo, hi = limits[motor]
    if not lo+3 <= target[motor] <= hi-3:
        raise ValueError('Jog outside calibrated joint limits: '+motor)
    if local_limit is not None and abs(target[motor]-origin[motor]) > local_limit:
        raise ValueError('Calibration jog outside local range')
    return target


def stream_input_stale(last_update,now):
    return now-last_update>.75


class Controller:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.operation_lock = threading.Lock()
        self.stop = threading.Event()
        self.commands = queue.Queue(maxsize=8)
        self.thread = None
        self.last_file = None
        self.gripper_torque = None
        self.gripper_restore_torque = None
        self.elbow_p = None
        self.restore_elbow_p = None
        self.shoulder_p = None
        self.restore_shoulder_p = None
        self.replay_from = 0
        self.software_temperature_trip = True
        self.calibration_local_limit = 180
        self.teleop_lock=threading.Lock()
        self.teleop_pending=None
        self.state = {'phase': 'idle', 'error': None, 'body_torque_off': None,
                      'recorded_frames': 0, 'object_pickup_verified': False}
        directory = self.root / 'hand_teach_sessions'
        if directory.exists():
            for path in sorted(directory.glob('*.json'), key=lambda p:p.stat().st_mtime, reverse=True):
                try:
                    if path.stat().st_size > 20_000_000: continue
                    data = json.loads(path.read_text())
                    if data.get('recording_complete') is not True: continue
                    validate_session(data, data['limits'])
                    self.last_file = path
                    self.state.update(phase='recorded', saved_file=str(path),
                                      recorded_frames=len(data['frames']), restored=True)
                    break
                except (ValueError, KeyError, TypeError, OSError): continue

    def update(self, **values):
        with self.lock: self.state.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.state, software_temperature_cutoff=self.software_temperature_trip,
                        calibration_local_limit_ticks=self.calibration_local_limit)

    def start(self, operation):
        if operation not in ('record', 'replay', 'return', 'cycle', 'adopt_replay', 'adopt_hold','hold'): raise ValueError('Unknown operation')
        with self.operation_lock:
            if self.thread and self.thread.is_alive():
                if operation in ('return', 'replay', 'cycle') and self.snapshot()['phase'] in ('holding', 'ready_start', 'complete', 'paused', 'released', 'cycle_complete'):
                    self.commands.put_nowait(operation)
                    return
                raise RuntimeError('มีงานกำลังทำอยู่')
            if operation != 'record' and not self.last_file: raise ValueError('ยังไม่มีรอบสอนใหม่')
            if operation == 'record':
                self.last_file = None
                self.update(saved_file=None, checkpoint_file=None, recorded_frames=0,
                            seconds=0, motion_observed=False)
            self.stop.clear()
            while not self.commands.empty(): self.commands.get_nowait()
            self.update(phase='checking', error=None, body_torque_off=None, torque_off_verified=None)
            self.thread = threading.Thread(target=self.run, args=(operation,))
            self.thread.start()

    def jog(self, motor, ticks):
        if motor not in MOTORS or type(ticks) is not int or not 0 < abs(ticks) <= 11:
            raise ValueError('Invalid calibration jog')
        s = self.snapshot()
        if not self.thread or not self.thread.is_alive() or s['phase'] not in ('holding','ready_start','complete','released','cycle_complete'):
            raise ValueError('Wait for stationary monitored hold')
        self.commands.put_nowait({'jog_motor':motor, 'ticks':ticks})

    def jaw(self, action):
        if action not in ('jaw_open','jaw_close','jaw_hold'):raise ValueError('Invalid jaw action')
        s=self.snapshot()
        if not self.thread or not self.thread.is_alive() or s['phase'] not in ('holding','ready_start','complete','released','cycle_complete','jaw_moving','teleoperating'):
            raise ValueError('Wait for stationary monitored arm')
        self.commands.put_nowait(action)

    def teleop(self, targets):
        if not isinstance(targets,dict) or set(targets)!=set(BODY) or any(type(v) is not int for v in targets.values()):
            raise ValueError('Expected five integer body targets; no gripper targets')
        s=self.snapshot()
        if not self.thread or not self.thread.is_alive() or s['phase'] not in ('holding','ready_start','complete','released','cycle_complete','teleoperating'):
            raise ValueError('Arm is not available for body streaming')
        if 'body_joint_limits' not in s:raise ValueError('Calibrated streaming limits are not ready')
        for m in BODY:
            lo,hi=s['body_joint_limits'][m]
            if not lo+3 <= targets[m] <= hi-3:raise ValueError('Stream target outside calibrated joint limits: '+m)
            if self.calibration_local_limit is not None and abs(targets[m]-s['body_origin_raw'][m])>self.calibration_local_limit:
                raise ValueError('Stream target outside local range: '+m)
        with self.teleop_lock:self.teleop_pending=(dict(targets),time.monotonic())

    def run(self, operation):
        robot = None
        enabled, speeds = set(), {}
        success = False
        journal = None
        original_gripper_torque = None
        original_elbow_p = None
        original_shoulder_p = None
        last_temperatures = {}
        temperature_spikes = 0
        def read(k, m):
            return int(robot.bus.read(k, m, normalize=False, num_retry=1))
        def write(k, m, value):
            robot.bus.write(k, m, value, normalize=False, num_retry=1)
            if read(k, m) != value: raise RuntimeError('Readback mismatch: ' + m + '/' + k)
        try:
            owners = subprocess.run(['lsof', '-t', SERIAL], capture_output=True, text=True, timeout=4)
            if owners.returncode not in (0, 1) or owners.stdout.strip():
                raise RuntimeError('มีโปรแกรมอื่นใช้ USB แขนจริงอยู่ ยังไม่เปิดพอร์ตซ้อน')
            robot = SO100Follower(SO100FollowerConfig(port=SERIAL, id=ROBOT_ID))
            robot.bus.connect()
            if not robot.is_calibrated: raise RuntimeError('Calibration ไม่ตรง ยังไม่เปลี่ยนค่าใด ๆ')
            adopting = operation in ('adopt_replay', 'adopt_hold')
            torques = {m: read('Torque_Enable', m) for m in MOTORS}
            if adopting:
                if any(torques[m] != 1 for m in BODY) or torques['gripper'] not in (0, 1):
                    raise RuntimeError('รับช่วงได้เฉพาะแขนที่พยุงครบห้าข้อ')
                enabled.update(m for m in MOTORS if torques[m])
                self.update(hold_adopted=True)
            elif any(torques.values()):
                raise RuntimeError('ต้องปิดแรงมอเตอร์เดิมก่อนเริ่มสอนมือหรือเล่นซ้ำ')
            limits = {m: (read('Min_Position_Limit', m), read('Max_Position_Limit', m)) for m in MOTORS}
            volts = {m: (read('Min_Voltage_Limit', m), read('Max_Voltage_Limit', m)) for m in MOTORS}
            offsets = {m: read('Homing_Offset', m) for m in MOTORS}
            if not self.software_temperature_trip:
                thermal = {m:{k:read(k,m) for k in ('Unloading_Condition','Max_Temperature_Limit')}
                           for m in MOTORS}
                if any(not (row['Unloading_Condition'] & 4) or not 1 <= row['Max_Temperature_Limit'] <= 80
                       for row in thermal.values()):
                    raise RuntimeError('การป้องกันความร้อนในเซอร์โวต้องยังเปิดอยู่ จึงปิดตัวตัดเสริมในโปรแกรมได้')
                self.update(firmware_thermal_protection_verified=True, firmware_thermal_settings=thermal)
            if any(read('Operating_Mode', m) != 0 or read('Goal_Time', m) != 0 for m in MOTORS):
                raise RuntimeError('โหมดหรือ Goal_Time ไม่พร้อม')
            if self.elbow_p is not None:
                baseline_p = read('P_Coefficient', 'elbow_flex')
                if self.elbow_p != 24 or baseline_p not in (16,24) or read('Lock','elbow_flex') != 1:
                    raise RuntimeError('ใช้ได้เฉพาะโปรไฟล์ศอกชั่วคราว P24 โดยไม่ปลดล็อก')
                original_elbow_p = baseline_p
                if self.restore_elbow_p is not None:
                    if self.restore_elbow_p not in (16,24) or self.restore_elbow_p > baseline_p:
                        raise RuntimeError('ค่าคืน P ศอกไม่ถูกต้อง')
                    original_elbow_p = self.restore_elbow_p
                write('P_Coefficient','elbow_flex',24)
                self.update(elbow_p_before=original_elbow_p, elbow_p_active=24)
                time.sleep(.2)
            if self.shoulder_p is not None:
                baseline_p = read('P_Coefficient','shoulder_lift')
                if self.shoulder_p != 24 or baseline_p not in (16,24) or read('Lock','shoulder_lift') != 1:
                    raise RuntimeError('ใช้ได้เฉพาะโปรไฟล์ไหล่ชั่วคราว P24 โดยไม่ปลดล็อก')
                original_shoulder_p = baseline_p
                if self.restore_shoulder_p is not None:
                    if self.restore_shoulder_p not in (16,24) or self.restore_shoulder_p > baseline_p:
                        raise RuntimeError('ค่าคืน P ไหล่ไม่ถูกต้อง')
                    original_shoulder_p = self.restore_shoulder_p
                write('P_Coefficient','shoulder_lift',24)
                self.update(shoulder_p_before=original_shoulder_p,shoulder_p_active=24)
                time.sleep(.2)
            def sample():
                nonlocal temperature_spikes
                rows = {}
                for m in MOTORS:
                    row = read_health(robot.bus, m)
                    rows[m] = row
                    if self.software_temperature_trip and not row['Status'] and row['Present_Temperature'] >= 55:
                        reported = row['Present_Temperature']
                        accepted, confirmations = confirm_temperature(read, m, reported, last_temperatures.get(m))
                        temperature_spikes += 1
                        row.update(Present_Temperature=accepted, reported_temperature=reported)
                        self.update(temperature_spike_count=temperature_spikes,
                                    last_temperature_recheck={'motor': m, 'reported': reported,
                                                              'confirmations': confirmations, 'time': time.time()})
                    if (row['Status'] or (self.software_temperature_trip and row['Present_Temperature'] >= 55)
                            or not volts[m][0] < row['Present_Voltage'] < volts[m][1]
                            or not limits[m][0] <= row['Present_Position'] <= limits[m][1]):
                        self.update(last_fault={'motor': m, 'reading': row})
                        raise RuntimeError('มอเตอร์ผิดปกติ: ' + m + ' ' + json.dumps(row))
                    expected = 1 if m in enabled else 0
                    if row['Torque_Enable'] != expected:
                        raise RuntimeError('สถานะแรงมอเตอร์ไม่ตรง: ' + m)
                    last_temperatures[m] = row['Present_Temperature']
                self.update(readings=rows, last_sample=time.time(), body_torque_off=not any(m in enabled for m in BODY))
                return {m: rows[m]['Present_Position'] for m in MOTORS}
            initial = sample()
            if adopting:
                self.update(gripper_diagnostics={k: read(k, 'gripper') for k in
                            ('Torque_Limit', 'Max_Torque_Limit', 'P_Coefficient', 'Lock', 'Present_Current')})
            if self.gripper_torque is not None:
                before = {k: read(k, 'gripper') for k in ('Torque_Limit', 'Max_Torque_Limit',
                          'Unloading_Condition', 'Protection_Current', 'Max_Temperature_Limit', 'P_Coefficient', 'Lock')}
                requested = self.gripper_torque
                if (type(requested) is not int or not 0 <= requested <= 600 or
                        not before['Torque_Limit'] <= requested <= before['Torque_Limit'] + 100 or before['Lock'] != 1):
                    raise RuntimeError('ปรับแรงชั่วคราวได้ครั้งละไม่เกิน 10% และไม่เกิน 60% โดยไม่ปลดล็อก')
                original_gripper_torque = before['Torque_Limit']
                if self.gripper_restore_torque is not None:
                    if not 0 <= self.gripper_restore_torque <= original_gripper_torque:
                        raise RuntimeError('ค่าคืนแรงหลังรับช่วงต้องไม่เพิ่มแรงเดิม')
                    original_gripper_torque = self.gripper_restore_torque
                write('Torque_Limit', 'gripper', requested)
                after = {k: read(k, 'gripper') for k in before}
                if any(after[k] != v for k,v in before.items() if k != 'Torque_Limit'):
                    raise RuntimeError('ค่าป้องกันกริปเปอร์เปลี่ยนผิดคาด')
                self.update(gripper_torque_before=before, gripper_torque_after=after,
                            gripper_torque_percent=requested / 10)
            if self.stop.is_set(): return
            if operation == 'record':
                frames, markers = [], []
                directory = self.root / 'hand_teach_sessions'
                directory.mkdir(exist_ok=True)
                stem = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
                metadata = {'schema': 'hand_teach_v1', 'robot_id': ROBOT_ID,
                            'recorded_at': datetime.now(timezone.utc).isoformat(),
                            'limits': limits, 'homing_offsets': offsets, 'markers': [],
                            'validation': {'object_pickup_verified': False}}
                journal = RecordingJournal(directory / (stem + '.jsonl'), metadata)
                started = time.monotonic()
                grip_command = grip_target = initial['gripper']
                minimum, maximum = dict(initial), dict(initial)
                self.update(phase='recording', recorded_frames=0, markers=[], gripper_enabled=False,
                            checkpoint_file=str(journal.path), checkpoint_frames=0, motion_observed=False)
                while not self.stop.is_set() and len(frames) < 6000:
                    actual = sample()
                    if self.stop.is_set(): break
                    try: command = self.commands.get_nowait()
                    except queue.Empty: command = None
                    if command in ('grasp', 'lift', 'place'):
                        marker = {'stage': command, 'frame_index': len(frames)}
                        markers.append(marker)
                        journal.write({'type': 'marker', 'marker': marker})
                    elif command in ('open', 'close'):
                        if 'gripper' not in enabled:
                            speeds['gripper'] = read('Goal_Velocity', 'gripper')
                            enabled.add('gripper')
                            write('Goal_Velocity', 'gripper', 2)
                            write('Goal_Position', 'gripper', actual['gripper'])
                            if abs(read('Present_Position', 'gripper') - actual['gripper']) > 5:
                                raise RuntimeError('กริปเปอร์ขยับระหว่างเตรียมเปิดแรง')
                            write('Torque_Enable', 'gripper', 1)
                            grip_command = grip_target = actual['gripper']
                        grip_target = max(limits['gripper'][0], min(limits['gripper'][1],
                                          grip_target + (80 if command == 'open' else -80)))
                        self.update(gripper_enabled=True)
                    if 'gripper' in enabled:
                        if abs(actual['gripper'] - grip_command) > 35:
                            raise RuntimeError('กริปเปอร์ตามคำสั่งไม่ทัน')
                        next_grip = gripper_step(grip_command, grip_target, actual['gripper'], limits['gripper'])
                        if next_grip != grip_command:
                            write('Goal_Position', 'gripper', next_grip)
                            grip_command = next_grip
                    else:
                        grip_command = grip_target = actual['gripper']
                    frames.append({'t': round(time.monotonic() - started, 3),
                                   'raw_position': actual, 'gripper_goal_raw': grip_command})
                    journal.frame(frames[-1])
                    for m in MOTORS:
                        minimum[m] = min(minimum[m], actual[m]); maximum[m] = max(maximum[m], actual[m])
                    self.update(recorded_frames=len(frames), seconds=frames[-1]['t'], markers=list(markers),
                                checkpoint_frames=journal.count,
                                motion_observed=any(maximum[m] - minimum[m] > 5 for m in MOTORS))
                    self.stop.wait(.1)
                if len(frames) >= 6000 and not self.stop.is_set():
                    raise RuntimeError('ถึงเวลาบันทึกสูงสุด เก็บข้อมูลบางส่วนไว้แล้ว แต่ยังไม่ให้เล่นซ้ำ')
                data = dict(metadata, markers=markers, frames=frames, recording_complete=True)
                validate_session(data, limits)
                path = directory / (stem + '.json')
                with path.open('x') as output:
                    json.dump(data, output, ensure_ascii=False, indent=2)
                    output.write('\n')
                    output.flush(); os.fsync(output.fileno())
                journal.close(True)
                self.last_file = path
                self.update(saved_file=str(path), recorded_frames=len(frames), phase='recorded')
                success = True
            else:
                data = json.loads(self.last_file.read_text())
                observed = validate_session(data, limits)
                returning = operation == 'return'
                cycle_active = operation == 'cycle'
                hold_only = operation in ('adopt_hold','hold')
                if hold_only:
                    frames = []
                elif cycle_active:
                    frames, returning, mode = cycle_entry(observed, initial)
                    self.update(cycle_stage='returning_start' if returning else 'picking', return_mode=mode)
                elif returning:
                    frames, mode = return_entry(observed, initial)
                    self.update(return_mode=mode)
                else:
                    frames = keyframes(observed)
                    if not 0 <= self.replay_from < len(frames):
                        raise ValueError('จุดเล่นต่ออยู่นอกเส้นทาง')
                    self.update(playback_offset=self.replay_from)
                    frames = frames[self.replay_from:]
                if data['homing_offsets'] != offsets or any(tuple(data['limits'][m]) != limits[m] for m in MOTORS):
                    raise RuntimeError('Calibration เปลี่ยนหลังสอน ไม่เล่นข้อมูลเก่า')
                first = initial if hold_only else frames[0]['raw_position']
                if any(abs(initial[m] - first[m]) > 90 for m in BODY):
                    self.update(start_difference_degrees={m: round((initial[m] - first[m])*360/4095, 1) for m in BODY})
                    raise RuntimeError('แขนไม่อยู่ใกล้ปลายทางที่สอน จึงยังย้อนเส้นทางไม่ได้' if returning else
                                       'จัดแขนใกล้ท่าเริ่มที่สอนก่อนเล่นซ้ำ ไม่วิ่งย้อนกลับเอง')
                for m in BODY:
                    if not limits[m][0] + 3 <= initial[m] <= limits[m][1] - 3:
                        raise RuntimeError('ตำแหน่งเริ่มชิดขอบ: ' + m)
                goals = dict(initial)
                active = BODY if returning and not cycle_active else MOTORS
                for m in active:
                    if self.stop.is_set(): return
                    speeds[m] = read('Goal_Velocity', m)
                    if adopting and m in enabled:
                        goals[m] = read('Goal_Position', m)
                        if abs(goals[m] - initial[m]) > 35 or speeds[m] not in ((2,6) if m in BODY else (2,)):
                            raise RuntimeError('ค่าพยุงเดิมไม่พร้อมรับช่วง: ' + m)
                        continue  # Preserve the servo hold without a torque gap.
                    enabled.add(m)  # Cleanup covers partial goal preparation too.
                    write('Goal_Velocity', m, 2)
                    write('Goal_Position', m, initial[m])
                if not adopting and any(abs(read('Present_Position', m) - initial[m]) > 5 for m in MOTORS):
                    raise RuntimeError('แขนเลื่อนระหว่างตั้งเป้าหมายเริ่ม')
                for m in ('shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'shoulder_pan', 'gripper'):
                    if m not in active: continue
                    if adopting and torques[m]: continue
                    if self.stop.is_set(): return
                    write('Torque_Enable', m, 1)
                stepper = RouteStepper({m: initial[m] for m in BODY}, {m: limits[m] for m in BODY})
                index, stable = 0, 0
                deadline = time.monotonic() + 30
                total_deadline = time.monotonic() + 900
                paused = False
                release_target = None
                release_deadline = None
                manual_jaw=False
                jaw_last_position=initial['gripper']
                jaw_last_progress=time.monotonic()
                calibration_origin = {m:initial[m] for m in BODY}
                calibration_move = False
                teleop_target=None
                teleop_last_update=0.
                teleop_stepper=None
                teleop_speed_set=False
                self.update(phase='holding' if hold_only else 'returning' if returning else 'replaying', playback_index=0,
                            playback_total=len(frames), gripper_enabled=not returning,command_raw=dict(goals),
                            body_joint_limits={m:list(limits[m]) for m in BODY},body_origin_raw=dict(calibration_origin))
                while not self.stop.is_set():
                    actual = sample()
                    if any(abs(actual[m] - goals[m]) > 35 for m in MOTORS):
                        raise RuntimeError('แขนตามคำสั่งไม่ทัน')
                    try: requested = self.commands.get_nowait()
                    except queue.Empty: requested = None
                    with self.teleop_lock:
                        incoming=self.teleop_pending
                        self.teleop_pending=None
                    if incoming is not None and self.snapshot()['phase'] in ('holding','ready_start','complete','released','cycle_complete','teleoperating'):
                        target,received=incoming
                        for m in BODY:
                            lo,hi=limits[m]
                            if not lo+3 <= target[m] <= hi-3:
                                raise ValueError('Stream target outside calibrated joint limits: '+m)
                            if self.calibration_local_limit is not None and abs(target[m]-calibration_origin[m])>self.calibration_local_limit:
                                raise ValueError('Stream target outside local range: '+m)
                        if teleop_stepper is None:
                            teleop_stepper=RouteStepper({m:actual[m] for m in BODY},{m:limits[m] for m in BODY},max_step=4,independent=True)
                        if not teleop_speed_set:
                            for m in BODY:write('Goal_Velocity',m,6)
                            teleop_speed_set=True
                        teleop_target,teleop_last_update=target,received
                        index=len(frames)
                        self.update(phase='teleoperating',teleop_target_raw=dict(target),teleop_input_stale=False,
                                    body_goal_velocity_raw=6,teleop_step_ticks=4)
                    if requested in ('jaw_open','jaw_close','jaw_hold'):
                        teleop_target=teleop_stepper=None
                        if self.snapshot()['phase'] not in ('holding','ready_start','complete','released','cycle_complete','jaw_moving','teleoperating'):
                            self.update(calibration_error='Wait for stationary arm before jaw command')
                        elif requested=='jaw_hold':
                            release_target=None
                            self.update(phase='holding',jaw_motion_reached=False,jaw_holding=True)
                        else:
                            teleop_target=teleop_stepper=None
                            manual_jaw=True
                            lo,hi=limits['gripper']
                            release_target=min(hi-3,2373) if requested=='jaw_open' else lo+3
                            release_deadline=time.monotonic()+20
                            jaw_last_position=actual['gripper']
                            jaw_last_progress=time.monotonic()
                            self.update(phase='jaw_moving',jaw_target_raw=release_target,
                                        jaw_motion_reached=False,jaw_stalled=False,error=None)
                        requested=None
                    if isinstance(requested, dict) and 'jog_motor' in requested:
                        motor, ticks = requested['jog_motor'], requested['ticks']
                        if self.snapshot()['phase'] not in ('holding','ready_start','complete','released','cycle_complete'):
                            self.update(calibration_error='Wait for previous movement')
                        else:
                            try:
                                if motor == 'gripper':
                                    target = {m:actual[m] for m in BODY}
                                    lo,hi=limits['gripper']
                                    if lo+3 > min(hi-3,2373):
                                        raise ValueError('Invalid calibrated gripper range')
                                    grip_goal=max(lo+3,min(min(hi-3,2373),actual['gripper']+ticks))
                                else:
                                    target = calibration_jog_target(actual,motor,ticks,limits,
                                                                    calibration_origin,self.calibration_local_limit)
                            except ValueError as exc:
                                self.update(calibration_error=str(exc))
                            else:
                                frames = [{'raw_position':target,**({'gripper_goal_raw':grip_goal} if motor=='gripper' else {})}]
                                index, stable, paused = 0, 0, False
                                returning, cycle_active, calibration_move = motor!='gripper', False, True
                                release_target = None
                                stepper = RouteStepper({m:actual[m] for m in BODY},{m:limits[m] for m in BODY})
                                deadline, total_deadline = time.monotonic()+30, time.monotonic()+900
                                self.update(phase='calibrating',calibration_error=None,error=None,
                                            playback_index=0,playback_total=1)
                        requested = None
                    if requested == 'release' and self.snapshot()['phase'] in ('complete','holding','released'):
                        manual_jaw=False
                        release_target = max(f['gripper_goal_raw'] for f in observed)
                        release_deadline = time.monotonic() + 15
                        self.update(phase='releasing', release_target_raw=release_target, error=None)
                    if release_target is not None:
                        if abs(actual['gripper']-jaw_last_position)>=2:
                            jaw_last_position=actual['gripper'];jaw_last_progress=time.monotonic()
                        next_grip = gripper_step(goals['gripper'], release_target, actual['gripper'], limits['gripper'])
                        if next_grip != goals['gripper'] and not self.stop.is_set():
                            write('Goal_Position','gripper',next_grip)
                            goals['gripper'] = next_grip
                        self.update(command_raw=dict(goals))
                        if goals['gripper'] == release_target and abs(actual['gripper']-release_target) <= 12:
                            release_target = None
                            self.update(phase='holding' if manual_jaw else 'cycle_complete' if cycle_active else 'released',
                                        release_motion_verified=not manual_jaw,
                                        jaw_motion_reached=manual_jaw,
                                        cycle_stage='finished' if cycle_active else None)
                            cycle_active = False
                        elif manual_jaw and release_target<actual['gripper'] and time.monotonic()-jaw_last_progress>1.2 and abs(goals['gripper']-actual['gripper'])>=24:
                            release_target=None
                            self.update(phase='holding',jaw_stalled=True,jaw_motion_reached=False,
                                        jaw_holding=True,object_pickup_verified=False)
                        elif time.monotonic() > release_deadline:
                            release_target = None
                            self.update(phase='paused', error='ก้ามเปิดตามเป้าไม่ทัน ยังพยุงแขนอยู่')
                    if requested in ('return', 'replay', 'cycle'):
                        teleop_target=teleop_stepper=None
                        if teleop_speed_set:
                            for m in BODY:write('Goal_Velocity',m,2)
                            teleop_speed_set=False
                            self.update(body_goal_velocity_raw=2)
                        manual_jaw=False
                        calibration_move = False
                        try:
                            next_returning = requested == 'return'
                            if requested == 'cycle':
                                new_frames, next_returning, mode = cycle_entry(observed, actual)
                                self.update(return_mode=mode, cycle_stage='returning_start' if next_returning else 'picking')
                            elif requested == 'return':
                                new_frames, mode = return_entry(observed, actual)
                                self.update(return_mode=mode)
                            else:
                                if any(abs(actual[m] - observed[0]['raw_position'][m]) > 90 for m in BODY):
                                    raise ValueError('ต้องกลับจุดเริ่มก่อนเล่นใหม่')
                                new_frames = keyframes(observed)
                            if requested in ('replay','cycle') and 'gripper' not in enabled:
                                speeds['gripper'] = read('Goal_Velocity', 'gripper')
                                enabled.add('gripper')
                                write('Goal_Velocity', 'gripper', 2)
                                write('Goal_Position', 'gripper', actual['gripper'])
                                goals['gripper'] = actual['gripper']
                                write('Torque_Enable', 'gripper', 1)
                                self.update(gripper_enabled=True)
                            cycle_active = requested == 'cycle'
                            returning = next_returning
                            release_target = None
                            self.update(release_motion_verified=False)
                            frames, index, stable, paused = new_frames, 0, 0, False
                            stepper = RouteStepper({m: actual[m] for m in BODY}, {m: limits[m] for m in BODY})
                            deadline, total_deadline = time.monotonic()+30, time.monotonic()+900
                            self.update(phase='returning' if returning else 'replaying', error=None,
                                        playback_index=0, playback_total=len(frames))
                        except (ValueError, RuntimeError) as exc:
                            self.update(error=str(exc))
                    if teleop_target is not None:
                        if stream_input_stale(teleop_last_update,time.monotonic()):
                            teleop_target=teleop_stepper=None
                            self.update(phase='holding',teleop_input_stale=True)
                        else:
                            commands,reached=teleop_stepper.step(teleop_target,{m:actual[m] for m in BODY})
                            for m in BODY:
                                if self.stop.is_set():break
                                if commands[m]!=goals[m]:write('Goal_Position',m,commands[m]);goals[m]=commands[m]
                            self.update(command_raw=dict(goals),nominal_raw=dict(teleop_stepper.nominal),teleop_reached=reached)
                    if index < len(frames) and not paused and teleop_target is None:
                        if time.monotonic() > deadline or time.monotonic() > total_deadline:
                            paused = True
                            self.update(phase='paused', error='เส้นทางตามไม่ทัน ค้างท่าไว้ รองแขนก่อนหยุด')
                            continue
                        frame = frames[index]
                        target = {m: frame['raw_position'][m] for m in BODY}
                        try:
                            commands, reached = stepper.step(target, {m: actual[m] for m in BODY})
                        except ValueError as exc:
                            paused = True
                            self.update(phase='paused', error=str(exc))
                            continue
                        for m in BODY:
                            if self.stop.is_set(): break
                            if commands[m] != goals[m]:
                                write('Goal_Position', m, commands[m]); goals[m] = commands[m]
                        if not returning:
                            grip_target = frame['gripper_goal_raw']
                            next_grip = gripper_step(goals['gripper'], grip_target, actual['gripper'], limits['gripper'])
                            if next_grip != goals['gripper'] and not self.stop.is_set():
                                write('Goal_Position', 'gripper', next_grip); goals['gripper'] = next_grip
                            # Jaw encoder equality is not proof of grasping.
                            reached = reached and abs(goals['gripper'] - grip_target) <= (0 if calibration_move else 8)
                        stable = stable + 1 if reached else 0
                        self.update(playback_index=index, command_raw=dict(goals), nominal_raw=dict(stepper.nominal))
                        if stable >= 2:
                            index, stable = index + 1, 0
                            deadline = time.monotonic() + 30
                            if index == len(frames):
                                if cycle_active and returning:
                                    returning = False
                                    frames, index, stable = keyframes(observed), 0, 0
                                    stepper = RouteStepper({m:actual[m] for m in BODY},{m:limits[m] for m in BODY})
                                    deadline, total_deadline = time.monotonic()+30, time.monotonic()+900
                                    self.update(phase='replaying',cycle_stage='picking',playback_index=0,
                                                playback_total=len(frames),error=None)
                                elif cycle_active:
                                    release_target = max(f['gripper_goal_raw'] for f in observed)
                                    release_deadline = time.monotonic()+15
                                    self.update(phase='releasing',cycle_stage='releasing',playback_index=index,
                                                release_target_raw=release_target,error=None)
                                else:
                                    self.update(phase='holding' if calibration_move else 'ready_start' if returning else 'complete', playback_index=index,
                                                object_pickup_verified=False, error=None)
                    self.stop.wait(.1)
                success = True
        except Exception as exc:
            self.update(phase='fault', error=str(exc))
            if journal is not None:
                try:
                    journal.close(False, str(exc))
                    self.update(recovery_file=str(journal.path), checkpoint_frames=journal.count)
                except Exception as save_exc:
                    self.update(checkpoint_error=str(save_exc))
        finally:
            if journal is not None and not journal.stream.closed:
                try: journal.close(False, 'Recording interrupted')
                except Exception as exc: self.update(checkpoint_error=str(exc))
            if robot is not None and robot.bus.is_connected:
                verified = {m: stop_motor(robot, m) for m in enabled}
                if original_elbow_p is not None:
                    try:
                        write('P_Coefficient','elbow_flex',original_elbow_p)
                        self.update(elbow_p_restored=True)
                    except Exception as exc: self.update(restore_warning=str(exc))
                if original_shoulder_p is not None:
                    try:
                        write('P_Coefficient','shoulder_lift',original_shoulder_p)
                        self.update(shoulder_p_restored=True)
                    except Exception as exc: self.update(restore_warning=str(exc))
                if original_gripper_torque is not None:
                    try:
                        write('Torque_Limit', 'gripper', original_gripper_torque)
                        self.update(gripper_torque_restored=True)
                    except Exception as exc: self.update(restore_warning=str(exc))
                for m in enabled:
                    if verified[m] and m in speeds:
                        try: write('Goal_Velocity', m, speeds[m])
                        except Exception as exc: self.update(restore_warning=str(exc))
                robot.bus.disconnect(disable_torque=False)
                if enabled:
                    self.update(torque_off_verified=all(verified.values()), body_torque_off=all(verified.values()))
                    if not all(verified.values()):
                        self.update(phase='fault', error='ยืนยันปิดแรงไม่ครบ รองแขนแล้วถอดไฟมอเตอร์')
            if success and operation in ('replay', 'return', 'cycle', 'adopt_replay') and self.snapshot()['phase'] != 'fault':
                self.update(phase='stopped')


controller = Controller()
PAGE = '''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>สอนมือ → เล่นซ้ำ</title><style>body{font:18px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px;background:#14251b;color:#edf7ef}button{font:inherit;padding:14px;margin:6px;border:0;border-radius:8px;background:#9bd8ad;cursor:pointer}section{background:#23412e;padding:18px;margin:16px 0;border-radius:10px}#error{color:#ffd0a4}img{width:100%}</style>
<h1>สอนด้วยแขนจริง → เล่นซ้ำ</h1><p id="phase">รอเริ่ม</p><p id="error"></p><p id="temperature-policy"></p>
<section><h2>สอนผ่านแขนสอน · ควบคุมก้ามด้วยปุ่ม</h2>
<p>ข้อต่อแขนตามแขนสอน แต่ไม่ใช้เซนเซอร์ก้ามแขนสอนที่มีแจ้งเตือนแรงดัน<br>จัดแขนให้หยุดนิ่งก่อนกดก้าม กดครั้งเดียวเพื่อเปิดหรือปิด ก้ามเคลื่อนช้า ๆ และพักเมื่อปิดติดสิ่งกีดขวาง แต่ยังไม่ถือว่าหยิบสำเร็จ</p>
<button data-action="jaw_open">เปิดก้ามจริง</button><button data-action="jaw_close">ปิดก้ามจริง</button><button data-action="jaw_hold">พักก้ามไว้</button>
<p id="jaw-position"></p></section>
<section><h2>เดโมช่องซ้ายล่าง → นอกตาราง</h2><p>วางของที่ตำแหน่งเดิมที่สอน ปุ่มเดียวจะกลับจุดเริ่มเมื่อจำเป็น เล่นเส้นทาง แล้วเปิดก้ามปล่อยที่ปลายทาง ยังไม่รองรับท่าหยิบของช่องอื่น</p><button data-action="cycle">ยกของออก 1 รอบ · ช่องซ้ายล่าง</button></section>
<section><h2>1. บันทึกหนึ่งรอบด้วยมือ</h2>
<p>เริ่มเหนือวัตถุช่องซ้ายล่าง ประคองแขนไว้ แล้วกดเริ่มสอน<br>ขยับแขนด้วยมือไปจับ → ยก → วางนอกตาราง → ปล่อย → กลับท่าเริ่ม<br>
ระหว่างสอน ข้อต่อแขนทั้งห้าปิดแรง กริปเปอร์เปิดแรงเฉพาะเมื่อกดปุ่ม อย่าเอานิ้วคั่นปากคีบ</p>
<button data-action="record">เริ่มสอนมือ · ไม่เปิดแรงแขน</button>
<button data-action="open">เปิดกริปเปอร์เล็กน้อย</button><button data-action="close">ปิดกริปเปอร์เล็กน้อย</button>
<p><button data-action="grasp">ทำเครื่องหมายจับ</button><button data-action="lift">ทำเครื่องหมายยก</button><button data-action="place">ทำเครื่องหมายวาง</button></p>
<button data-action="stop">วางของและรองแขนแล้ว · จบ/หยุด</button><p id="record"></p><p id="checkpoint"></p><p id="motion"></p></section>
<section><h2>2. ทดลองเล่นซ้ำ</h2><p>วางวัตถุกลับตำแหน่งเดิม จัดแขนใกล้ท่าเริ่ม เอามือพ้นแนวแขน แล้วกดเล่น<br>เมื่อเล่นจบจะยังพยุงอยู่ ต้องรองแขนก่อนกดหยุด โปรแกรมไม่ยืนยันว่าหยิบสำเร็จจาก encoder เพียงอย่างเดียว</p>
<button data-action="return">กลับจุดเริ่มที่บันทึกไว้ · ไม่สั่งก้าม</button>
<button data-action="release">เปิดก้ามถึงท่าเปิดที่สอน · ปล่อยของที่ปลายทาง</button>
<button data-action="replay">เล่นรอบที่บันทึกสำเร็จล่าสุด · แขนจะขยับจริง</button><p id="playback"></p><p id="file"></p></section>
<img src="http://127.0.0.1:8765/stream.mjpg" alt="กล้อง Pi">
<script>
const labels={idle:'พร้อมสอนมือ — ยังไม่เปิดพอร์ต',checking:'กำลังตรวจแขน',recording:'กำลังบันทึก — ประคองแขนและขยับด้วยมือ',recorded:'บันทึกแล้ว — ปิดแรง',replaying:'กำลังเล่นซ้ำ — เปิดแรง',paused:'หยุดเส้นทาง — ยังพยุงอยู่',complete:'เล่นเส้นทางครบแล้ว — ยังพยุงอยู่ ตรวจว่าวัตถุย้ายจริง',stopped:'หยุดและปิดแรงแล้ว',fault:'หยุดเพราะข้อผิดพลาด'};
async function poll(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
labels.returning='กำลังกลับจุดเริ่ม — ไม่สั่งก้าม';labels.ready_start='กลับถึงจุดเริ่มแล้ว — ยังเปิดแรงพยุงอยู่';labels.holding='พักเส้นทาง — ยังเปิดแรงพยุงอยู่';
labels.releasing='กำลังเปิดก้ามที่ปลายทาง — ยังพยุงแขน';labels.released='เปิดก้ามถึงเป้าแล้ว — ยังพยุงแขน';
labels.cycle_complete='เดินเส้นทางและเปิดก้ามครบแล้ว — ตรวจผลวัตถุจริง แขนยังพยุงอยู่';
labels.jaw_moving='กำลังขยับก้ามจริง — แขนยังพยุงอยู่';
labels.teleoperating='กำลังตามแขนสอนพร้อมกันหลายข้อ';
document.getElementById('phase').textContent=labels[s.phase]||s.phase;
document.getElementById('temperature-policy').textContent='ตัวตัดอุณหภูมิเสริมในโปรแกรม: '+(s.software_temperature_cutoff?'เปิด':'ปิด · ยังอ่านค่าและหยุดเมื่อเซอร์โวแจ้งข้อผิดพลาด');
document.getElementById('error').textContent=s.error||'';
document.getElementById('record').textContent='ข้อมูล '+(s.recorded_frames||0)+' เฟรม · '+(s.seconds||0)+' วินาที';
document.getElementById('checkpoint').textContent=s.checkpoint_file?'เซฟระหว่างสอนแล้ว '+(s.checkpoint_frames||0)+' เฟรม · หากขาดการเชื่อมต่อจะเก็บเป็นข้อมูลบางส่วน ไม่เล่นเอง':'';
document.getElementById('motion').textContent=s.phase==='recording'?(s.motion_observed?'ตรวจพบการเคลื่อนที่ของแขน/กริปเปอร์แล้ว':'กำลังรอให้ขยับแขน — ยังไม่พบการเคลื่อนที่'):'';
document.getElementById('playback').textContent=s.playback_total?'เส้นทาง '+s.playback_index+'/'+s.playback_total:'';
document.getElementById('file').textContent=s.saved_file||'';
document.getElementById('jaw-position').textContent=s.readings?'ตำแหน่งก้ามจริง: '+s.readings.gripper.Present_Position:'';
for(const b of document.querySelectorAll('button[data-jaw]'))b.disabled=!s.busy||!['holding','calibrating','ready_start','complete','released','cycle_complete'].includes(s.phase);
for(const b of document.querySelectorAll('button[data-action]')){const a=b.dataset.action,held=['holding','ready_start','complete','paused','released','cycle_complete'].includes(s.phase);b.disabled=a.startsWith('jaw_')?!s.busy||!['holding','ready_start','complete','released','cycle_complete','jaw_moving','teleoperating'].includes(s.phase):a==='record'?s.busy:(a==='replay'||a==='return'||a==='cycle')?(s.busy&&!held)||!s.saved_file:a==='release'?!s.busy||!['holding','complete','released'].includes(s.phase):a==='stop'?!s.busy:s.phase!=='recording';}
}catch(e){document.getElementById('error').textContent='การเชื่อมต่อขาด รองรับแขนและตรวจมอเตอร์';}}
let jawToken=0;
const jawStop=()=>{jawToken++;};
window.addEventListener('pointerup',jawStop);window.addEventListener('pointercancel',jawStop);window.addEventListener('blur',jawStop);
for(const b of document.querySelectorAll('button[data-jaw]')){b.style.touchAction='none';b.onpointerdown=async e=>{
e.preventDefault();const token=++jawToken;
try{while(token===jawToken){const s=await (await fetch('/status',{cache:'no-store'})).json();
if(s.error||s.calibration_error)throw Error(s.error||s.calibration_error);
if(['holding','ready_start','complete','released','cycle_complete'].includes(s.phase)){
if(token!==jawToken)break;
const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'jog',motor:'gripper',ticks:Number(b.dataset.jaw)})});
if(!r.ok)throw Error((await r.json()).error);
}else if(s.phase!=='calibrating')throw Error('แขนไม่พร้อมควบคุมก้าม');
await new Promise(resolve=>setTimeout(resolve,250));}}
catch(e){jawStop();document.getElementById('error').textContent=e.message;}};}
for(const b of document.querySelectorAll('button[data-action]'))b.onclick=async()=>{jawStop();const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:b.dataset.action})}),v=await r.json();if(!r.ok)document.getElementById('error').textContent=v.error;else await poll();};poll();setInterval(poll,500);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, data, code=200, html=False):
        encoded = (data if html else json.dumps(data, ensure_ascii=False)).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(encoded)
    def do_GET(self):
        if self.path == '/': self.send(PAGE, html=True)
        elif self.path == '/status':
            s = controller.snapshot(); s['busy'] = bool(controller.thread and controller.thread.is_alive()); self.send(s)
        else: self.send({'error': 'Not found'}, 404)
    def do_POST(self):
        if self.path != '/command': self.send({'error': 'Not found'}, 404); return
        if self.headers.get('Origin') not in (None, 'http://127.0.0.1:8768', 'http://localhost:8768'):
            self.send({'error': 'Invalid origin'}, 403); return
        if self.headers.get_content_type() != 'application/json': self.send({'error': 'Expected JSON'}, 415); return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 512: raise ValueError('Invalid size')
            payload = json.loads(self.rfile.read(length))
            action = payload['action']
            if action in ('record', 'replay', 'return', 'cycle'): controller.start(action)
            elif action == 'jog': controller.jog(payload['motor'],payload['ticks'])
            elif action == 'teleop':controller.teleop(payload['targets'])
            elif action in ('jaw_open','jaw_close','jaw_hold'):controller.jaw(action)
            elif action == 'stop': controller.stop.set()
            elif action == 'release':
                if controller.snapshot()['phase'] not in ('complete','holding','released'): raise ValueError('ต้องพักที่ปลายทางก่อนเปิดก้าม')
                controller.commands.put_nowait('release')
            elif action in ('open', 'close', 'grasp', 'lift', 'place'):
                if controller.snapshot()['phase'] != 'recording': raise ValueError('ต้องเริ่มบันทึกก่อน')
                controller.commands.put_nowait(action)
            else: raise ValueError('Unknown command')
            self.send({'accepted': action})
        except (ValueError, TypeError, KeyError, RuntimeError, queue.Full) as exc:
            self.send({'error': str(exc)}, 409)
    def log_message(self, *args): pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt-hold-replay', action='store_true',
                        help='Explicit controller handoff only: adopt five already-held body motors and replay')
    parser.add_argument('--adopt-hold', action='store_true', help='Explicit handoff: monitor existing servo hold without starting a path')
    parser.add_argument('--hold-current', action='store_true',help='From torque-off only: seed current measured goals and enable hold without replaying a path')
    parser.add_argument('--gripper-torque', type=int, help='Temporary SRAM torque limit, maximum 600 (60%), at most +100')
    parser.add_argument('--restore-gripper-torque', type=int, help='Preserve the known original, lower torque limit across a controller handoff')
    parser.add_argument('--elbow-p', type=int, choices=[24], help='Temporary, previously tested elbow P24 profile; EEPROM stays locked')
    parser.add_argument('--restore-elbow-p', type=int, choices=[16,24], help='Preserve the known original P across a handoff')
    parser.add_argument('--shoulder-p', type=int, choices=[24], help='Temporary shoulder P24 tracking profile; EEPROM stays locked')
    parser.add_argument('--restore-shoulder-p', type=int, choices=[16,24], help='Preserve the known original shoulder P across a handoff')
    parser.add_argument('--replay-from', type=int, default=0, help='Resume an explicit keyframe index after controller handoff')
    parser.add_argument('--no-software-temp-cutoff', action='store_true',
                        help='Disable ONLY the extra software temperature trip; retain telemetry, servo thermal protection and all status alarms')
    parser.add_argument('--no-calibration-local-limit', action='store_true',
                        help='Disable only the +/-180-tick origin radius; retain calibrated joint limits and bounded jogs')
    args = parser.parse_args()
    controller.gripper_torque = args.gripper_torque
    controller.gripper_restore_torque = args.restore_gripper_torque
    controller.elbow_p = args.elbow_p
    controller.restore_elbow_p = args.restore_elbow_p
    controller.shoulder_p = args.shoulder_p
    controller.restore_shoulder_p = args.restore_shoulder_p
    controller.replay_from = args.replay_from
    controller.software_temperature_trip = not args.no_software_temp_cutoff
    controller.calibration_local_limit = None if args.no_calibration_local_limit else 180
    server = ThreadingHTTPServer(('127.0.0.1', 8768), Handler)
    signal.signal(signal.SIGTERM, lambda *_: controller.stop.set())
    print('Hand teaching: http://127.0.0.1:8768/ (idle; no startup motor I/O)', flush=True)
    if args.hold_current:controller.start('hold')
    elif args.adopt_hold: controller.start('adopt_hold')
    elif args.adopt_hold_replay: controller.start('adopt_replay')
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        controller.stop.set()
        if controller.thread: controller.thread.join(timeout=15)
        server.server_close()
