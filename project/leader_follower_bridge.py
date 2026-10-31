"""Relative, clutched five-joint leader/follower bridge. Startup is read-only."""
import json
import queue
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from route_stepper import RouteStepper
from test_so100 import stop_motor

BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
LEADER_PORT = '/dev/cu.usbmodem5B421342561'
FOLLOWER_PORT = '/dev/cu.usbmodem5B421348831'
PORT = 8767
commands = queue.Queue(maxsize=8)
shutdown = threading.Event()
state_lock = threading.Lock()
state = {'phase': 'connecting', 'error': None, 'armed': False}


def update(**values):
    with state_lock:
        state.update(values)


def relative_targets(leader, leader_anchor, follower_anchor, limits):
    targets = {}
    for motor in BODY:
        delta = leader[motor] - leader_anchor[motor]
        if motor == 'wrist_roll':
            delta = (delta + 2048) % 4096 - 2048
        targets[motor] = follower_anchor[motor] + delta
        low, high = limits[motor]
        if not low + 3 <= targets[motor] <= high - 3:
            raise ValueError('เป้าหมายเกินช่วงข้อต่อ: ' + motor)
    return targets


def worker():
    leader = SO100Leader(SO100LeaderConfig(port=LEADER_PORT, id='pawacha_so100_leader', use_degrees=True))
    follower = SO100Follower(SO100FollowerConfig(port=FOLLOWER_PORT, id='pawacha_so100_follower', use_degrees=True))
    armed = False
    following = False
    old_speeds = {}

    def read(bus, register, motor):
        try:
            return int(bus.read(register, motor, normalize=False, num_retry=2))
        except Exception as exc:
            role = 'leader' if bus is leader.bus else 'follower'
            raise RuntimeError(f'{role}/{motor}/{register}: {exc}') from exc

    def write(register, motor, value):
        try:
            follower.bus.write(register, motor, value, normalize=False, num_retry=2)
        except ConnectionError:
            if read(follower.bus, register, motor) != value or read(follower.bus, 'Status', motor):
                raise
        if read(follower.bus, register, motor) != value:
            raise RuntimeError('Write readback mismatch: ' + motor + '/' + register)

    def off():
        nonlocal armed, following
        following = False
        if armed:
            confirmed = {m: stop_motor(follower, m) for m in BODY}
            for m in BODY:
                if confirmed[m] and m in old_speeds:
                    write('Goal_Velocity', m, old_speeds[m])
            armed = not all(confirmed.values())
            update(torque_off_verified=not armed)
            if armed:
                raise RuntimeError('ยืนยันปิดแรงไม่ครบ ให้รองแขนและถอดไฟมอเตอร์')
        update(armed=armed)

    try:
        # Do not use device.connect(): it can configure or energize the motors.
        leader.bus.connect()
        follower.bus.connect()
        if not follower.is_calibrated:
            raise RuntimeError('Follower calibration does not match; no writes made')
        leader_cal = leader.bus.read_calibration()
        if any(c.range_max <= c.range_min for c in leader_cal.values()):
            raise RuntimeError('Leader stored joint ranges are invalid')
        # In-memory import only. Relative control does not require overwriting
        # leader homing offsets or saving a new calibration file.
        leader.bus.calibration = leader_cal
        limits = {m: (read(follower.bus, 'Min_Position_Limit', m),
                      read(follower.bus, 'Max_Position_Limit', m)) for m in BODY}
        voltage_limits = {m: (read(follower.bus, 'Min_Voltage_Limit', m),
                              read(follower.bus, 'Max_Voltage_Limit', m)) for m in BODY}
        for bus in (leader.bus, follower.bus):
            if any(read(bus, 'Torque_Enable', m) for m in bus.motors):
                raise RuntimeError('พบมอเตอร์เปิดแรงอยู่ ไม่เปลี่ยนสถานะอัตโนมัติ')
        update(phase='connected', leader_port=LEADER_PORT, follower_port=FOLLOWER_PORT,
               connected=True, gripper_control=False)
        while not shutdown.is_set():
            left = leader.bus.sync_read('Present_Position', normalize=False, num_retry=1)
            right = follower.bus.sync_read('Present_Position', normalize=False, num_retry=1)
            actual = {m: int(right[m]) for m in BODY}
            blocked = [m for m in BODY if not limits[m][0] + 3 <= actual[m] <= limits[m][1] - 3]
            update(leader_raw=left, follower_raw=right, blocked_joints=blocked,
                   ready=not blocked, last_sample=time.time(), armed=armed)
            if any(read(leader.bus, 'Torque_Enable', m) for m in leader.bus.motors):
                raise RuntimeError('แขนสอนมีแรงมอเตอร์ เปิดตามต่อไม่ได้')
            if read(follower.bus, 'Torque_Enable', 'gripper'):
                raise RuntimeError('กริปเปอร์ถูกเปิดแรงโดยโปรแกรมอื่น')
            if armed:
                for m in BODY:
                    status = read(follower.bus, 'Status', m)
                    torque = read(follower.bus, 'Torque_Enable', m)
                    temperature = read(follower.bus, 'Present_Temperature', m)
                    voltage = read(follower.bus, 'Present_Voltage', m)
                    low, high = voltage_limits[m]
                    if not low < voltage < high:
                        voltage = read(follower.bus, 'Present_Voltage', m)
                    if status or torque != 1 or temperature >= 55 or not low < voltage < high:
                        update(last_fault={'motor': m, 'status': status, 'torque': torque,
                                           'temperature': temperature, 'voltage_raw': voltage})
                        raise RuntimeError('มอเตอร์ผิดปกติ: ' + m)
                    if abs(actual[m] - goals[m]) > 35:
                        raise RuntimeError('แขนตามคำสั่งไม่ได้: ' + m)
            try:
                action = commands.get_nowait()
            except queue.Empty:
                action = None
            if action == 'off':
                off()
                update(phase='connected', error=None)
            elif action == 'hold':
                following = False
                update(phase='holding' if armed else 'connected')
            elif action == 'start':
                if blocked:
                    update(error='ประคองแขนทำงานออกจากปลายช่วงก่อน: ' + ', '.join(blocked))
                    continue
                if not armed:
                    if any(read(follower.bus, 'Torque_Enable', m) for m in BODY):
                        raise RuntimeError('มีโปรแกรมอื่นควบคุม follower อยู่')
                    for m in BODY:
                        if read(follower.bus, 'Operating_Mode', m) != 0 or read(follower.bus, 'Status', m):
                            raise RuntimeError('โหมดหรือสถานะไม่พร้อม: ' + m)
                        if read(follower.bus, 'Present_Temperature', m) >= 55 or read(follower.bus, 'Goal_Time', m) != 0:
                            raise RuntimeError('อุณหภูมิหรือเวลาควบคุมไม่พร้อม: ' + m)
                    old_speeds = {m: read(follower.bus, 'Goal_Velocity', m) for m in BODY}
                    initial_voltage = {m: read(follower.bus, 'Present_Voltage', m) for m in BODY}
                    if any(not voltage_limits[m][0] < initial_voltage[m] < voltage_limits[m][1] for m in BODY):
                        raise RuntimeError('แรงดันเริ่มต้นไม่พร้อม')
                    armed = True  # Cleanup covers all writes, including lost ACKs.
                    goals = dict(actual)
                    for m in BODY:
                        write('Goal_Velocity', m, 2)
                        write('Goal_Position', m, actual[m])
                    check = follower.bus.sync_read('Present_Position', normalize=False, num_retry=1)
                    if any(abs(check[m] - actual[m]) > 5 for m in BODY):
                        raise RuntimeError('แขนเลื่อนระหว่างเตรียมเปิดแรง')
                    for m in ('shoulder_lift','elbow_flex','wrist_flex','wrist_roll','shoulder_pan'):
                        write('Torque_Enable', m, 1)
                leader_anchor, follower_anchor = dict(left), dict(actual)
                planner = RouteStepper(actual, limits)
                following = True
                update(phase='following', error=None, armed=True)
            if following:
                try:
                    desired = relative_targets(left, leader_anchor, follower_anchor, limits)
                    next_goals, _ = planner.step(desired, actual)
                except ValueError as exc:
                    following = False
                    update(phase='holding', error=str(exc) + ' — ค้างท่าไว้ กดเริ่มเพื่อจับจุดอ้างอิงใหม่')
                    continue
                for m in BODY:
                    if shutdown.is_set(): break
                    if goals[m] != next_goals[m]:
                        write('Goal_Position', m, next_goals[m])
                        goals[m] = next_goals[m]
                update(nominal_raw=dict(planner.nominal), target_raw=dict(goals))
            shutdown.wait(.05)
    except Exception as exc:
        update(phase='fault', error=str(exc))
    finally:
        if follower.bus.is_connected:
            try:
                off()
            except Exception as exc:
                update(error=str(exc), torque_off_verified=False)
            finally:
                follower.bus.disconnect(disable_torque=False)
        if leader.bus.is_connected:
            leader.bus.disconnect(disable_torque=False)
        update(connected=False)


PAGE = '''<!doctype html><meta charset="utf-8"><title>แขนสอน ↔ แขนทำงาน</title>
<style>body{font:18px/1.7 system-ui;max-width:1000px;margin:24px auto;padding:20px;background:#15201b;color:#eef5ef}button{font:inherit;padding:14px;margin:6px;border:0;border-radius:8px;background:#337b5c;color:white}#error{color:#ffbc9b}table{width:100%;text-align:left}img{max-width:100%}</style>
<h1>แขนสอน ↔ แขนทำงาน</h1><div id="status"></div><div id="error"></div>
<p>เริ่มต้นอ่านค่าเท่านั้น ยังไม่เปิดแรง · ทำงานแบบอ้างอิงท่าปัจจุบัน ไม่กระโดดหาท่าอีกแขน<br>
ทดลองตามเฉพาะ 5 ข้อต่อ กริปเปอร์ยังไม่คีบ · ก่อนเริ่ม เอามือและสิ่งกีดขวางพ้นแขนทำงาน<br>
ค้างท่า = ยังมีแรงพยุง · ก่อนปิดแรงต้องรองแขน ไม่เช่นนั้นแขนอาจร่วง<br>
หากการเชื่อมต่อหรือมอเตอร์ขัดข้อง ระบบอาจปิดแรง ต้องมีที่รองรับระหว่างทดลอง</p>
<button data-action="start">เริ่มตาม / จับจุดอ้างอิงใหม่</button><button data-action="hold">ค้างท่า ไม่ปิดแรง</button><button data-action="off">รองรับแขนแล้ว · ปิดแรง</button>
<p id="ready"></p><table><thead><tr><th>ข้อต่อ</th><th>แขนสอน (ค่าดิบ)</th><th>แขนทำงาน (ค่าดิบ)</th></tr></thead><tbody id="rows"></tbody></table>
<img src="http://127.0.0.1:8765/stream.mjpg" alt="กล้อง Pi">
<script>
const names={shoulder_pan:'ฐาน',shoulder_lift:'หัวไหล่',elbow_flex:'ข้อศอก',wrist_flex:'ข้อมือพับ',wrist_roll:'ข้อมือหมุน',gripper:'กริปเปอร์ (อ่านค่าเท่านั้น)'};
const labels={connecting:'กำลังเชื่อมต่อ',connected:'เชื่อมทั้งสองแขนแล้ว — ปิดแรง',following:'กำลังตามแขนสอน — เปิดแรง',holding:'ค้างท่า — ยังเปิดแรง',fault:'หยุดเพราะข้อผิดพลาด'};
async function poll(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();document.getElementById('status').textContent=labels[s.phase]||s.phase;document.getElementById('error').textContent=s.error||'';document.getElementById('ready').textContent=s.blocked_joints?.length?'ยังเริ่มไม่ได้ ข้อต่อชิด/เลยช่วง: '+s.blocked_joints.map(m=>names[m]).join(', '):'ตำแหน่งอยู่ในช่วง เริ่มจากเคลื่อนแขนสอนช้า ๆ';document.getElementById('rows').replaceChildren(...Object.keys(names).map(m=>{const tr=document.createElement('tr');for(const text of [names[m],s.leader_raw?.[m]??'—',s.follower_raw?.[m]??'—']){const td=document.createElement('td');td.textContent=text;tr.append(td);}return tr;}));for(const b of document.querySelectorAll('button'))b.disabled=!s.connected;}catch(e){document.getElementById('error').textContent='การเชื่อมต่อขาด — รองรับแขนและตรวจมอเตอร์';}}
for(const b of document.querySelectorAll('button'))b.onclick=async()=>{await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:b.dataset.action})});await poll();};poll();setInterval(poll,500);
</script>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, body, status=200, html=False):
        data = (body if html else json.dumps(body)).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == '/': self.send(PAGE, html=True)
        elif self.path == '/status':
            with state_lock: snapshot = dict(state)
            self.send(snapshot)
        else: self.send({'error':'Not found'},404)

    def do_POST(self):
        if self.headers.get('Origin') not in (None, 'http://127.0.0.1:8767','http://localhost:8767'):
            self.send({'error':'Invalid origin'},403); return
        if self.path != '/command' or self.headers.get_content_type() != 'application/json':
            self.send({'error':'Invalid command'},400); return
        try:
            size = int(self.headers.get('Content-Length','0'))
            if not 0 < size <= 128: raise ValueError('Invalid body size')
            action = json.loads(self.rfile.read(size))['action']
            if action not in ('start','hold','off'): raise ValueError('Invalid action')
            commands.put_nowait(action)
            self.send({'queued':action})
        except (ValueError, KeyError, queue.Full) as exc: self.send({'error':str(exc)},400)

    def log_message(self, *args): pass


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1',PORT),Handler)
    thread = threading.Thread(target=worker)
    thread.start()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    print('Leader/follower controls: http://127.0.0.1:8767',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        shutdown.set()
        thread.join(timeout=20)
        server.server_close()
