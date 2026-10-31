"""Leader-only reader -> bounded follower HTTP jogs. Never opens follower USB.

Leader is always passive. Input faults pause forwarding, never turn off the
independently monitored follower. Starts paused; explicit start clutches anchors.
"""
import json
import queue
import signal
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

BODY = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
LEADER_PORT = '/dev/cu.usbmodem5B421342561'
FOLLOWER_URL = 'http://127.0.0.1:8766'
ROOT = Path(__file__).resolve().parent
shutdown = threading.Event()
commands = queue.Queue(maxsize=8)
lock = threading.Lock()
state = {'phase': 'connecting', 'following': False, 'error': None}


def update(**values):
    with lock: state.update(values)


def follower_status():
    with urlopen(FOLLOWER_URL + '/status', timeout=2) as response:
        return json.load(response)


def follower_jog(motor, ticks):
    req = Request(FOLLOWER_URL + '/jog',
                  data=json.dumps({'motor': motor, 'ticks': ticks}).encode(),
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=2) as response:
        return json.load(response)


def relative_goal(left, left_anchor, right_anchor):
    result = {}
    for m in BODY:
        delta = int(left[m]) - int(left_anchor[m])
        if m == 'wrist_roll': delta = (delta + 2048) % 4096 - 2048
        result[m] = right_anchor[m] + delta
    return result


def next_jog(desired, current):
    # One small command at a time; caller waits for acknowledgement and reach.
    m = max(BODY, key=lambda name: abs(desired[name] - current[name]))
    delta = desired[m] - current[m]
    if abs(delta) <= 3: return None
    return m, max(-11, min(11, delta))


def require_follower(s):
    if s.get('phase') != 'holding' or not s.get('manual_mode') or s.get('error') or s.get('manual_error'):
        raise RuntimeError('แขนจริงไม่พร้อมในโหมดระยะสั้น: ' + str(s.get('error') or s.get('manual_error') or s.get('phase')))
    for m in BODY:
        row = s['readings'][m]
        if row['Status'] or row['Torque_Enable'] != 1 or row['Present_Temperature'] >= 55:
            raise RuntimeError('สถานะแขนจริงผิดปกติ: ' + m)


class TeachingSession:
    def __init__(self):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = ROOT / ('leader_teaching_' + stamp + '_' + uuid.uuid4().hex[:8] + '.json')
        self.data = {'schema': 'leader_teaching_observations_v1',
                     'robot_id': 'pawacha_so100_follower', 'poses': {}}

    def capture(self, stage, left, right):
        if stage not in ('grasp', 'lift', 'place'):
            raise ValueError('Unknown teaching stage')
        require_follower(right)
        if not right.get('manual_move_complete'):
            raise RuntimeError('รอแขนจริงถึงเป้าหมายก่อนบันทึก')
        poses = self.data['poses'].setdefault('bottom_left', {})
        if stage in poses:
            raise ValueError('บันทึกท่านี้แล้ว จะไม่เขียนทับข้อมูลเดิม')
        row = {'recorded_at': datetime.now(timezone.utc).isoformat(),
               'raw_position': {m: r['Present_Position'] for m, r in right['readings'].items()},
               'nominal_raw': dict(right['manual_target_raw']),
               'command_raw': dict(right['target_raw']),
               'leader_raw': dict(left),
               'validation': {'ready_for_motion': False, 'object_grasp_verified': False}}
        poses[stage] = row
        temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('x') as output:
                json.dump(self.data, output, indent=2, ensure_ascii=False)
                output.write('\n')
            temporary.replace(self.path)
        except Exception:
            poses.pop(stage)
            raise
        return str(self.path)


def run_reader():
    robot = SO100Leader(SO100LeaderConfig(port=LEADER_PORT, id='pawacha_so100_leader'))
    session = TeachingSession()
    following = False
    left_anchor = right_anchor = None
    pending = None
    capture_stage = None
    last_seconds, last_progress = None, time.monotonic()
    try:
        owners = subprocess.run(['lsof', '-t', LEADER_PORT], capture_output=True, text=True, timeout=4)
        if owners.returncode not in (0, 1) or owners.stdout.strip():
            raise RuntimeError('พอร์ตแขนสอนถูกใช้งานอยู่ หยุดโดยไม่เปิด serial ซ้อน')
        robot.bus.connect(handshake=False)
        while not shutdown.is_set():
            try:
                left = {m: int(robot.bus.read('Present_Position', m, normalize=False, num_retry=1))
                        for m in BODY}
                for m in BODY:
                    def read(k):
                        try:
                            return int(robot.bus.read(k, m, normalize=False, num_retry=1))
                        except Exception as exc:
                            raise RuntimeError('leader/' + m + '/' + k + ': ' + str(exc)) from exc
                    if read('Torque_Enable') != 0 or read('Status'):
                        raise RuntimeError('แขนสอนต้องปิดแรงและไม่มีแจ้งเตือน: ' + m)
                    if read('Present_Temperature') >= 55:
                        raise RuntimeError('อุณหภูมิแขนสอนสูง: ' + m)
                    voltage, low, high = read('Present_Voltage'), read('Min_Voltage_Limit'), read('Max_Voltage_Limit')
                    if not low < voltage < high:
                        raise RuntimeError('แรงดันแขนสอนนอกช่วงตั้งค่า: ' + m)
                # Gripper is not a control input. Preserve its alarm visibly,
                # but do not couple an unused sensor fault to the five healthy
                # input joints. No alarm clearing or leader motor writes.
                gripper_warning = None
                try:
                    grip_position = int(robot.bus.read('Present_Position', 'gripper', normalize=False, num_retry=1))
                    grip_torque = int(robot.bus.read('Torque_Enable', 'gripper', normalize=False, num_retry=1))
                    grip_status = int(robot.bus.read('Status', 'gripper', normalize=False, num_retry=1))
                    left['gripper'] = grip_position
                    if grip_status:
                        gripper_warning = 'กริปเปอร์แขนสอน Status=' + str(grip_status) + ' — ไม่ใช้ควบคุม'
                except Exception as exc:
                    left['gripper'] = None
                    grip_torque = None
                    gripper_warning = 'กริปเปอร์แขนสอนอ่านไม่ได้ — ไม่ใช้ควบคุม: ' + str(exc)
                if grip_torque == 1:
                    raise RuntimeError('กริปเปอร์แขนสอนเปิดแรงอยู่ หยุดตาม')
                right = follower_status()
                require_follower(right)
                seconds = right.get('seconds')
                if seconds != last_seconds:
                    last_progress, last_seconds = time.monotonic(), seconds
                if time.monotonic() - last_progress > 2:
                    raise RuntimeError('สถานะแขนจริงค้าง หยุดส่งคำสั่ง')
                update(leader_raw=left, follower_raw={m:r['Present_Position'] for m,r in right['readings'].items()},
                       connected=True, last_sample=time.time(), following=following, error=None,
                       gripper_input_enabled=False, gripper_warning=gripper_warning)
                try: action = commands.get_nowait()
                except queue.Empty: action = None
                if action == 'start':
                    if pending or not right.get('manual_move_complete'):
                        update(error='รอแขนจริงถึงเป้าหมายก่อนเริ่มตาม')
                    else:
                        left_anchor, right_anchor = dict(left), dict(right['manual_target_raw'])
                        following, pending, capture_stage = True, None, None
                        update(error=None)
                elif action == 'pause':
                    following, capture_stage = False, None
                elif action in ('grasp', 'lift', 'place'):
                    following, capture_stage = False, action
                elif action in ('open', 'close'):
                    following = False
                    if pending or not right.get('manual_move_complete'):
                        update(error='รอแขนหยุดก่อนใช้กริปเปอร์')
                    else:
                        follower_jog('gripper', 80 if action == 'open' else -80)
                        update(error=None)
                if pending:
                    m, expected, sent = pending
                    if right['manual_target_raw'][m] == expected:
                        pending = None
                    elif time.monotonic() - sent > 3:
                        raise RuntimeError('ยืนยันรับคำสั่งแขนจริงไม่ได้ หยุดส่งคำสั่ง')
                if capture_stage and not pending and right.get('manual_move_complete'):
                    saved = session.capture(capture_stage, left, right)
                    update(saved_file=saved, saved_stages=list(session.data['poses']['bottom_left']), error=None)
                    capture_stage = None
                if following and not pending and right.get('manual_move_complete'):
                    desired = relative_goal(left, left_anchor, right_anchor)
                    # A large input displacement requires reclutching, not a jump.
                    if any(abs(desired[m] - right_anchor[m]) > 180 for m in BODY):
                        raise RuntimeError('แขนสอนพ้นช่วงทดลองประมาณ 16° หยุดตามและค้างท่า')
                    step = next_jog(desired, right['manual_target_raw'])
                    if step:
                        m, ticks = step
                        follower_jog(m, ticks)
                        pending = (m, right['manual_target_raw'][m] + ticks, time.monotonic())
                update(phase='following' if following else 'paused', following=following)
            except Exception as exc:
                # Critical distinction from the old bridge: no follower torque
                # command, reconnect, or serial takeover on an input fault.
                following, pending, capture_stage = False, None, None
                update(phase='paused', following=False, connected=False,
                       error=str(exc), last_input_fault=str(exc), fault_time=time.time())
            shutdown.wait(.15)
    except Exception as exc:
        update(phase='fault', error=str(exc), connected=False, following=False)
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
        update(connected=False, following=False)


PAGE = '''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>สอนแขนจริงด้วยแขนสอน</title><style>body{background:#15231c;color:#ecf5ee;font:18px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px}button{font:inherit;padding:12px;margin:6px;background:#9bd8af;border:0;border-radius:8px;cursor:pointer}section{padding:18px;margin:16px 0;background:#23402d;border-radius:10px}#error{color:#ffce9d}table{width:100%;text-align:left}img{max-width:100%}</style>
<h1>สอนแขนจริงด้วยแขนสอน</h1><p id="status">กำลังเชื่อมต่อแขนสอน</p><p id="error"></p><p id="gripperWarning" style="color:#ffce9d"></p>
<p>โปรแกรมไม่เปิดแรงแขนสอน · เริ่มจากท่าปัจจุบัน ไม่กระโดดไปหาท่าเดียวกัน · ขยับช้า ๆ ทีละข้อต่อ<br>
ตามเฉพาะ 5 ข้อต่อ กริปเปอร์ใช้ปุ่มเปิด/ปิดด้านล่าง · ถ้าอ่านแขนสอนขัดข้อง จะหยุดส่งคำสั่ง แต่แขนจริงยังพยุงด้วยตัวควบคุมเดิม</p>
<section><button data-action="start">เริ่มตาม / จับจุดอ้างอิงใหม่</button><button data-action="pause">หยุดตาม · ยังพยุงแขนจริง</button>
<button data-action="open">เปิดกริปเปอร์เล็กน้อย</button><button data-action="close">ปิดกริปเปอร์เล็กน้อย</button></section>
<section><h2>บันทึกท่าช่องซ้ายล่าง</h2><p>เมื่อถึงท่าจริง กดบันทึก ระบบจะหยุดตามและเก็บตำแหน่งแขนจริง ไม่เขียนทับท่าเดิม</p>
<button data-action="grasp">บันทึกท่าจับ</button><button data-action="lift">บันทึกท่ายก</button><button data-action="place">บันทึกท่าวาง</button><p id="saved"></p>
<small>ท่าที่เก็บเป็น observation ยังไม่ใช่เส้นทางหยิบอัตโนมัติที่ทดสอบผ่าน</small></section>
<table><thead><tr><th>ข้อต่อ</th><th>แขนสอน</th><th>แขนจริง</th></tr></thead><tbody id="rows"></tbody></table>
<p><a style="color:#b9efca" href="http://127.0.0.1:8766/">ตัวควบคุมแขนจริงและปุ่มปิดแรง</a></p>
<img src="http://127.0.0.1:8765/stream.mjpg" alt="กล้อง Pi">
<script>
const names={shoulder_pan:'ฐาน',shoulder_lift:'ไหล่',elbow_flex:'ศอก',wrist_flex:'ข้อมือพับ',wrist_roll:'ข้อมือหมุน',gripper:'กริปเปอร์ (ไม่ตามอัตโนมัติ)'};
async function poll(){try{
const r=await fetch('/status',{cache:'no-store'}),s=await r.json();
document.getElementById('status').textContent=s.following?'กำลังตามแขนสอน':'หยุดตาม / อ่านตำแหน่ง';
document.getElementById('error').textContent=s.error||'';
document.getElementById('gripperWarning').textContent=s.gripper_warning||'กริปเปอร์แขนสอนไม่ได้ใช้ควบคุม — ใช้ปุ่มเปิด/ปิดแขนจริง';
document.getElementById('saved').textContent=s.saved_stages?.length?'บันทึกแล้ว: '+s.saved_stages.join(', '):'ยังไม่บันทึกท่า';
document.getElementById('rows').replaceChildren(...Object.keys(names).map(m=>{const tr=document.createElement('tr');for(const val of [names[m],s.leader_raw?.[m]??'—',s.follower_raw?.[m]??'—']){const td=document.createElement('td');td.textContent=val;tr.append(td);}return tr;}));
for(const b of document.querySelectorAll('button'))b.disabled=!s.connected;
}catch(e){document.getElementById('error').textContent='การเชื่อมต่อหน้าสอนขาด — ดูสถานะแขนจริงที่พอร์ต8766';}}
for(const b of document.querySelectorAll('button'))b.onclick=async()=>{const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:b.dataset.action})});if(!r.ok)document.getElementById('error').textContent=(await r.json()).error;else await poll();};poll();setInterval(poll,500);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def send(self, data, code=200, html=False):
        encoded = (data if html else json.dumps(data, ensure_ascii=False)).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(encoded)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == '/': self.send(PAGE, html=True)
        elif self.path == '/status':
            with lock: snapshot = dict(state)
            self.send(snapshot)
        else: self.send({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path != '/command': self.send({'error': 'Not found'}, 404); return
        if self.headers.get('Origin') not in (None, 'http://127.0.0.1:8767', 'http://localhost:8767'):
            self.send({'error': 'Invalid origin'}, 403); return
        if self.headers.get_content_type() != 'application/json':
            self.send({'error': 'Expected JSON'}, 415); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 128: raise ValueError('Invalid size')
            action = json.loads(self.rfile.read(size))['action']
            if action not in ('start', 'pause', 'open', 'close', 'grasp', 'lift', 'place'):
                raise ValueError('Unknown action')
            commands.put_nowait(action)
            self.send({'queued': action})
        except (ValueError, TypeError, KeyError, queue.Full) as exc:
            self.send({'error': str(exc)}, 400)

    def log_message(self, *args): pass


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8767), Handler)
    worker = threading.Thread(target=run_reader)
    worker.start()
    signal.signal(signal.SIGTERM, lambda *_: shutdown.set())
    print('Leader teaching: http://127.0.0.1:8767/ (leader USB only, starts paused)', flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        shutdown.set()
        worker.join(timeout=10)
        server.server_close()
