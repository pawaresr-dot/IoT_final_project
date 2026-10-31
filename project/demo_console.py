"""Read-only submission console: selection, real preflight, sequence preview.

No motion endpoint. Selecting a cell or starting the server never energizes motors.
"""
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from pick_sequence import readiness

ROOT = Path(__file__).resolve().parent
PORT = '/dev/cu.usbmodem5B421348831'
lock = threading.Lock()
snapshot = None


def check_arm():
    global snapshot
    result = {'checked_at': time.time(), 'motors': {}, 'calibrated': False}
    robot = None
    try:
        owners = subprocess.run(['lsof', '-t', PORT], capture_output=True, text=True, timeout=4)
        if owners.returncode not in (0, 1):
            raise RuntimeError('ตรวจผู้ใช้พอร์ตไม่ได้')
        if owners.stdout.strip():
            raise RuntimeError('มีโปรแกรมอื่นใช้พอร์ตแขนอยู่; ไม่เปิด serial ซ้อน')
        robot = SO100Follower(SO100FollowerConfig(port=PORT, id='pawacha_so100_follower'))
        robot.bus.connect()
        result['calibrated'] = robot.is_calibrated
        registers = ('Present_Position', 'Torque_Enable', 'Operating_Mode', 'Status',
                     'Present_Temperature', 'Present_Voltage', 'Min_Voltage_Limit',
                     'Max_Voltage_Limit', 'Min_Position_Limit', 'Max_Position_Limit')
        for m in robot.bus.motors:
            result['motors'][m] = {k: int(robot.bus.read(k, m, normalize=False, num_retry=1))
                                   for k in registers}
    except Exception as exc:
        result['error'] = str(exc)
    finally:
        if robot is not None and robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=False)
    snapshot = result
    return result


PAGE = '''<!doctype html><html lang="th"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>เดโมนำของออกจากตาราง</title>
<style>body{font:18px/1.6 system-ui;background:#14221c;color:#edf6ef;max-width:1080px;margin:auto;padding:24px}button,select{font:inherit;padding:12px;border-radius:8px}button{cursor:pointer;background:#9ad8ae}section{background:#21392b;padding:18px;border-radius:12px;margin:16px 0}.grid{display:grid;grid-template-columns:1.3fr 1fr;gap:18px}img{width:100%}li{margin-bottom:8px}.warn{color:#ffcb93}small{color:#bbd0c2}@media(max-width:700px){.grid{display:block}}</style>
<h1>นำของออกจากตาราง</h1><p>เลือกช่อง → คีบ → ยกออก → ปล่อยที่จุดวางคงที่</p>
<section><label>ช่องต้นทาง <select id="cell"><option value="bottom_left">ซ้ายล่าง</option><option value="top_left">ซ้ายบน</option><option value="top_right">ขวาบน</option><option value="bottom_right">ขวาล่าง</option></select></label>
<button id="check">ตรวจแขนจริง (ไม่ขยับ)</button><p id="result">กำลังอ่านข้อมูลที่บันทึกไว้</p>
<small>หน้านี้แสดงแผนและตรวจมอเตอร์เท่านั้น ยังไม่สั่งหยิบอัตโนมัติ และไม่ใช้แขนสอน</small></section>
<div class="grid"><section><h2>กล้อง Pi</h2><img src="http://127.0.0.1:8765/stream.mjpg" alt="ภาพสดตารางและแขน"><p><a style="color:#b7edc8" href="http://127.0.0.1:8765/" target="_blank">หน้าตั้งกรอบกล้อง</a></p></section>
<section><h2>ความพร้อม</h2><ul id="reasons"></ul><p id="age"></p><table><thead><tr><th>ข้อต่อ</th><th>ตำแหน่ง</th><th>°C</th><th>V</th></tr></thead><tbody id="motors"></tbody></table></section></div>
<section><h2>ลำดับที่เตรียมไว้</h2><ol id="steps"></ol><small>แผนจากท่าที่บันทึก ไม่ใช่ผลการหยิบจริง ช่องที่ไม่มีท่าจะไม่สร้างเส้นทางขึ้นเอง</small></section>
<script>
const names={shoulder_pan:'ฐาน',shoulder_lift:'ไหล่',elbow_flex:'ศอก',wrist_flex:'ข้อมือพับ',wrist_roll:'ข้อมือหมุน',gripper:'กริปเปอร์'};
function item(text){const e=document.createElement('li');e.textContent=text;return e;}
async function refresh(){const r=await fetch('/api/plan?cell='+encodeURIComponent(document.getElementById('cell').value)),s=await r.json();
document.getElementById('result').textContent=s.plan.ready?'ผ่านการตรวจข้อมูล — ยังไม่สั่งขยับ':'ยังไม่พร้อมสั่งหยิบจริง';
document.getElementById('reasons').replaceChildren(...s.plan.reasons.map(item));
document.getElementById('steps').replaceChildren(...s.plan.steps.map(x=>item(x.label)));
document.getElementById('age').textContent=s.snapshot?'ตรวจล่าสุด '+new Date(s.snapshot.checked_at*1000).toLocaleTimeString('th-TH')+' · ไม่ใช่ข้อมูลสตรีมสด':'ยังไม่ได้อ่านมอเตอร์';
document.getElementById('motors').replaceChildren(...Object.entries(s.snapshot?.motors||{}).map(([m,row])=>{const tr=document.createElement('tr');for(const val of [names[m],row.Present_Position,row.Present_Temperature,row.Present_Voltage/10]){const td=document.createElement('td');td.textContent=val;tr.append(td);}return tr;}));}
document.getElementById('cell').onchange=()=>refresh().catch(showError);
function showError(e){document.getElementById('result').textContent='เชื่อมต่อไม่สำเร็จ: '+e.message;}
document.getElementById('check').onclick=async()=>{const b=document.getElementById('check');b.disabled=true;try{document.getElementById('result').textContent='กำลังอ่านแขน — ไม่เปิดแรง';const r=await fetch('/api/check',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});if(!r.ok)throw Error((await r.json()).error);await refresh();}catch(e){showError(e);}finally{b.disabled=false;}};refresh().catch(showError);
</script></html>'''


class Handler(BaseHTTPRequestHandler):
    def reply(self, value, status=200, html=False):
        data = (value if html else json.dumps(value, ensure_ascii=False)).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path)
        if path.path == '/':
            self.reply(PAGE, html=True)
        elif path.path == '/api/plan':
            cell = parse_qs(path.query).get('cell', ['bottom_left'])[0]
            try:
                document = json.loads((ROOT / 'pick_place_poses.json').read_text())
                with lock:
                    current = snapshot
                self.reply({'cell': cell, 'plan': readiness(document, cell, current), 'snapshot': current})
            except Exception as exc:
                self.reply({'error': str(exc)}, 500)
        else:
            self.reply({'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path != '/api/check':
            self.reply({'error': 'Not found'}, 404)
            return
        if self.headers.get('Origin') not in (None, 'http://127.0.0.1:8768', 'http://localhost:8768'):
            self.reply({'error': 'Invalid origin'}, 403)
            return
        if self.headers.get_content_type() != 'application/json':
            self.reply({'error': 'Expected JSON'}, 415)
            return
        if not lock.acquire(blocking=False):
            self.reply({'error': 'กำลังตรวจแขนอยู่'}, 409)
            return
        try:
            self.reply(check_arm())
        finally:
            lock.release()

    def log_message(self, *args):
        pass


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8768), Handler)
    print('Submission console: http://127.0.0.1:8768/ (no startup motor I/O)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
