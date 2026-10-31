"""Gripper controls for the existing single-owner follower controller."""
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

URL = 'http://127.0.0.1:8768'
PAGE = '''<!doctype html><html lang="th"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>ควบคุมก้ามแขนจริง</title>
<style>body{background:#15231c;color:#ecf5ee;font:20px system-ui;max-width:950px;margin:30px auto;padding:24px}button{font:inherit;padding:22px;margin:8px;border:0;border-radius:10px;background:#9bd8af;cursor:pointer}button:disabled{opacity:.4}img{width:100%}a{color:#9bd8af}</style>
<h1>ควบคุมก้ามแขนจริง</h1><p id="status">กำลังเชื่อมต่อ</p>
<button data-action="jaw_open" disabled>กางก้าม</button>
<button data-action="jaw_close" disabled>หุบก้าม</button>
<button data-action="jaw_hold" disabled>พักก้าม</button><p id="error"></p>
<p>กดหนึ่งครั้ง ก้ามจะเคลื่อนไปถึงเป้าหมาย</p>
<p><a href="http://127.0.0.1:8768/">เปิดตัวควบคุมแขนจริง</a></p>
<img src="http://127.0.0.1:8765/stream.mjpg" alt="กล้อง Pi">
<script>
let sending=false;
async function poll(){try{const r=await fetch('/status',{cache:'no-store'}),s=await r.json();if(!r.ok)throw Error(s.error);
document.querySelector('#status').textContent='แขนจริง: '+s.phase+' · ตำแหน่งก้าม '+s.readings.gripper.Present_Position;
document.querySelector('#error').textContent=s.error||'';
for(const b of document.querySelectorAll('button'))b.disabled=sending||!s.jaw_ready;
}catch(e){document.querySelector('#error').textContent=e.message;for(const b of document.querySelectorAll('button'))b.disabled=true;}}
for(const b of document.querySelectorAll('button'))b.onclick=async()=>{sending=true;try{const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:b.dataset.action})});const s=await r.json();if(!r.ok)throw Error(s.error);}catch(e){document.querySelector('#error').textContent=e.message;}finally{sending=false;await poll();}};
poll();setInterval(poll,500);
</script></html>'''

def follower_status():
    with urlopen(URL+'/status',timeout=2) as response:
        s=json.load(response)
    s['jaw_ready']=(s.get('busy') and not s.get('error') and not s.get('calibration_error')
        and 0<=time.time()-s.get('last_sample',0)<.5
        and s.get('phase') in ('holding','ready_start','complete','released','cycle_complete','jaw_moving','teleoperating')
        and s.get('readings',{}).get('gripper',{}).get('Torque_Enable')==1)
    return s

class Handler(BaseHTTPRequestHandler):
    def send(self,data,code=200,html=False):
        body=(data if html else json.dumps(data,ensure_ascii=False)).encode()
        self.send_response(code);self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json')
        self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        try:
            if self.path=='/':self.send(PAGE,html=True)
            elif self.path=='/status':self.send(follower_status())
            else:self.send({'error':'Not found'},404)
        except Exception as exc:self.send({'error':str(exc)},503)
    def do_POST(self):
        try:
            if self.path!='/command':self.send({'error':'Not found'},404);return
            if self.headers.get('Origin') not in (None,'http://127.0.0.1:8767','http://localhost:8767'):
                self.send({'error':'Invalid origin'},403);return
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=128:raise ValueError('Invalid size')
            action=json.loads(self.rfile.read(size))['action']
            if action not in ('jaw_open','jaw_close','jaw_hold'):raise ValueError('Unknown action')
            if not follower_status()['jaw_ready']:raise ValueError('แขนจริงยังไม่พร้อม')
            req=Request(URL+'/command',data=json.dumps({'action':action}).encode(),headers={'Content-Type':'application/json'})
            with urlopen(req,timeout=3) as response:self.send(json.load(response))
        except Exception as exc:self.send({'error':str(exc)},409)
    def log_message(self,*args):pass

if __name__=='__main__':
    print('Gripper controls: http://127.0.0.1:8767/',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8767),Handler).serve_forever()
