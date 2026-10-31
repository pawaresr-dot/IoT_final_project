"""Teach four image/joint correspondences; no automatic target execution."""
import json,sys,time,signal,subprocess,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.request import urlopen
import numpy as np
from camera_arm_mapping import capture_sample,interpolate,BODY

ROOT=Path(__file__).resolve().parent
RESUME=Path(sys.argv[sys.argv.index('--mapping')+1]).resolve() if '--mapping' in sys.argv else None
DIRECTORY=RESUME.parent if RESUME else ROOT/'vision_state'/('local_mapping_'+time.strftime('%Y%m%dT%H%M%S'))
DIRECTORY.mkdir(exist_ok=True)
DATA={'schema':'local_camera_joint_mapping_v1','samples':[],'motion_validated':False,
      'working_plane_operator_instruction':'same grasp height and wrist orientation for all four points'}
if RESUME:
    DATA=json.loads(RESUME.read_text())
    if DATA.get('schema')!='local_camera_joint_mapping_v1':raise ValueError('Unknown mapping schema')
mutex=threading.Lock();leader=None;leader_output=None
def get(url):
    with urlopen(url,timeout=3) as r:return json.load(r)
def stop_leader():
    global leader
    if leader and leader.poll() is None:
        leader.send_signal(signal.SIGINT)
        try:leader.wait(timeout=4)
        except subprocess.TimeoutExpired:raise RuntimeError('Leader reader has not stopped')
    leader=None
def capture(pixel):
    stop_leader();deadline=time.monotonic()+4
    while True:
        s=get('http://127.0.0.1:8768/status')
        if s['phase']=='holding':break
        if time.monotonic()>deadline:raise ValueError('รอแขนหยุดก่อนบันทึก')
        time.sleep(.1)
    h=get('http://127.0.0.1:8765/health')
    sample=capture_sample(pixel,s,h)
    if time.time()-s['last_sample']>.5:raise ValueError('Telemetry stale')
    time.sleep(.2);again=get('http://127.0.0.1:8768/status')
    if any(abs(again['readings'][m]['Present_Position']-sample['raw'][m])>5 for m in BODY):
        raise ValueError('แขนยังเคลื่อนอยู่ ลองบันทึกอีกครั้ง')
    if DATA['samples'] and sample['camera_generation']!=DATA['samples'][0]['camera_generation']:
        raise ValueError('กล้องเชื่อมต่อใหม่ ให้ตรวจมุมกล้องก่อนใช้จุดร่วมกัน')
    if any(np.linalg.norm(np.array(pixel)-old['pixel'])<20 for old in DATA['samples']):
        raise ValueError('จุดใหม่ใกล้จุดเดิมเกินไป ให้เลือกอีกมุมของพื้นที่')
    if len(DATA['samples'])>=4:raise ValueError('บันทึกครบสี่จุดแล้ว')
    DATA['samples'].append(sample)
    (DIRECTORY/'mapping.json').write_text(json.dumps(DATA,indent=2)+'\n')
    return sample
def info():
    points=DATA['samples'];ready=False;reason='ต้องบันทึก 4 จุด'
    if len(points)>=4:
        try:
            center=np.mean([s['pixel'] for s in points],axis=0).tolist()
            candidate=interpolate(points,center)
            ready=True;reason='คำนวณกลางพื้นที่ได้ ยังต้องทดสอบตำแหน่งจริง'
        except ValueError as exc:reason=str(exc)
    return {'samples':points,'ready_for_validation':ready,'reason':reason,
            'following':bool(leader and leader.poll() is None),'file':str(DIRECTORY/'mapping.json')}

PAGE='''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>สอนจุดจับ 4 จุด</title><style>body{background:#14251b;color:#edf7ef;font:19px system-ui;margin:24px;max-width:1250px}button{font:inherit;background:#9bd8af;padding:14px;border:0;border-radius:8px;margin:5px}canvas{max-width:100%;cursor:crosshair;border:1px solid #aaa}#error{color:#ffbb99}</style>
<h1>สอนจุดจับ 4 จุดในพื้นที่เล็ก</h1>
<p>ใช้จุดวงกลมเลข 1 → 2 → 3 → 4 ที่มาร์กในช่องซ้ายล่าง ไม่ใช่มุมกระดาษทั้งแผ่น</p>
<p>1. ก่อนแขนบังภาพ คลิกจุดเป้าหมายที่มีรอยหรือเครื่องหมายให้มองจากด้านข้างได้ ภาพจะหยุดไว้<br>
2. กด “เปิดตามแขนสอน” แล้วจัดกึ่งกลางก้ามตรงจุดนั้น ที่ระดับจับของเดียวกันทุกจุด ข้อมือแนวเดียวกัน<br>
3. กด “หยุดและบันทึกจุด” ระบบอ่านมุมจริงแล้วกลับภาพสด ทำซ้ำจนครบ 4 จุด</p>
<button id="next">เลือกจุดหมายเลขถัดไป</button><button id="follow">เปิดตามแขนสอน</button><button id="save">หยุดและบันทึกจุด</button><button id="live">ดูภาพสด / เลือกใหม่</button>
<p id="status">กำลังอ่านข้อมูล</p><p id="selected">ยังไม่เลือกจุด</p><p id="error"></p>
<canvas id="view" width="1280" height="720"></canvas><p id="points"></p>
<p>หน้านี้บันทึกพิกัด ยังไม่สั่งแขนวิ่งหาจุดที่คำนวณได้</p>
<script>
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');let pixel=null,frozen=false,points=[];
const guides=[[350,450],[500,450],[500,600],[350,600]];
function markers(){ctx.save();ctx.font='bold 26px system-ui';ctx.lineWidth=3;ctx.strokeStyle='#00e5ff';ctx.setLineDash([8,6]);ctx.strokeRect(350,450,150,150);ctx.setLineDash([]);for(const [i,p] of guides.entries()){ctx.fillStyle=i<points.length?'#00ff80':'#00e5ff';ctx.strokeStyle='#10251b';ctx.beginPath();ctx.arc(...p,19,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.fillStyle='#10251b';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),p[0],p[1]);}if(pixel){ctx.strokeStyle='#ffcc00';ctx.lineWidth=5;ctx.beginPath();ctx.arc(...pixel,26,0,Math.PI*2);ctx.stroke();}ctx.restore();}
async function frame(){try{if(!frozen){const im=new Image();im.src='/snapshot.jpg?t='+Date.now();await im.decode();if(!frozen){ctx.drawImage(im,0,0);markers();}}}catch(e){}setTimeout(frame,350);}
function selectGuide(i){if(i>=4)return;pixel=guides[i].slice();frozen=true;markers();document.querySelector('#selected').textContent='จุด '+(i+1)+' ที่ '+pixel.join(', ')+' · ภาพหยุดไว้ จัดกึ่งกลางก้ามตรงวงกลมนี้ ที่ระดับจับเดียวกัน';}
canvas.onclick=e=>{const r=canvas.getBoundingClientRect(),p=[(e.clientX-r.left)*1280/r.width,(e.clientY-r.top)*720/r.height];let i=guides.map((q,i)=>[Math.hypot(q[0]-p[0],q[1]-p[1]),i]).sort((a,b)=>a[0]-b[0])[0][1];selectGuide(i);};
async function refresh(){const s=await(await fetch('/status')).json();points=s.samples;document.querySelector('#status').textContent='บันทึก '+points.length+'/4 จุด · '+(s.following?'กำลังตามแขนสอน':'แขนพยุงนิ่ง')+' · '+s.reason;document.querySelector('#points').textContent=points.map((p,i)=>'จุด '+(i+1)+': '+p.pixel.join(', ')).join(' | ');}
async function send(action){try{document.querySelector('#error').textContent='';const r=await fetch('/command',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,pixel})});const s=await r.json();if(!r.ok)throw Error(s.error);if(action==='capture'){pixel=null;frozen=false;}await refresh();}catch(e){document.querySelector('#error').textContent=e.message;}}
document.querySelector('#next').onclick=()=>selectGuide(points.length);document.querySelector('#follow').onclick=()=>send('follow');document.querySelector('#save').onclick=()=>send('capture');document.querySelector('#live').onclick=()=>{pixel=null;frozen=false;};refresh();frame();setInterval(refresh,1000);
</script></html>'''
class Handler(BaseHTTPRequestHandler):
    def send(self,data,code=200,mime='application/json'):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def do_GET(self):
        try:
            if self.path=='/':self.send(PAGE.encode(),mime='text/html; charset=utf-8')
            elif self.path=='/status':self.send(info())
            elif self.path.startswith('/snapshot.jpg'):
                with urlopen('http://127.0.0.1:8765/snapshot.jpg',timeout=3) as r:self.send(r.read(),mime='image/jpeg')
            else:self.send({'error':'Not found'},404)
        except Exception as exc:self.send({'error':str(exc)},503)
    def do_POST(self):
        global leader,leader_output
        if self.path!='/command':self.send({'error':'Not found'},404);return
        if self.headers.get('Origin') not in (None,'http://127.0.0.1:8770','http://localhost:8770'):
            self.send({'error':'Invalid origin'},403);return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=256:raise ValueError('Invalid size')
            data=json.loads(self.rfile.read(size))
            with mutex:
                if data['action']=='follow':
                    if not leader or leader.poll() is not None:
                        leader_output=(DIRECTORY/('leader_'+str(time.time_ns())+'.log')).open('x')
                        leader=subprocess.Popen([sys.executable,'-u',str(ROOT/'calibration_leader.py'),'--no-leader-range-limit','--scale','1.0','--stream-body-targets'],cwd=ROOT,stdout=leader_output,stderr=subprocess.STDOUT)
                    self.send({'following_started':True})
                elif data['action']=='capture':
                    pixel=data.get('pixel')
                    if not isinstance(pixel,list) or len(pixel)!=2 or not all(isinstance(v,(int,float)) and np.isfinite(v) for v in pixel) or not 0<=pixel[0]<1280 or not 0<=pixel[1]<720:raise ValueError('คลิกเลือกจุดในภาพก่อน')
                    self.send({'sample':capture(pixel)})
                else:raise ValueError('Unknown action')
        except Exception as exc:self.send({'error':str(exc)},409)
    def log_message(self,*args):pass
if __name__=='__main__':
    print('Mapping teach: http://127.0.0.1:8770/',flush=True)
    try:ThreadingHTTPServer(('127.0.0.1',8770),Handler).serve_forever()
    finally:stop_leader()
