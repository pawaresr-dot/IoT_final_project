"""Camera-validated fixed-point taught pickup. Not a general ACT policy."""
import argparse,json,time,io
from pathlib import Path
from urllib.request import Request,urlopen
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import cv2
import numpy as np
from pick_verification import choose_grasp_cycle,verify_follow,retry_allowed

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'training_captures/20261031T190513_a9654b10'
BODY=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll')
CTRL='http://127.0.0.1:8768'
def get(url):
    with urlopen(url,timeout=3) as r:return json.load(r)
def snapshot():
    deadline=time.monotonic()+3
    while True:
        h=get('http://127.0.0.1:8765/health')
        if h.get('live') and h['frame_age_seconds']<=.25:break
        if time.monotonic()>deadline:raise ValueError('Camera stale')
        time.sleep(.05)
    with urlopen('http://127.0.0.1:8765/snapshot.jpg',timeout=3) as r:data=r.read()
    image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
    if image is None or image.shape[:2]!=(720,1280):raise ValueError('Unexpected camera frame')
    return image
def detect(image,bottom=660):
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(0,0,0),(179,255,85))
    mask[:150]=0;mask[bottom:]=0;mask[:,:230]=0;mask[:,930:]=0
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    objects=[]
    for c in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
        x,y,w,h=cv2.boundingRect(c);area=cv2.contourArea(cv2.convexHull(c))
        if not 2000<area<35000 or not 1.2<max(w,h)/min(w,h)<4:continue
        rect=cv2.minAreaRect(c);corners=cv2.boxPoints(rect)
        edges=np.roll(corners,-1,axis=0)-corners;edge=edges[np.argmax(np.linalg.norm(edges,axis=1))]
        angle=float(np.degrees(np.arctan2(edge[1],edge[0]))%180)
        objects.append({'center':[float(v) for v in rect[0]],'box':[x,y,w,h],
            'angle':angle,'area':float(area)})
    return objects
def reference():
    candidates=detect(cv2.imread(str(SOURCE/'frames/000000.jpg')))
    if len(candidates)!=1:raise ValueError('Ambiguous taught reference')
    return candidates[0]
def assess(image):
    ref=reference();found=detect(image)
    result={'reference':ref,'candidates':found,'ready':False,
        'mode':'camera-validated taught fixed-point trajectory','grasp_visibility':'occluded during descent; taught joint pose required'}
    if len(found)!=1:result['reason']='Object missing, ambiguous or occluded';return result
    obj=found[0];distance=float(np.linalg.norm(np.array(obj['center'])-ref['center']))
    angle=abs((obj['angle']-ref['angle']+90)%180-90)
    ratio=obj['area']/ref['area']
    result.update(center_error_pixels=distance,angle_error_degrees=angle,area_ratio=ratio)
    result['ready']=distance<=20 and angle<=12 and .7<=ratio<=1.3
    result['reason']='Ready at taught pickup point' if result['ready'] else 'Move and rotate object into green reference outline'
    return result
def status():
    s=get(CTRL+'/status')
    if s.get('error') or s.get('calibration_error') or not 0<=time.time()-s.get('last_sample',0)<.5:
        raise RuntimeError('Controller unavailable')
    if any(s['readings'][m]['Status'] or s['readings'][m]['Torque_Enable']!=1 for m in (*BODY,'gripper')):
        raise RuntimeError('Servo not ready')
    return s
def command(action,**kwargs):
    req=Request(CTRL+'/command',data=json.dumps({'action':action,**kwargs}).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=3) as r:return json.load(r)
def move(target,timeout=35,precision=False):
    end=time.monotonic()+timeout
    correction={m:0 for m in BODY}
    while time.monotonic()<end:
        s=status();actual={m:s['readings'][m]['Present_Position'] for m in BODY}
        for m in BODY:
            lo,hi=s['body_joint_limits'][m]
            if not lo+3<=target[m]<=hi-3:raise ValueError('Taught target outside limit: '+m)
        tolerance=6 if precision else 18
        if all(abs(target[m]-actual[m])<=tolerance for m in BODY):return
        issued=dict(target)
        if precision:
            for m in BODY:
                error=target[m]-actual[m]
                if abs(error)>6 and abs(error)<30:
                    correction[m]=max(-24,min(24,correction[m]+(1 if error>0 else -1)))
                lo,hi=s['body_joint_limits'][m]
                issued[m]=max(lo+3,min(hi-3,target[m]+correction[m]))
        command('teleop',targets=issued);time.sleep(.1)
    raise RuntimeError('Taught waypoint did not converge')
def jaw(action):
    # Clear outstanding body motion and stale jaw completion flags first.
    command('jaw_hold');hold_deadline=time.monotonic()+3
    while True:
        s=status()
        if s['phase']=='holding' and not s.get('jaw_motion_reached'):break
        if time.monotonic()>hold_deadline:raise RuntimeError('Jaw hold acknowledgement timeout')
        time.sleep(.05)
    issued=time.time();command(action);end=time.monotonic()+22
    while time.monotonic()<end:
        s=status()
        if s['last_sample']>issued and s['phase']=='holding' and (s.get('jaw_motion_reached') or s.get('jaw_stalled')):return s
        time.sleep(.1)
    raise RuntimeError('Jaw motion timeout')
def follow(rows):
    last=None
    for r in rows:
        target=dict(zip(BODY,r['state_after_raw'][:5]))
        if last is None or max(abs(target[m]-last[m]) for m in BODY)>=8:
            move(target);last=target
    if rows:move(dict(zip(BODY,rows[-1]['state_after_raw'][:5])))
def appearance_track(image):
    reference_image=cv2.imread(str(SOURCE/'frames/000000.jpg'))
    x,y,w,h=reference()['box'];template=reference_image[y:y+h,x:x+w]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    best=None
    for rotation in (None,cv2.ROTATE_90_CLOCKWISE,cv2.ROTATE_180,cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated=template if rotation is None else cv2.rotate(template,rotation)
        for angle in (-30,-15,0,15,30):
            hh,ww=rotated.shape
            matrix=cv2.getRotationMatrix2D((ww/2,hh/2),angle,1)
            angled=cv2.warpAffine(rotated,matrix,(ww,hh),borderMode=cv2.BORDER_REPLICATE)
            for scale in (.8,1.,1.2):
                resized=cv2.resize(angled,None,fx=scale,fy=scale)
                scores=cv2.matchTemplate(gray,resized,cv2.TM_CCOEFF_NORMED)
                _,score,_,point=cv2.minMaxLoc(scores)
                if best is None or score>best['score']:
                    best={'score':float(score),'center':[point[0]+resized.shape[1]/2,point[1]+resized.shape[0]/2]}
    return best
def inspect_pick(rows,grasp,trial,attempt,origin_center):
    # Reverse the taught approach with the jaw closed to expose the source.
    # Use a small A-B-A base probe only if an object appearance is visible.
    follow(list(reversed(rows[:grasp+1])))
    image=snapshot();cv2.imwrite(str(trial/f'attempt_{attempt}_probe_a.jpg'),image)
    source_visible=any(np.linalg.norm(np.array(c['center'])-origin_center)<35 for c in detect(image))
    a=appearance_track(image)
    if source_visible:return verify_follow(True,[])
    if a and a['score']>=.70 and 230<a['center'][0]<930 and 250<a['center'][1]<660:
        return {'verdict':'not_picked','reason':'Object visibly remains on work surface away from raised gripper',
                'object_center':a['center'],'match_score':a['score']}
    if a is None or a['score']<.70:return verify_follow(False,[a])
    initial=status();origin={m:initial['readings'][m]['Present_Position'] for m in BODY}
    probe=dict(origin);probe['shoulder_pan']+=30
    lo,hi=initial['body_joint_limits']['shoulder_pan']
    if not lo+3<=probe['shoulder_pan']<=hi-3:return {'verdict':'unknown','reason':'Probe pose unavailable'}
    try:
        move(probe,precision=True);image=snapshot();b=appearance_track(image)
        cv2.imwrite(str(trial/f'attempt_{attempt}_probe_b.jpg'),image)
    finally:move(origin,precision=True)
    image=snapshot();c=appearance_track(image)
    cv2.imwrite(str(trial/f'attempt_{attempt}_probe_return.jpg'),image)
    return verify_follow(False,[a,b,c])
def execute(max_attempts=3):
    if not 1<=max_attempts<=3:raise ValueError('Attempts must be between 1 and 3')
    # Recheck the live visual precondition; the UI alone never authorizes motion.
    for _ in range(5):
        check=assess(snapshot())
        if not check['ready']:raise RuntimeError(check['reason']+' '+str(check.get('center_error_pixels')))
        time.sleep(.15)
    manifest=json.loads((SOURCE/'manifest.json').read_text())
    if not manifest.get('task_success_operator_confirmed'):raise ValueError('Unconfirmed taught route')
    rows=[json.loads(x) for x in (SOURCE/'samples.jsonl').read_text().splitlines()]
    goals=[r['action_after_raw'][5] for r in rows]
    grasp=next(i for i,v in enumerate(goals) if v<goals[0]-12)
    grasp,closed_index,release,opened_index=choose_grasp_cycle(rows)
    trial=ROOT/'vision_state'/('vision_pick_'+time.strftime('%Y%m%dT%H%M%S'));trial.mkdir()
    result={'source':str(SOURCE),'mode':'vision-validated taught trajectory',
            'initial_vision':check,'autonomous_pickup_verified':False,'stages':[],
            'max_attempts':max_attempts,'attempts':[],'cycle_indices':[grasp,closed_index,release,opened_index]}
    try:
        for attempt in range(1,max_attempts+1):
            current=assess(snapshot())
            if not current['ready']:raise RuntimeError('Object moved outside taught grasp coverage')
            status();jaw('jaw_open')
            print('Attempt',attempt,'approaching taught grasp pose',flush=True);follow(rows[:grasp+1])
            target=dict(zip(BODY,rows[grasp]['state_after_raw'][:5]));move(target,precision=True)
            s=status()
            if any(abs(s['readings'][m]['Present_Position']-target[m])>6 for m in BODY):raise RuntimeError('Grasp pose mismatch')
            cv2.imwrite(str(trial/f'attempt_{attempt}_before_close.jpg'),snapshot())
            print('Closing and checking object follow',flush=True);closed=jaw('jaw_close')
            verification=inspect_pick(rows,grasp,trial,attempt,np.array(current['candidates'][0]['center']))
            result['attempts'].append({'attempt':attempt,'gripper_after_close':closed['readings']['gripper'],
                'grasp_pose_actual':{m:s['readings'][m]['Present_Position'] for m in BODY},'verification':verification})
            print('Verification:',verification,flush=True)
            if verification['verdict']=='followed_visually':break
            if verification['verdict']=='unknown':raise RuntimeError('Cannot verify grasp: '+verification['reason'])
            jaw('jaw_open')
            aligned=assess(snapshot())['ready']
            if not retry_allowed(verification['verdict'],attempt,max_attempts,aligned):
                raise RuntimeError('Pickup failed; attempt limit reached or object moved')
            print('Object stayed at source; retrying',flush=True)
        # Verification returns to a raised approach pose with the object held.
        # Re-enter the demonstrated path with jaw kept closed before transfer.
        follow(rows[:grasp+1])
        result['stages'].append('object_follow_verified_visually_not_height')
        print('Transferring taught route',flush=True);follow(rows[grasp:release+1])
        jaw('jaw_open');result['stages'].append('release_command_completed')
        follow(rows[release:]);result['stages'].append('retreat_completed')
        final=snapshot();cv2.imwrite(str(trial/'final.jpg'),final)
        result['final_detection']=assess(final)
        result['object_absent_at_source']=not any(np.linalg.norm(np.array(c['center'])-check['reference']['center'])<35 for c in result['final_detection']['candidates'])
        result['outcome']='trajectory_completed_requires_physical_lift_confirmation'
    except (Exception,KeyboardInterrupt) as exc:
        result['outcome']=str(exc) or 'interrupted';print('STOPPED',result['outcome'],flush=True)
    finally:
        try:command('jaw_hold')
        except Exception:pass
        (trial/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)

PAGE='''<!doctype html><html lang="th"><meta charset="utf-8"><title>กล้องช่วยจับของ</title>
<style>body{background:#15231c;color:white;font:20px system-ui;margin:25px}canvas{max-width:95vw}a{color:#9bd8af}</style>
<h1>จัดพวงกุญแจเข้าจุดจับที่สอน</h1><p>กรอบเขียว = ตำแหน่งตัวพวงกุญแจที่ยกสำเร็จ · กรอบส้ม = ของที่เห็นตอนนี้</p>
<p id="status">กำลังตรวจภาพ</p><canvas width="1280" height="720" id="view"></canvas>
<p>ใช้ตำแหน่งและแนวเดิมก่อน เวอร์ชันนี้ยังไม่ย้ายจุดจับไปตามวัตถุทั่วตาราง</p>
<p>หลังจับจะยกกลับท่าตรวจภาพ ถ้าของยังอยู่ที่เดิมจะลองใหม่สูงสุด 3 ครั้ง ถ้าภาพบังจะหยุดรอตรวจ</p>
<p><a href="http://127.0.0.1:8767/">ปุ่มก้าม</a></p><script>
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d');
async function poll(){try{const r=await fetch('/analysis'),s=await r.json();if(!r.ok)throw Error(s.error);
const im=new Image();im.src='/snapshot.jpg?t='+Date.now();await im.decode();ctx.drawImage(im,0,0);
for(const [color,items] of [['#00ff80',[s.reference]],['#ffaa00',s.candidates]]){ctx.strokeStyle=color;ctx.lineWidth=4;for(const o of items)ctx.strokeRect(...o.box);}
document.querySelector('#status').textContent=s.ready?'ตำแหน่งพร้อมทดสอบ':s.reason+' · ห่าง '+Math.round(s.center_error_pixels||0)+' px';
}catch(e){document.querySelector('#status').textContent=e.message;}setTimeout(poll,500);}poll();</script></html>'''
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            if self.path=='/':data=PAGE.encode();mime='text/html; charset=utf-8'
            elif self.path=='/analysis':data=json.dumps(assess(snapshot())).encode();mime='application/json'
            elif self.path.startswith('/snapshot.jpg'):
                with urlopen('http://127.0.0.1:8765/snapshot.jpg',timeout=3) as r:data=r.read()
                mime='image/jpeg'
            else:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        except Exception as exc:
            data=json.dumps({'error':str(exc)}).encode();self.send_response(503);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(data)
    def log_message(self,*args):pass
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--check',action='store_true');p.add_argument('--max-attempts',type=int,default=3);a=p.parse_args()
    if a.execute:execute(a.max_attempts)
    elif a.check:print(json.dumps(assess(snapshot()),indent=2))
    else:ThreadingHTTPServer(('127.0.0.1',8769),Handler).serve_forever()
