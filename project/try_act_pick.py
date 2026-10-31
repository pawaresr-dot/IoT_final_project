"""One bounded ACT trial through the existing monitored HTTP controller."""
import argparse
import io
import json
import time
from collections import deque
from pathlib import Path
from urllib.request import Request,urlopen
import numpy as np
import torch
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SO100Follower,SO100FollowerConfig
from lerobot.motors import MotorCalibration

NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
URL='http://127.0.0.1:8768'
def get(url):
    with urlopen(url,timeout=2) as response:return json.load(response)
def command(action,**kwargs):
    req=Request(URL+'/command',data=json.dumps({'action':action,**kwargs}).encode(),headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=2) as response:return json.load(response)
def validate(s):
    if s.get('error') or s.get('calibration_error') or not 0<=time.time()-s.get('last_sample',0)<.5:
        raise RuntimeError('Controller unavailable: '+str(s.get('error') or s.get('calibration_error')))
    if any(s['readings'][m]['Status'] or s['readings'][m]['Torque_Enable']!=1 for m in NAMES):
        raise RuntimeError('Servo alarm or torque change')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path);p.add_argument('dataset',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--seconds',type=float,default=120)
    a=p.parse_args()
    if not 0<a.seconds<=180:raise ValueError('Trial duration must be <=180 seconds')
    model=ACTPolicy.from_pretrained(str(a.checkpoint),local_files_only=True);model.eval();model.reset()
    pre,post=make_pre_post_processors(model.config,pretrained_path=str(a.checkpoint))
    math=SO100Follower(SO100FollowerConfig(port='/dev/DO_NOT_CONNECT',id='pawacha_so100_follower',use_degrees=True))
    math.bus.calibration={m:MotorCalibration(**v) for m,v in json.loads((a.dataset/'calibration_snapshot.json').read_text()).items()}
    ids={m:math.bus.motors[m].id for m in NAMES}
    root=Path(__file__).resolve().parent/'vision_state'/('ai_trial_'+time.strftime('%Y%m%dT%H%M%S'))
    root.mkdir();count=0;generation=None;stale_since=None;candidate=None;votes=0
    started=time.monotonic();outcome='duration_elapsed';warmed=False
    recent=deque()
    print(('EXECUTING' if a.execute else 'SHADOW')+': '+str(root),flush=True)
    try:
        with (root/'commands.jsonl').open('x',buffering=1) as log:
            while time.monotonic()-started<a.seconds:
                tick=time.monotonic();s=get(URL+'/status');validate(s)
                if s['phase'] in ('calibrating','jaw_moving'):
                    model.reset();time.sleep(.05);continue
                health=get('http://127.0.0.1:8765/health')
                if generation is None:generation=health['camera_generation']
                if health['camera_generation']!=generation:raise RuntimeError('Camera reconnected')
                if not health.get('live') or health['frame_age_seconds']>.25:
                    model.reset()
                    stale_since=tick if stale_since is None else stale_since
                    if tick-stale_since>3:raise RuntimeError('Camera stale for 3 seconds')
                    time.sleep(.1);continue
                stale_since=None
                with urlopen('http://127.0.0.1:8765/snapshot.jpg',timeout=2) as response:data=response.read()
                image=Image.open(io.BytesIO(data)).convert('RGB').resize((320,180),Image.Resampling.BILINEAR)
                raw={ids[m]:s['readings'][m]['Present_Position'] for m in NAMES}
                normalized=math.bus._normalize(raw)
                obs={'observation.state':torch.tensor([normalized[ids[m]] for m in NAMES],dtype=torch.float32),
                     'observation.images.overhead':torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255}
                with torch.inference_mode():pred=post(model.select_action(pre(obs))).detach().cpu().numpy().reshape(6)
                if not np.isfinite(pred).all():raise RuntimeError('Nonfinite model output')
                if not warmed:
                    warmed=True
                    model.reset()
                    started=time.monotonic()
                    print('Model warmup complete; refreshing image and state before commands',flush=True)
                    continue
                proposed=math.bus._unnormalize({ids[m]:float(v) for m,v in zip(NAMES,pred)})
                fresh=get(URL+'/status');validate(fresh)
                if time.monotonic()-tick>.5:raise RuntimeError('Inference/image timing exceeded 0.5 seconds')
                actual={m:fresh['readings'][m]['Present_Position'] for m in NAMES}
                targets={}
                for m in NAMES[:5]:
                    lo,hi=fresh['body_joint_limits'][m]
                    if not lo+3<=actual[m]<=hi-3:raise RuntimeError('Measured joint at limit: '+m)
                    if not lo+3<=proposed[ids[m]]<=hi-3:raise RuntimeError('Model target outside calibrated range: '+m)
                    targets[m]=max(actual[m]-12,min(actual[m]+12,proposed[ids[m]]))
                grip=max(math.bus.calibration['gripper'].range_min+3,min(2373,proposed[ids['gripper']]))
                grip_delta=grip-actual['gripper']
                intent='close' if grip_delta<=-16 else 'open' if grip_delta>=16 else 'hold'
                votes=votes+1 if intent==candidate else 1;candidate=intent
                sent=None
                if a.execute:
                    if votes>=3 and intent!='hold' and fresh['phase'] not in ('jaw_moving','calibrating'):
                        # Training actions are incremental servo goals. Match
                        # their direction without requiring an already-closed
                        # absolute goal or issuing a full force-close command.
                        command('jaw_hold')
                        deadline=time.monotonic()+2
                        while True:
                            held=get(URL+'/status');validate(held)
                            if held['phase']=='holding':break
                            if time.monotonic()>deadline:raise RuntimeError('Jaw hold acknowledgement timeout')
                            time.sleep(.03)
                        command('jog',motor='gripper',ticks=max(-11,min(11,grip_delta)))
                        sent='gripper_jog_'+intent;model.reset();votes=0
                    elif fresh['phase'] not in ('jaw_moving','calibrating'):command('teleop',targets=targets);sent='body'
                    else:model.reset()
                log.write(json.dumps({'time':time.time(),'prediction':pred.tolist(),'proposed_raw':[proposed[ids[m]] for m in NAMES],
                    'actual':actual,'bounded_body_target':targets,'sent':sent})+'\n')
                if count%10==0:(root/f'{count:05d}.jpg').write_bytes(data)
                count+=1
                recent.append((time.monotonic(),dict(actual)))
                while recent and time.monotonic()-recent[0][0]>10:recent.popleft()
                if a.execute and time.monotonic()-started>20 and recent and time.monotonic()-recent[0][0]>9:
                    spans=[max(row[m] for _,row in recent)-min(row[m] for _,row in recent) for m in NAMES]
                    if max(spans)<6:raise RuntimeError('Policy stalled: less than 6 ticks of motion over 9 seconds')
                if count%50==0:print('samples',count,'elapsed',round(time.monotonic()-started,1),flush=True)
                time.sleep(max(0,.1-(time.monotonic()-tick)))
    except KeyboardInterrupt:
        outcome='interrupted_by_operator'
        print('TRIAL STOPPED:',outcome,flush=True)
    except Exception as exc:
        outcome=str(exc);print('TRIAL STOPPED:',outcome,flush=True)
    finally:
        if a.execute:
            try:command('jaw_hold')
            except Exception:pass
        (root/'result.json').write_text(json.dumps({'execute':a.execute,'samples':count,'outcome':outcome,
            'checkpoint':str(a.checkpoint.resolve()),'autonomous_pickup_verified':False},indent=2))
        print('Saved',count,'samples;',outcome,flush=True)
if __name__=='__main__':main()
