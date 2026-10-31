"""Offline and live-shadow ACT checks. No motor writes or HTTP POSTs."""
import argparse
import io
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torchvision.transforms import v2
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.motors import MotorCalibration
from safetensors.numpy import load_file

def get_json(url):
    with urllib.request.urlopen(url,timeout=3) as response:
        return json.load(response)

def run(checkpoint,dataset,output,live_samples):
    if output.exists(): raise FileExistsError(output)
    policy=ACTPolicy.from_pretrained(str(checkpoint),local_files_only=True)
    policy.eval(); policy.reset()
    pre,post=make_pre_post_processors(policy.config,pretrained_path=str(checkpoint))
    resize=v2.Resize((180,320),antialias=True)
    provenance=json.loads((dataset/'capture_provenance.json').read_text())
    def predict(state,image):
        if provenance.get('image_resize')=='PIL bilinear':
            image=image.convert('RGB').resize((320,180),Image.Resampling.BILINEAR)
        pixels=torch.from_numpy(np.array(image.convert('RGB'),copy=True)).permute(2,0,1).float()/255
        obs={'observation.state':torch.tensor(state,dtype=torch.float32),
             'observation.images.overhead':resize(pixels)}
        with torch.inference_mode():
            chunk=policy.predict_action_chunk(pre(obs))
            action=post(chunk[:,0]).detach().cpu().numpy().reshape(6)
        if not np.isfinite(action).all(): raise ValueError('Nonfinite action')
        return action
    tables=[pq.read_table(p) for p in (dataset/'data').rglob('*.parquet')]
    rows=[r for t in tables for r in t.to_pylist()]
    selected=np.unique(np.linspace(0,len(rows)-1,24,dtype=int))
    predictions=[]; references=[]; states=[]
    for index in selected:
        row=rows[index]; payload=row['observation.images.overhead']
        with Image.open(io.BytesIO(payload['bytes']) if payload['bytes'] else dataset/payload['path']) as image:
            action=predict(row['observation.state'],image)
        predictions.append(action); references.append(row['action']); states.append(row['observation.state'])
    predictions=np.array(predictions); references=np.array(references); states=np.array(states)
    mae=np.abs(predictions-references).mean(axis=0)
    baseline=np.abs(states-references).mean(axis=0)
    processor_config=json.loads((checkpoint/'policy_postprocessor.json').read_text())
    normalizer_step=next(s for s in processor_config['steps'] if s['registry_name']=='unnormalizer_processor')
    stats=load_file(str(checkpoint/normalizer_step['state_file']))
    frozen_mean=stats['action.mean'].reshape(6)
    mean_baseline=np.abs(frozen_mean-references).mean(axis=0)
    names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
    split_path=dataset/'evaluation_split.json'
    split=json.loads(split_path.read_text()) if split_path.exists() else {}
    held_out=split.get('exclude_from_training') is True and split.get('intended_split')=='held_out_evaluation'
    if held_out and Path(split['frozen_checkpoint']).resolve()!=checkpoint.resolve():
        raise ValueError('Held-out evaluation must use its frozen checkpoint')
    report={'checkpoint':str(checkpoint.resolve()),'evaluation_split':'held-out human demonstration, interrupted capture' if held_out and not provenance.get('source_capture_complete',True) else 'held-out human demonstration' if held_out else 'TRAINING episode only, NOT held-out',
        'source_capture_complete':provenance.get('source_capture_complete',True),
        'constant_baseline_source':'frozen checkpoint training action.mean; no evaluation-derived fitting',
        'sample_count':len(selected),'units':['degrees']*5+['calibrated_0_100'],
        'motor_order':names,'mae_policy':mae.tolist(),'mae_hold_position_baseline':baseline.tolist(),
        'mae_constant_mean_baseline':mean_baseline.tolist(),
        'offline_predictions':predictions.tolist(),'source_indices':selected.tolist(),
        'live_shadow':[], 'robot_commands_sent':0,'autonomous_pickup_verified':False}
    # Construct calibration math only. Deliberately never call connect().
    robot=SO100Follower(SO100FollowerConfig(port='/dev/DO_NOT_CONNECT',id='pawacha_so100_follower',use_degrees=True))
    calibration=json.loads((dataset/'calibration_snapshot.json').read_text())
    robot.bus.calibration={m:MotorCalibration(**calibration[m]) for m in names}
    for _ in range(live_samples):
        health=get_json('http://127.0.0.1:8765/health')
        status=get_json('http://127.0.0.1:8768/status')
        if not health.get('live') or health['frame_age_seconds']>.25:
            report['live_shadow_stop']='Camera unavailable or stale'; break
        if time.time()-status.get('last_sample',0)>.5 or status.get('error'):
            report['live_shadow_stop']='Controller unavailable or stale'; break
        raw={robot.bus.motors[m].id:status['readings'][m]['Present_Position'] for m in names}
        normalized=robot.bus._normalize(raw)
        with urllib.request.urlopen('http://127.0.0.1:8765/snapshot.jpg',timeout=3) as response:
            image=Image.open(io.BytesIO(response.read())); image.load()
        action=predict([normalized[robot.bus.motors[m].id] for m in names],image)
        proposed=robot.bus._unnormalize({robot.bus.motors[m].id:float(v) for m,v in zip(names,action)})
        report['live_shadow'].append({'time':time.time(),'prediction':action.tolist(),
            'proposed_raw':[proposed[robot.bus.motors[m].id] for m in names],
            'delta_from_actual_raw':[proposed[robot.bus.motors[m].id]-raw[robot.bus.motors[m].id] for m in names]})
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('sample_count','mae_policy','mae_hold_position_baseline','mae_constant_mean_baseline','robot_commands_sent')},indent=2),flush=True)
    print('Shadow samples:',len(report['live_shadow']),'; report:',output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path); parser.add_argument('dataset',type=Path)
    parser.add_argument('output',type=Path); parser.add_argument('--live-samples',type=int,default=0)
    args=parser.parse_args(); run(args.checkpoint,args.dataset,args.output,args.live_samples)
