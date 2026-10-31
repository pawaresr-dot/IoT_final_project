"""Offline input interventions and future-action audit; no robot access."""
import io
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'training_datasets/five_successes_active_20261031_320'
CHECKPOINT=ROOT/'training_outputs/act_five_active_20261031/checkpoints/001800/pretrained_model'
OUTPUT=ROOT/'vision_state/act_vision_diagnosis_20261031.json'
def main():
    global CHECKPOINT,OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=CHECKPOINT)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();CHECKPOINT=args.checkpoint;OUTPUT=args.output
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    model=ACTPolicy.from_pretrained(str(CHECKPOINT),local_files_only=True);model.eval();model.reset()
    pre,post=make_pre_post_processors(model.config,pretrained_path=str(CHECKPOINT))
    rows=sorted([r for f in (DATA/'data').rglob('*.parquet') for r in pq.read_table(f).to_pylist()],key=lambda r:r['index'])
    valid=[i for i in range(len(rows)-15) if rows[i]['episode_index']==rows[i+15]['episode_index']]
    selected=[valid[i] for i in np.unique(np.linspace(0,len(valid)-1,32,dtype=int))]
    def image(i):
        payload=rows[i]['observation.images.overhead']
        with Image.open(io.BytesIO(payload['bytes']) if payload['bytes'] else DATA/payload['path']) as im:
            return np.array(im.convert('RGB').resize((320,180),Image.Resampling.BILINEAR),copy=True)
    def predict(state,pixels):
        obs={'observation.state':torch.tensor(state,dtype=torch.float32),
             'observation.images.overhead':torch.from_numpy(pixels.copy()).permute(2,0,1).float()/255}
        with torch.inference_mode():
            chunk=model.predict_action_chunk(pre(obs))
            return np.stack([post(chunk[:,h]).detach().cpu().numpy().reshape(6) for h in range(16)])
    records=[]
    for n,i in enumerate(selected):
        state=np.asarray(rows[i]['observation.state']);pixels=image(i)
        original=predict(state,pixels)
        swapped=predict(state,image(selected[(n+len(selected)//2)%len(selected)]))
        blank=predict(state,np.full_like(pixels,127))
        shifted=state.copy();shifted[:5]+=5
        changed=predict(shifted,pixels)
        repeat=predict(state,pixels)
        references=np.asarray([rows[i+h]['action'] for h in range(16)])
        records.append({'index':int(i),'episode':int(rows[i]['episode_index']),
            'image_swap_change':np.abs(swapped-original).mean(axis=0).tolist(),
            'blank_image_change':np.abs(blank-original).mean(axis=0).tolist(),
            'state_plus_5_degrees_change':np.abs(changed-original).mean(axis=0).tolist(),
            'repeat_max_difference':float(np.max(np.abs(repeat-original))),
            'policy_future_error':np.abs(original-references).tolist(),
            'hold_future_error':np.abs(state[None,:]-references).tolist(),
            'reference_future_motion':np.abs(references[-1]-state).tolist()})
    summary={key:np.asarray([r[key] for r in records]).mean(axis=0).tolist()
             for key in ('image_swap_change','blank_image_change','state_plus_5_degrees_change')}
    for horizon in (0,8,15):
        summary['horizon_'+str(horizon)]={k:np.asarray([r[k][horizon] for r in records]).mean(axis=0).tolist()
            for k in ('policy_future_error','hold_future_error')}
    moving=[r for r in records if max(r['reference_future_motion'][:5])>3 or r['reference_future_motion'][5]>3]
    summary['moving_samples']=len(moving)
    if moving:
        summary['moving_horizon_15']={k:np.asarray([r[k][15] for r in moving]).mean(axis=0).tolist()
            for k in ('policy_future_error','hold_future_error')}
    report={'checkpoint':str(CHECKPOINT),'dataset':str(DATA),'split':'training data diagnostic; not held-out',
            'sample_count':len(records),'motor_order':['pan','lift','elbow','wrist','roll','gripper'],
            'units':['degrees']*5+['gripper_0_100'],
            'robot_commands_sent':0,'summary':summary,'records':records}
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
