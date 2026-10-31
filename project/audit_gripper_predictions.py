"""Test grip decisions on recorded grasp states and compare live stall state."""
import io,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

root=Path(__file__).resolve().parent
checkpoint=root/'training_outputs/act_five_active_20261031/checkpoints/006000/pretrained_model'
data=root/'training_datasets/five_successes_active_20261031_320'
output=root/'vision_state/gripper_prediction_audit_6000.json'
if output.exists():raise FileExistsError(output)
rows=[r for p in (data/'data').rglob('*.parquet') for r in pq.read_table(p).to_pylist()]
# Same controller thresholds expressed using saved calibration, not guessed units.
cal=json.loads((data/'calibration_snapshot.json').read_text())['gripper']
raw=lambda value:round(cal['range_min']+value/100*(cal['range_max']-cal['range_min']))
groups={label:[i for i,r in enumerate(rows) if predicate(raw(r['action'][5]))]
        for label,predicate in [('closed',lambda v:v<2200),('open',lambda v:v>2300)]}
model=ACTPolicy.from_pretrained(str(checkpoint),local_files_only=True);model.eval()
pre,post=make_pre_post_processors(model.config,pretrained_path=str(checkpoint))
result={'checkpoint':str(checkpoint),'split':'training observations, not held-out','robot_commands_sent':0,'groups':{}}
for label,indices in groups.items():
    chosen=[indices[i] for i in np.unique(np.linspace(0,len(indices)-1,32,dtype=int))]
    records=[]
    for i in chosen:
        r=rows[i];payload=r['observation.images.overhead']
        with Image.open(io.BytesIO(payload['bytes']) if payload['bytes'] else data/payload['path']) as image:
            pixels=np.array(image.convert('RGB'),copy=True)
        obs={'observation.state':torch.tensor(r['observation.state'],dtype=torch.float32),
             'observation.images.overhead':torch.from_numpy(pixels).permute(2,0,1).float()/255}
        with torch.inference_mode():pred=post(model.predict_action_chunk(pre(obs))[:,0]).detach().cpu().numpy().reshape(6)
        value=raw(float(pred[5]));decision='closed' if value<2200 else 'open' if value>2300 else 'middle'
        records.append({'index':i,'reference_raw':raw(r['action'][5]),'predicted_raw':value,'decision':decision})
    result['groups'][label]={'samples':len(records),'matching_decisions':sum(r['decision']==label for r in records),'records':records}
output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:{f:v for f,v in values.items() if f!='records'} for k,values in result['groups'].items()},indent=2),flush=True)
