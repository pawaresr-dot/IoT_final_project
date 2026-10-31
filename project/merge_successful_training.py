"""Merge the four operator-confirmed demonstrations locally."""
import json
import shutil
import sys
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

root=Path(__file__).resolve().parent
names=['20261031T112940_normalized_320','20261031T180447_segmented_320',
       '20261031T181529_segmented_320','20261031T182139_segmented_320']
paths=[root/'training_datasets'/n for n in names]
destination=root/'training_datasets'/'four_successes_20261031_320'
if '--active' in sys.argv or '--correction' in sys.argv:
    names=[n+'_active320' for n in ['20261031T112940_697ea07c','20261031T180447_81f2160b','20261031T181529_f98af656','20261031T182139_a0a94a9c']]
    paths=[root/'training_datasets'/n for n in names]
    destination=root/'training_datasets'/'four_successes_active_20261031_320'
if '--correction' in sys.argv:
    paths.append(root/'training_datasets'/'20261031T190513_a9654b10_active320')
    destination=root/'training_datasets'/'five_successes_active_20261031_320'
if destination.exists():raise FileExistsError(destination)
calibration=(paths[0]/'calibration_snapshot.json').read_bytes()
for p in paths:
    if (p/'calibration_snapshot.json').read_bytes()!=calibration:raise ValueError('Calibration mismatch')
    if not json.loads((p/'capture_provenance.json').read_text())['object_pickup_verified']:raise ValueError('Unconfirmed capture')
datasets=[LeRobotDataset('local/keychain_normalized',root=p) for p in paths]
merged=merge_datasets(datasets,'local/four_successes',destination)
shutil.copy2(paths[0]/'calibration_snapshot.json',destination/'calibration_snapshot.json')
(destination/'capture_provenance.json').write_text(json.dumps({
    'sources':[str(p) for p in paths], 'independent_demonstrations':len(paths),
    'image_resize':'PIL bilinear','source_capture_complete':True,
    'object_pickup_verified':True,'verification_source':'operator_confirmation_in_chat',
    'training_ready':False,'frames':merged.num_frames,
    'segments':merged.num_episodes,'segments_are_independent_demonstrations':False},indent=2)+'\n')
print('Merged',merged.num_frames,'frames from',len(paths),'demonstrations into',merged.num_episodes,'segments',flush=True)
