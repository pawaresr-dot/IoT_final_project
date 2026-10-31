"""Export a local, unverified raw-tick episode; never upload or train."""
import argparse
import bisect
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def export(source, destination, calibration_path=None, image_size=None, allow_incomplete_eval=False, split_gap_seconds=None, active_window_seconds=None):
    manifest = json.loads((source/'manifest.json').read_text())
    split_path=source/'evaluation_split.json'
    split=json.loads(split_path.read_text()) if split_path.exists() else {}
    if allow_incomplete_eval and not (split.get('exclude_from_training') is True and split.get('intended_split')=='held_out_evaluation'):
        raise ValueError('Incomplete export requires an explicitly excluded evaluation capture')
    if not manifest['capture_complete'] and not allow_incomplete_eval:
        raise ValueError('Incomplete capture')
    if destination.exists():
        raise FileExistsError(destination)
    rows = [json.loads(x) for x in (source/'samples.jsonl').read_text().splitlines()]
    times = [r['image_received'] for r in rows]
    if any(b <= a for a,b in zip(times,times[1:])):
        raise ValueError('Non-monotonic image timestamps')
    with Image.open(source/rows[0]['image']) as im:
        width,height = im.size
    if image_size:
        width,height=image_size
        if width<32 or height<32: raise ValueError('Image dimensions must be >=32')
    motors = manifest['motor_order']
    calibration_bytes = calibration_path.read_bytes() if calibration_path else None
    robot = None
    if calibration_path:
        from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
        from lerobot.motors import MotorCalibration
        robot = SO100Follower(SO100FollowerConfig(port='/dev/DO_NOT_CONNECT',id=manifest['robot_id'],use_degrees=True))
        calibration = json.loads(calibration_bytes)
        robot.bus.calibration = {m:MotorCalibration(**calibration[m]) for m in motors}
    max_roundtrip_error = 0
    def coordinates(values):
        nonlocal max_roundtrip_error
        if robot is None:
            return np.array(values,dtype=np.float32)
        raw = {robot.bus.motors[m].id:int(v) for m,v in zip(motors,values)}
        normalized = robot.bus._normalize(raw)
        restored = robot.bus._unnormalize(normalized)
        error = max(abs(restored[k]-v) for k,v in raw.items())
        if error>1:
            raise ValueError('Normalization roundtrip exceeds one encoder tick')
        max_roundtrip_error=max(max_roundtrip_error,error)
        return np.array([normalized[robot.bus.motors[m].id] for m in motors],dtype=np.float32)
    # Validate every original state and issued action, not only resampled rows.
    for row in rows:
        coordinates(row['state_after_raw']); coordinates(row['action_after_raw'])
    names = [m+('.pos' if robot else '.pos_raw') for m in motors]
    features = {
        'observation.state':{'dtype':'float32','shape':(6,), 'names':names},
        'action':{'dtype':'float32','shape':(6,), 'names':names},
        'observation.images.overhead':{'dtype':'image','shape':(height,width,3),'names':['height','width','channels']},
    }
    dataset = LeRobotDataset.create('local/keychain_normalized' if robot else 'local/keychain_unverified_raw',fps=10,
        features=features,root=destination,robot_type='so100_follower' if robot else 'so100_raw_ticks',use_videos=False)
    if split_gap_seconds is not None and split_gap_seconds<=0:
        raise ValueError('Split gap must be positive')
    keep=np.ones(len(rows),dtype=bool)
    if active_window_seconds is not None:
        if active_window_seconds<=0:raise ValueError('Active window must be positive')
        states=np.asarray([r['state_after_raw'] for r in rows])
        goals=np.asarray([r['action_after_raw'] for r in rows])
        active=np.r_[False,(np.max(np.abs(np.diff(states,axis=0)),axis=1)>=4)|(np.max(np.abs(np.diff(goals,axis=0)),axis=1)>=4)]
        centers=np.asarray(times)[active]
        if not len(centers):raise ValueError('No active movement found')
        keep=np.asarray([np.min(np.abs(centers-t))<=active_window_seconds for t in times])
    segments=[];start=None
    for i in range(len(rows)):
        gap=i>0 and split_gap_seconds is not None and times[i]-times[i-1]>split_gap_seconds
        if start is not None and (not keep[i] or gap):
            segments.append((start,i));start=None
        if keep[i] and start is None:start=i
    if start is not None:segments.append((start,len(rows)))
    mapping=[]
    for episode,(start,end) in enumerate(segments):
        for n in range(int((times[end-1]-times[start])*10)+1):
            target=times[start]+n/10
            k=bisect.bisect_left(times,target,start,end)
            idx=min({max(start,k-1),min(k,end-1)},key=lambda j:abs(times[j]-target))
            row=rows[idx]
            with Image.open(source/row['image']) as im:
                rgb=im.convert('RGB')
                if image_size: rgb=rgb.resize((width,height),Image.Resampling.BILINEAR)
                frame=np.array(rgb)
            dataset.add_frame({'observation.state':coordinates(row['state_after_raw']),
                'action':coordinates(row['action_after_raw']),
                'observation.images.overhead':frame,'task':manifest['task']})
            mapping.append({'dataset_index':len(mapping),'episode_index':episode,'source_index':row['index'],'image_time_offset_seconds':times[idx]-target})
        dataset.save_episode()
    dataset.finalize()
    audit={'source':str(source.resolve()),'training_ready':False,
           'source_capture_complete':manifest['capture_complete'],
           'source_capture_error':manifest.get('error'),
           'intended_split':split.get('intended_split','unspecified'),
           'exclude_from_training':split.get('exclude_from_training',False),
           'incomplete_export_for_offline_evaluation':not manifest['capture_complete'],
           'object_pickup_verified':manifest.get('object_pickup_verified',False),
           'verification_source':manifest.get('verification_source'),
           'verified_scope':manifest.get('verified_scope'),
           'units':'body degrees; gripper calibrated 0..100' if robot else 'raw encoder ticks, NOT normalized LeRobot joint coordinates',
           'normalization':'installed LeRobot bus._normalize; use_degrees=True' if robot else None,
           'max_roundtrip_error_ticks':max_roundtrip_error if robot else None,
           'image_size_width_height':[width,height],
           'image_resize':'PIL bilinear' if image_size else None,
           'resampling':'nearest image sample at 10Hz; state and issued action kept from same source row',
           'split_gap_seconds':split_gap_seconds,'segment_count':len(segments),
           'active_window_seconds':active_window_seconds,'original_rows_retained':int(keep.sum()),
           'segments_are_independent_demonstrations':False,
           'frames':len(mapping),'sample_mapping':mapping}
    (destination/'capture_provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
    if calibration_bytes:
        (destination/'calibration_snapshot.json').write_bytes(calibration_bytes)
        (destination/'calibration_sha256.txt').write_text(hashlib.sha256(calibration_bytes).hexdigest()+'\n')
    if split: (destination/'evaluation_split.json').write_text(json.dumps(split,indent=2)+'\n')
    print(f'Exported {len(mapping)} frames locally; NOT training-ready',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path); parser.add_argument('destination',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--image-size',type=int,nargs=2,metavar=('WIDTH','HEIGHT'))
    parser.add_argument('--allow-incomplete-eval',action='store_true',help='Only for explicitly excluded offline evaluation captures; preserves interruption metadata')
    parser.add_argument('--split-gap-seconds',type=float,help='Start a new episode at image gaps above this duration; never resample across a gap')
    parser.add_argument('--active-window-seconds',type=float,help='Keep this much context around measured or commanded changes >=4 ticks; split removed idle spans')
    args=parser.parse_args(); export(args.source,args.destination,args.calibration,args.image_size,args.allow_incomplete_eval,args.split_gap_seconds,args.active_window_seconds)
