"""Timestamped capture staging for LeRobot conversion; never controls motors.

Not yet a validated training dataset: gripper demonstrations and action/image
alignment checks must be completed before conversion and model training.
"""
import argparse
import json
import os
import shutil
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import uuid

MOTORS=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
stopping=False


def validate_status(status, now):
    if status.get('phase') not in ('holding','calibrating','jaw_moving','teleoperating'):
        raise ValueError('Follower not in teleoperation hold: '+str(status.get('phase')))
    if status.get('error') or status.get('calibration_error'):
        raise ValueError('Follower reports an error')
    if not 0 <= now-status.get('last_sample',0) <= .5:
        raise ValueError('Stale follower telemetry')
    if any(status['readings'][m]['Status'] or status['readings'][m]['Torque_Enable']!=1 for m in MOTORS):
        raise ValueError('Follower alarm or torque change')
    if not status.get('command_raw') or any(m not in status['command_raw'] for m in MOTORS):
        raise ValueError('Missing issued action targets; do not substitute observed states')


def get_json(url):
    return json.load(urllib.request.urlopen(url,timeout=3))


def duration_deadline(seconds,now):
    if seconds<0:raise ValueError('Duration must be nonnegative')
    return now+seconds if seconds else None


def validate_input_heartbeat(event, now):
    if event.get('event') not in ('sample','jog_accepted','batch_accepted','batch_deferred'):
        raise ValueError('Teleoperation input is stopped or not ready: '+str(event.get('event')))
    if not 0 <= now-event.get('time',0) <= 1:
        raise ValueError('Teleoperation input heartbeat is stale')


def input_heartbeat(path):
    with path.open('rb') as source:
        source.seek(0,2);end=source.tell();source.seek(max(0,end-8192));data=source.read()
    lines=data.splitlines()
    if not data.endswith(b'\n'):lines=lines[:-1]
    if not lines:raise ValueError('No complete teleoperation heartbeat')
    return json.loads(lines[-1])


def main():
    global stopping
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=float,default=0,help='0 records until explicit stop; positive values set a duration')
    parser.add_argument('--teleop-log',type=Path,required=True,help='Live input log; capture stops if forwarding process exits or becomes stale')
    parser.add_argument('--manual-gripper-buttons',action='store_true',help='Gripper commands come from the follower control buttons, not the leader gripper')
    parser.add_argument('--camera-stale-grace',type=float,default=3.0,
                        help='Skip brief stale camera checks for this many consecutive seconds before stopping')
    parser.add_argument('--gripper-calibration',type=Path,
                        help='Include the verified teleoperation gripper mapping in capture metadata')
    args=parser.parse_args()
    if args.seconds<0:raise ValueError('Duration must be nonnegative')
    if args.camera_stale_grace<0:raise ValueError('Camera stale grace must be nonnegative')
    if args.manual_gripper_buttons and args.gripper_calibration:
        raise ValueError('Choose one gripper input source')
    validate_input_heartbeat(input_heartbeat(args.teleop_log),time.time())
    grip_config=None
    if args.gripper_calibration:
        grip_config=json.loads(args.gripper_calibration.read_text())
        if grip_config.get('operator_sequence')!='closed_then_open':
            raise ValueError('Gripper input mapping is not labeled')
        processes=subprocess.run(['pgrep','-fl','calibration_leader.py'],capture_output=True,text=True).stdout
        if '--gripper-calibration '+str(args.gripper_calibration) not in processes:
            raise ValueError('Matching gripper-enabled teleoperation process is not running')
    root=Path(__file__).resolve().parent/'training_captures'/(
        time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    frames=root/'frames';frames.mkdir(parents=True)
    manifest={'schema':'timestamped_robot_capture_v1','robot_id':'pawacha_so100_follower',
              'motor_order':MOTORS,'intended_fps':10,'started_at':time.time(),
              'gripper_input_enabled':bool(grip_config),'gripper_input_mapping':grip_config,
              'gripper_control_source':'manual_buttons' if args.manual_gripper_buttons else 'leader' if grip_config else 'not_configured',
              'teleoperation_log':str(args.teleop_log),
              'task':'pick up the keychain from the grid and release it outside the grid',
              'native_lerobot_dataset':False,
              'training_ready':False,'object_pickup_verified':False,
              'image_source':'http://127.0.0.1:8765/snapshot.jpg',
              'action_semantics':'issued servo Goal_Position targets, raw encoder units',
              'requires_before_training':['successful grasp demonstrations including gripper control',
                  'review image/action timestamp alignment','convert and validate LeRobotDataset'],
              'frames':0,'capture_complete':False}
    manifest['duration_limit_seconds']=args.seconds or None
    manifest['camera_stale_grace_seconds']=args.camera_stale_grace
    manifest['camera_stale_checks_skipped']=0
    def stop(*_):
        global stopping
        stopping=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    print('CAPTURING (staging only): '+str(root),flush=True)
    last_sequence=None
    camera_stale_since=None
    deadline=duration_deadline(args.seconds,time.monotonic())
    try:
        with (root/'samples.jsonl').open('x',buffering=1) as journal:
            while not stopping and (deadline is None or time.monotonic()<deadline):
                tick=time.monotonic()
                if shutil.disk_usage(root).free<1024**3:raise ValueError('Less than 1 GiB disk space remains')
                heartbeat=input_heartbeat(args.teleop_log)
                validate_input_heartbeat(heartbeat,time.time())
                health=get_json('http://127.0.0.1:8765/health')
                if not health.get('live') or health.get('frame_age_seconds',99)>.25:
                    camera_stale_since=camera_stale_since or time.monotonic()
                    manifest['camera_stale_checks_skipped']+=1
                    if time.monotonic()-camera_stale_since>args.camera_stale_grace:
                        raise ValueError('Camera unavailable or stale beyond grace period')
                    time.sleep(.05)
                    continue
                camera_stale_since=None
                sequence=(health['camera_generation'],health['frames'])
                if last_sequence and sequence[0]!=last_sequence[0]:
                    raise ValueError('Camera reconnected; start a new capture')
                if sequence==last_sequence:
                    time.sleep(.03);continue
                before=get_json('http://127.0.0.1:8768/status');validate_status(before,time.time())
                t0=time.time()
                image=urllib.request.urlopen(manifest['image_source'],timeout=3).read()
                t1=time.time()
                if not image.startswith(b'\xff\xd8') or not image.endswith(b'\xff\xd9'):
                    raise ValueError('Invalid JPEG')
                after=get_json('http://127.0.0.1:8768/status');validate_status(after,time.time())
                if time.monotonic()-tick>.5:
                    raise ValueError('Capture timing window exceeded 0.5 seconds')
                index=manifest['frames'];relative=f'frames/{index:06d}.jpg'
                with (root/relative).open('xb') as output:output.write(image)
                row={'index':index,'image':relative,'image_request_started':t0,'image_received':t1,
                     'teleoperation_heartbeat_timestamp':heartbeat['time'],
                     'camera_health_sequence':sequence,'camera_age_at_check':health['frame_age_seconds'],
                     'state_before_timestamp':before['last_sample'],'state_after_timestamp':after['last_sample'],
                     'state_before_raw':[before['readings'][m]['Present_Position'] for m in MOTORS],
                     'state_after_raw':[after['readings'][m]['Present_Position'] for m in MOTORS],
                     'action_before_raw':[before['command_raw'][m] for m in MOTORS],
                     'action_after_raw':[after['command_raw'][m] for m in MOTORS],
                     'action_changed_during_capture':before['command_raw']!=after['command_raw']}
                journal.write(json.dumps(row)+'\n')
                manifest['frames']+=1;last_sequence=sequence
                if manifest['frames']%10==0:os.fsync(journal.fileno())
                time.sleep(max(0,.1-(time.monotonic()-tick)))
            journal.flush();os.fsync(journal.fileno())
        manifest['capture_complete']=True
    except Exception as exc:
        manifest['error']=str(exc)
        print('CAPTURE STOPPED: '+str(exc),flush=True)
    finally:
        manifest['ended_at']=time.time()
        with (root/'manifest.json').open('x') as output:json.dump(manifest,output,indent=2)
        print('Saved frames: '+str(manifest['frames'])+'; training_ready=false',flush=True)


if __name__=='__main__':main()
