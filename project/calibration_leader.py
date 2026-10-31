"""Passive leader -> existing single-owner controller, for local point capture.

Relative anchors are captured once; this never jumps to the leader's absolute
pose and never opens the follower serial port. Stop this process to stop new
jogs; the follower retains its independent monitored hold.
"""
import json
import argparse
import subprocess
import time
import urllib.request
import urllib.error
import uuid
from pathlib import Path

from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from camera_arm_mapping import BODY

URL = 'http://127.0.0.1:8768'
LEADER = '/dev/cu.usbmodem5B421342561'


def status():
    return json.load(urllib.request.urlopen(URL+'/status',timeout=3))


def motion_timer(started, moving, now):
    if not moving:return None,False
    started=now if started is None else started
    return started,now-started>35


def relative_body_targets(left,anchor,origin,scale=.25):
    if not .25 <= scale <= 1.:raise ValueError('Unsupported input scale')
    # Fine positioning: a large hand movement becomes a small follower movement.
    desired = {m:origin[m]+round((left[m]-anchor[m])*scale) for m in BODY}
    desired['wrist_roll'] = origin['wrist_roll']+round(((left['wrist_roll']-anchor['wrist_roll']+2048)%4096-2048)*scale)
    return desired


def next_jog(left, anchor, origin, actual, leader_range_limit=170, scale=.25):
    desired=relative_body_targets(left,anchor,origin,scale)
    exceeded = {m:{'leader':left[m], 'anchor':anchor[m],
                   'follower_delta':desired[m]-origin[m]}
                for m in BODY if leader_range_limit is not None and abs(desired[m]-origin[m]) > leader_range_limit}
    if exceeded:
        raise ValueError('Local calibration range exceeded: '+json.dumps(exceeded))
    motor = max(BODY,key=lambda m:abs(desired[m]-actual[m]))
    delta = desired[motor]-actual[motor]
    if abs(delta) <= 10:
        return None
    return motor, max(-11,min(11,delta))


def gripper_target(position, config):
    closed,opened=config['leader_closed_raw'],config['leader_open_raw']
    if abs(opened-closed)<100:raise ValueError('Leader gripper range too small')
    fraction=max(0.,min(1.,(position-closed)/(opened-closed)))
    low,high=config['follower_closed_raw'],config['follower_open_raw']
    if not 0 <= low < high <= 4095:raise ValueError('Invalid follower gripper range')
    return round(low+fraction*(high-low))


def checked_positions(read, previous=None, on_confirmation=None, motors=BODY):
    """Reject inconsistent jumps without sending a command or moving anchors."""
    result, rejected = {}, {}
    for m in motors:
        value=read('Present_Position',m)
        delta=0 if previous is None else value-previous[m]
        if m=='wrist_roll':delta=(delta+2048)%4096-2048
        if not 0 <= value <= 4095 or abs(delta)>128:
            confirmations=[read('Present_Position',m) for _ in range(3)]
            if on_confirmation:on_confirmation(m,value,confirmations)
            if not all(0 <= v <= 4095 for v in confirmations) or max(confirmations)-min(confirmations)>16:
                rejected[m]={'first':value,'confirmations':confirmations}
                continue
            value=sorted(confirmations)[1]
        result[m]=value
    return (None,rejected) if rejected else (result,{})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe-seconds',type=float,default=0,
                        help='Read-only observation; exits without forwarding any jogs')
    parser.add_argument('--no-leader-range-limit',action='store_true',
                        help='Disable only the leader displacement stop; follower limits and bounded jogs remain')
    parser.add_argument('--gripper-calibration',type=Path,help='User-labeled closed/open leader gripper measurements')
    parser.add_argument('--wait-gripper-match',action='store_true',help='Wait without motor commands until leader jaw matches the held follower jaw')
    parser.add_argument('--scale',type=float,choices=(.25,.5,1.),default=.25,
                        help='Relative body input scale; bounded follower jogs and joint limits unchanged')
    parser.add_argument('--stream-body-targets',action='store_true',help='Send all five current body targets together without waiting for single-joint jog completion')
    args=parser.parse_args()
    if args.stream_body_targets and args.gripper_calibration:
        raise ValueError('Streaming mode uses manual jaw buttons, not leader gripper input')
    grip_config=json.loads(args.gripper_calibration.read_text()) if args.gripper_calibration else None
    if grip_config:
        if grip_config.get('operator_sequence')!='closed_then_open':raise ValueError('Unconfirmed gripper direction')
        gripper_target(grip_config['leader_open_raw'],grip_config)
    input_motors=BODY+('gripper',) if grip_config else BODY
    if subprocess.run(['lsof','-t',LEADER],capture_output=True,text=True).stdout.strip():
        raise RuntimeError('Leader serial already in use')
    robot = SO100Leader(SO100LeaderConfig(port=LEADER,id='pawacha_so100_leader'))
    directory=Path(__file__).resolve().parent/'vision_state'
    directory.mkdir(exist_ok=True)
    log_path=directory/('leader_diagnostic_'+time.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]+'.jsonl')
    journal=log_path.open('x',buffering=1)
    def log(event,**values):
        journal.write(json.dumps({'time':time.time(),'event':event,**values})+'\n')
    print('Diagnostic log: '+str(log_path),flush=True)
    try:
        robot.bus.connect(handshake=False)
        left_anchor = origin = None
        anchor_samples=[]
        previous=None
        rejected_count=0
        last_forwarded_motor=None
        waiting_for_gripper=False
        started=time.monotonic()
        last_progress = time.monotonic()
        motion_started=None
        while True:
            def read(key,motor):
                try:return int(robot.bus.read(key,motor,normalize=False,num_retry=2))
                except Exception as exc:
                    log('read_error',motor=motor,register=key,error=str(exc))
                    raise
            for m in input_motors:
                if read('Torque_Enable',m) or read('Status',m):
                    raise RuntimeError('Leader must be passive without servo alarms: '+m)
            left,rejected=checked_positions(read,previous,
                lambda motor,first,confirmations:log('input_reread',motor=motor,first=first,confirmations=confirmations),motors=input_motors)
            if rejected:
                log('inconsistent_input_no_command',joints=rejected)
                rejected_count+=1
                if rejected_count>=5:raise RuntimeError('Repeated inconsistent leader input; forwarding stopped')
                time.sleep(.2)
                continue
            previous=left
            rejected_count=0
            s=status()
            if time.time()-s.get('last_sample',0)>2 or s.get('error') or s.get('calibration_error'):
                raise RuntimeError('Follower unavailable: '+str(s.get('error') or s.get('calibration_error')))
            if any(r['Status'] or r['Torque_Enable']!=1 for r in s['readings'].values()):
                raise RuntimeError('Follower alarm or torque change')
            if s['phase'] not in ('holding','ready_start','complete','released','cycle_complete','calibrating','jaw_moving','teleoperating'):
                raise RuntimeError('Unexpected follower operation: '+s['phase'])
            actual={m:s['readings'][m]['Present_Position'] for m in BODY}
            if left_anchor is None:
                anchor_samples.append(dict(left))
                anchor_samples=anchor_samples[-3:]
                if len(anchor_samples)<3 or any(max(row[m] for row in anchor_samples)-min(row[m] for row in anchor_samples)>16 for m in BODY):
                    log('waiting_for_stable_anchor',leader=left)
                    time.sleep(.2)
                    continue
                left_anchor,origin=dict(left),dict(actual)
                if grip_config and abs(gripper_target(left['gripper'],grip_config)-s['readings']['gripper']['Present_Position'])>35:
                    if args.wait_gripper_match:
                        log('waiting_for_gripper_match',leader_raw=left['gripper'],
                            mapped_target=gripper_target(left['gripper'],grip_config),
                            follower_raw=s['readings']['gripper']['Present_Position'])
                        if not waiting_for_gripper:
                            print('WAITING: align leader jaw opening with follower; no jogs forwarded yet.',flush=True)
                        waiting_for_gripper=True
                        left_anchor=origin=None
                        anchor_samples=[]
                        time.sleep(.2)
                        continue
                    raise RuntimeError('Open leader gripper to match follower before enabling; no startup jaw jump')
                log('anchors',leader_anchor=left_anchor,follower_origin=origin)
                log('gripper_input_policy',enabled=bool(grip_config),calibration=grip_config)
                log('leader_range_policy',enabled=not args.no_leader_range_limit)
                print('OBSERVING ONLY' if args.observe_seconds else
                      'FOLLOWING: relative anchors captured; scale '+str(args.scale)+'; gripper input '+('ENABLED' if grip_config else 'disabled')+'; leader displacement stop '+
                      ('OFF; calibrated follower joint limits remain; local radius='+str(s.get('calibration_local_limit_ticks','unknown')) if args.no_leader_range_limit else 'ON (+/-170 scaled ticks).'),flush=True)
            log('sample',leader=left,leader_delta={m:left[m]-left_anchor[m] for m in BODY},
                follower=actual,follower_phase=s['phase'])
            if args.observe_seconds:
                if time.monotonic()-started>=args.observe_seconds:
                    print('Read-only observation completed; no jogs forwarded.',flush=True)
                    return
                time.sleep(.2)
                continue
            motion_started,timed_out=motion_timer(motion_started,s['phase']=='calibrating',time.monotonic())
            if timed_out:raise RuntimeError('Follower jog timeout')
            if args.stream_body_targets:
                if s['phase'] not in ('calibrating','jaw_moving'):
                    targets=relative_body_targets(left,left_anchor,origin,args.scale)
                    if not args.no_leader_range_limit and any(abs(targets[m]-origin[m])>170 for m in BODY):
                        raise ValueError('Local input displacement limit exceeded')
                    req=urllib.request.Request(URL+'/command',data=json.dumps({'action':'teleop','targets':targets}).encode(),headers={'Content-Type':'application/json'})
                    try:
                        with urllib.request.urlopen(req,timeout=3) as response:json.load(response)
                        log('batch_accepted',targets=targets,follower_before=actual)
                    except urllib.error.HTTPError as exc:
                        current=status()
                        if exc.code!=409 or current.get('error') or current.get('calibration_error') or current['phase'] not in ('jaw_moving','calibrating'):
                            raise
                        log('batch_deferred',reason=current['phase'])
                time.sleep(.1)
                continue
            if s['phase'] not in ('calibrating','jaw_moving'):
                step=next_jog(left,left_anchor,origin,actual,
                              leader_range_limit=None if args.no_leader_range_limit else 170,scale=args.scale)
                if grip_config:
                    delta=gripper_target(left['gripper'],grip_config)-s['readings']['gripper']['Present_Position']
                    if abs(delta)>10 and (step is None or last_forwarded_motor!='gripper'):
                        step=('gripper',max(-11,min(11,delta)))
                if step:
                    req=urllib.request.Request(URL+'/command',data=json.dumps({'action':'jog','motor':step[0],'ticks':step[1]}).encode(),headers={'Content-Type':'application/json'})
                    with urllib.request.urlopen(req,timeout=3) as response:json.load(response)
                    log('jog_accepted',motor=step[0],ticks=step[1],follower_before=actual)
                    last_forwarded_motor=step[0]
                    last_progress=time.monotonic()
            time.sleep(.2)
    except Exception as exc:
        log('stopped',error=str(exc))
        raise
    finally:
        if robot.bus.is_connected:robot.bus.disconnect(disable_torque=False)
        journal.close()


if __name__=='__main__':main()
