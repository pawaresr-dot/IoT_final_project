"""Record a bounded local-offset pickup; physical outcome requires operator review."""
import json
import argparse
import math
import threading
import time
from pathlib import Path
import cv2
from camera_arm_mapping import interpolate
from vision_guided_pick import snapshot, detect, status, move, jaw, command

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed', action='store_true', help='Require object near verified reference and replay its exact grasp pose')
    args = parser.parse_args()
    sequence = json.loads((ROOT/'vision_state/verified_adjusted_pick_sequence_20261031.json').read_text())
    mapping = json.loads((ROOT/'vision_state/local_mapping_20261031T202450/mapping.json').read_text())
    image = snapshot()
    reference = json.loads((ROOT/'vision_state/recorded_local_pick_20261031T221534/result.json').read_text()) if args.fixed else None
    options = []
    for obj in detect(image, bottom=710):
        if not 7000 <= obj['area'] <= 26000 or abs((obj['angle']-90+90)%180-90) > 15:
            continue
        if reference and (math.dist(obj['center'], reference['object']['center']) > 12
                          or abs((obj['angle']-reference['object']['angle']+90)%180-90) > 10
                          or not .85 <= obj['area']/reference['object']['area'] <= 1.15):
            continue
        try:
            mapped = interpolate(mapping['samples'], obj['center'])
        except ValueError:
            continue
        options.append((obj, mapped))
    if len(options) != 1:
        raise RuntimeError('No unique object in measured area; no motion sent')
    status()
    folder = ROOT/'vision_state'/('recorded_local_pick_'+time.strftime('%Y%m%dT%H%M%S'))
    folder.mkdir()
    writer = cv2.VideoWriter(str(folder/'trial.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (1280,720))
    if not writer.isOpened():
        raise RuntimeError('Video writer unavailable; no motion sent')
    writer.write(image)
    stop = threading.Event()
    result = {'object': options[0][0], 'operator_confirmed_success': None,
              'video_fps':5, 'frames':1, 'camera_errors':[], 'sequence_completed':False}
    def record():
        while not stop.wait(.2):
            try:
                writer.write(snapshot())
                result['frames'] += 1
            except Exception as exc:
                result['camera_errors'].append(str(exc))
    thread = threading.Thread(target=record)
    thread.start()
    anchor = interpolate(mapping['samples'], [426.022,585.122])
    target = {k:sequence['grasp_raw'][k]+options[0][1][k]-anchor[k] for k in anchor}
    if reference:
        target = dict(reference['target_raw'])
    result['mode'] = 'camera-gated verified fixed pose' if reference else 'experimental local mapping offset'
    result['target_raw'] = target
    high = dict(target, shoulder_lift=1719, elbow_flex=2820, wrist_flex=2420)
    print('RECORDING', folder, flush=True)
    try:
        move(high, timeout=45, precision=True)
        jaw('jaw_open')
        move(target, timeout=45, precision=True)
        closed = jaw('jaw_close')
        result['grip_raw'] = closed['readings']['gripper']['Present_Position']
        if result['grip_raw'] < 1980:
            jaw('jaw_open')
            raise RuntimeError('Empty grasp; stopped')
        move(high, timeout=45, precision=True)
        move(sequence['outside_high_raw'], timeout=35, precision=True)
        move(sequence['outside_release_raw'], timeout=35, precision=True)
        jaw('jaw_open')
        move(sequence['outside_high_raw'], timeout=35, precision=True)
        result['sequence_completed'] = True
    except Exception as exc:
        result['error'] = str(exc)
        raise
    finally:
        try:
            command('jaw_hold')
        finally:
            stop.set()
            thread.join()
            writer.release()
            (folder/'result.json').write_text(json.dumps(result, indent=2))
    print('COMPLETED', folder, flush=True)

if __name__ == '__main__':
    main()
