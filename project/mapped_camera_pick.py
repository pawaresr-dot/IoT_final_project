"""Local four-point camera pickup with re-detection after a failed grasp."""
import argparse,json,time
from pathlib import Path
import cv2
import numpy as np
from camera_arm_mapping import interpolate,BODY
from vision_guided_pick import snapshot,detect,status,move,jaw,command,appearance_track
from pick_verification import verify_follow

ROOT=Path(__file__).resolve().parent
MAPPING=ROOT/'vision_state/local_mapping_20261031T202450/mapping.json'
def plan(image,mapping):
    candidates=[];outside=[]
    for obj in detect(image,bottom=710):
        # Exclude board-edge fragments and cropped silhouettes.
        x,y,w,h=obj['box']
        if x<=230 or y<=150 or x+w>=930 or y+h>=710:continue
        if not 7000<=obj['area']<=26000:continue
        try:target=interpolate(mapping['samples'],obj['center'])
        except ValueError:outside.append(obj);continue
        angle_error=abs((obj['angle']-90+90)%180-90)
        candidates.append({'object':obj,'target_raw':target,'orientation_ready':angle_error<=15})
    if len(candidates)!=1:
        return {'ready':False,'reason':'วางศูนย์กลางตัวพวงกุญแจภายในกรอบ 4 จุด ให้เห็นเพียงชิ้นเดียว',
                'candidates':candidates,'outside':outside}
    result=candidates[0];result['ready']=result['orientation_ready']
    result['reason']='คำนวณจุดจับแล้ว' if result['ready'] else 'จัด KU บน CS ล่าง ให้แนวยาวใกล้แนวตั้ง'
    return result
def raised(target):
    return dict(target,shoulder_lift=1719,elbow_flex=2820,wrist_flex=2420)
def inspect(target,pixel,folder):
    high=raised(target);move(high)
    image=snapshot();cv2.imwrite(str(folder/'raised.jpg'),image)
    visible=any(np.linalg.norm(np.array(o['center'])-pixel)<40 for o in detect(image,bottom=710))
    if visible:return verify_follow(True,[])
    a=appearance_track(image)
    if not a or a['score']<.70:return verify_follow(False,[a])
    # Visible matching keychain still inside the taught working patch is below
    # the raised inspection pose: treat as a failed pickup, not disappearance.
    if 350<=a['center'][0]<=500 and 450<=a['center'][1]<=600:
        return {'verdict':'not_picked','reason':'Object remains on taught work surface','track':a}
    probe=dict(high,shoulder_pan=high['shoulder_pan']+30)
    try:
        move(probe,precision=True);image=snapshot();b=appearance_track(image)
        cv2.imwrite(str(folder/'probe.jpg'),image)
    finally:move(high,precision=True)
    image=snapshot();c=appearance_track(image);cv2.imwrite(str(folder/'probe_return.jpg'),image)
    return verify_follow(False,[a,b,c])
def run(mapping,attempts):
    validation=mapping.get('center_validation',{})
    if not validation.get('operator_center_confirmed') or not validation.get('height_confirmed'):
        raise ValueError('Center position and working height must be checked first')
    if not 1<=attempts<=3:raise ValueError('Maximum three attempts')
    folder=ROOT/'vision_state'/('mapped_pick_'+time.strftime('%Y%m%dT%H%M%S'));folder.mkdir()
    result={'mapping':str(MAPPING),'mode':'local image-to-joint interpolation, taught lift/drop',
            'attempts':[],'autonomous_pickup_verified':False}
    try:
        for attempt in range(1,attempts+1):
            image=snapshot();candidate=plan(image,mapping)
            if not candidate['ready']:raise RuntimeError(candidate['reason'])
            # Independently re-detect before motion; do not reuse an old pixel.
            again=plan(snapshot(),mapping)
            if not again['ready'] or np.linalg.norm(np.array(candidate['object']['center'])-again['object']['center'])>8:
                raise RuntimeError('Object detection is unstable')
            candidate=again;target=candidate['target_raw'];pixel=np.array(candidate['object']['center'])
            record={'attempt':attempt,'plan':candidate};result['attempts'].append(record)
            print('Attempt',attempt,'pixel',pixel.tolist(),'target',target,flush=True)
            status();move(raised(target));jaw('jaw_open');move(target,precision=True)
            cv2.imwrite(str(folder/f'attempt_{attempt}_before_close.jpg'),snapshot())
            closed=jaw('jaw_close');record['gripper']=closed['readings']['gripper']
            checkfolder=folder/f'attempt_{attempt}';checkfolder.mkdir()
            verification=inspect(target,pixel,checkfolder);record['verification']=verification
            print('Verification',verification,flush=True)
            if verification['verdict']=='not_picked':
                jaw('jaw_open')
                if attempt==attempts:raise RuntimeError('Pickup failed after maximum attempts')
                print('Re-detecting moved object and recalculating grasp',flush=True)
                continue
            if verification['verdict']!='followed_visually':raise RuntimeError('Grasp visibility inconclusive; holding')
            # Transfer high first, then use the demonstrated outside drop pose.
            move(dict(raised(target),shoulder_pan=1303))
            move({'shoulder_pan':1303,'shoulder_lift':2123,'elbow_flex':2518,'wrist_flex':2288,'wrist_roll':2106})
            jaw('jaw_open');move({'shoulder_pan':1303,'shoulder_lift':1719,'elbow_flex':2820,'wrist_flex':2420,'wrist_roll':2106})
            cv2.imwrite(str(folder/'final.jpg'),snapshot())
            result['outcome']='visual_follow_and_release_completed_height_not_independently_verified';break
    except (Exception,KeyboardInterrupt) as exc:
        result['outcome']=str(exc) or 'interrupted';print('STOPPED',result['outcome'],flush=True)
    finally:
        try:command('jaw_hold')
        except Exception:pass
        (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--attempts',type=int,default=3);a=p.parse_args()
    mapping=json.loads(MAPPING.read_text())
    if a.execute:run(mapping,a.attempts)
    else:print(json.dumps(plan(snapshot(),mapping),indent=2))
