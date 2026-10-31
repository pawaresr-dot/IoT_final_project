"""Draw saved board coordinates on a fresh image; never change calibration."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from grid_vision import locate_board, validate_corners

def check(image_path, config_path, output_prefix):
    frame=cv2.imread(str(image_path))
    if frame is None: raise ValueError('Image unreadable')
    config=json.loads(config_path.read_text())
    corners=np.array(config['corners'],dtype=np.float32)
    overlay=frame.copy()
    cv2.polylines(overlay,[corners.astype(np.int32)],True,(0,0,255),3)
    for i,point in enumerate(corners):
        cv2.circle(overlay,tuple(point.astype(int)),7,(0,0,255),-1)
        cv2.putText(overlay,'OLD '+str(i+1),tuple(point.astype(int)+(8,20)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
    report={'source_image':str(image_path.resolve()),'saved_config':str(config_path.resolve()),
        'saved_corners':corners.tolist(),'calibration_updated':False,
        'alignment_status':'requires_review','automatic_candidate_accepted':False}
    try:
        candidate,method=locate_board(frame,'auto')
        validate_corners(candidate,frame)
        h,w=frame.shape[:2]
        clipped=bool(np.any(candidate[:,0]<=5) or np.any(candidate[:,1]<=5) or
                     np.any(candidate[:,0]>=w-6) or np.any(candidate[:,1]>=h-6))
        report.update(automatic_candidate=candidate.tolist(),method=method,candidate_touches_frame_edge=clipped)
        report['candidate_note']='Rejected: outline clipped by frame' if clipped else 'Needs visual confirmation: occlusion can produce a false outline'
    except ValueError as exc:
        report['candidate_error']=str(exc)
    cv2.rectangle(overlay,(0,0),(frame.shape[1],40),(0,0,0),-1)
    cv2.putText(overlay,'RED = OLD calibration, NOT current board boundary',(15,27),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
    cv2.imwrite(str(output_prefix.with_suffix('.jpg')),overlay)
    output_prefix.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path); parser.add_argument('config',type=Path); parser.add_argument('output_prefix',type=Path)
    args=parser.parse_args();check(args.image,args.config,args.output_prefix)
