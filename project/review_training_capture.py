"""Generate a read-only capture audit and a time-labelled contact sheet."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

def jaw_motion_summary(rows, threshold=40):
    values=[r['action_after_raw'][5] for r in rows]
    low=high=values[0]
    closed=opened=False
    for value in values:
        closed=closed or high-value>=threshold
        opened=opened or value-low>=threshold
        low=min(low,value); high=max(high,value)
    return {'gripper_closing_observed':closed,'gripper_opening_observed':opened,
            'full_pick_sequence_observed':False,
            'sequence_note':'Jaw direction alone does not prove grasp, lift, or placement.'}

def review(root):
    rows = [json.loads(line) for line in (root / 'samples.jsonl').read_text().splitlines()]
    selected = sorted(set(round(i * (len(rows)-1)/23) for i in range(24)))
    sheet = Image.new('RGB', (1280, 6*205), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, idx in enumerate(selected):
        row = rows[idx]
        with Image.open(root / row['image']) as image:
            image.verify()
        with Image.open(root / row['image']) as image:
            thumb = image.convert('RGB'); thumb.thumbnail((320,180))
            x, y = n%4*320, n//4*205
            sheet.paste(thumb, (x,y))
            draw.text((x+4,y+182), f"{idx}: {row['image_received']-rows[0]['image_received']:.1f}s", fill='black')
    sheet.save(root / 'review_contact_sheet.jpg')
    times = [r['image_received'] for r in rows]
    gaps = [b-a for a,b in zip(times,times[1:])]
    prior=json.loads((root/'review.json').read_text()) if (root/'review.json').exists() else {}
    report = {'frames':len(rows), 'duration_seconds':times[-1]-times[0],
              'max_frame_gap_seconds':max(gaps),
              'max_camera_age_seconds':max(r['camera_age_at_check'] for r in rows),
              'state_ranges_raw':[[min(r['state_after_raw'][j] for r in rows),max(r['state_after_raw'][j] for r in rows)] for j in range(6)],
              'object_pickup_verified':prior.get('object_pickup_verified',False), 'training_ready':False,
              **jaw_motion_summary(rows)}
    for key in ('verification_source','operator_statement','verified_scope','visual_verification'):
        if key in prior: report[key]=prior[key]
    for row in rows:
        with Image.open(root / row['image']) as image: image.verify()
    (root/'review.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': review(Path(sys.argv[1]))
