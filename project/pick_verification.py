"""Pure visual pickup decisions and bounded retry policy."""
import numpy as np

def choose_grasp_cycle(rows):
    goals=[r['action_after_raw'][5] for r in rows]
    # The correction demo contains an earlier close/open attempt. Select the
    # last closed interval followed by a full release, not the global minimum.
    intervals=[];start=None
    for i,g in enumerate(goals):
        if start is None and g<2200:start=i
        if start is not None and g>2330:
            intervals.append((start,i));start=None
    if not intervals:raise ValueError('No complete close/release cycle')
    closed,opened=intervals[-1]
    grasp=closed
    while grasp>0 and goals[grasp-1]<2330:grasp-=1
    release=opened
    while release>closed and goals[release-1]>goals[closed]+12:release-=1
    return grasp,closed,release,opened

def verify_follow(source_visible,tracks,minimum_score=.70):
    """A-B-A visual test: disappearance alone never counts as following."""
    if source_visible:return {'verdict':'not_picked','reason':'Object remains at source'}
    if len(tracks)!=3 or any(t is None or t['score']<minimum_score for t in tracks):
        return {'verdict':'unknown','reason':'Object occluded or appearance match is weak'}
    a,b,c=[np.asarray(t['center'],dtype=float) for t in tracks]
    displacement=float(np.linalg.norm(b-a));return_error=float(np.linalg.norm(c-a))
    if displacement<8 and return_error<=10:
        return {'verdict':'not_picked','reason':'Visible object stayed stationary during arm probe'}
    if displacement>=12 and return_error<=10:
        return {'verdict':'followed_visually','reason':'Object moved and returned with arm probe',
                'probe_displacement_pixels':displacement,'return_error_pixels':return_error,
                'height_verified':False}
    return {'verdict':'unknown','reason':'Object did not follow the arm probe consistently'}

def retry_allowed(verdict,attempt,max_attempts,aligned):
    return verdict=='not_picked' and attempt<max_attempts and aligned
