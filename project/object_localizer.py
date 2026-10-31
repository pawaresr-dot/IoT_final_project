"""Locate dark tabletop objects. No motor access; not a general AI detector.

Pixel locations are NOT robot coordinates. A detected object must never trigger
the old taught route without a separately measured camera-to-robot mapping.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from grid_vision import detect_cardboard_corners, validate_corners


def detect_objects(frame, corners):
    validate_corners(np.float32(corners), frame)
    region = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, np.int32(corners), 255)
    region = cv2.erode(region, np.ones((21, 21), np.uint8))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = np.uint8((gray < 75) & (region > 0)) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board_area = cv2.contourArea(np.float32(corners))
    border = region - cv2.erode(region, np.ones((9, 9), np.uint8))
    objects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 600 <= area <= board_area * .08:
            continue
        component = np.zeros_like(region)
        cv2.drawContours(component, [contour], -1, 255, -1)
        touches_boundary = bool(np.any((component > 0) & (border > 0)))
        rect = cv2.minAreaRect(contour)
        a, b = rect[1]
        if min(a, b) < 12 or max(a, b) / min(a, b) > 4:
            continue
        if area / (a * b) < .45:
            continue
        # The present board is clipped by the image top. Retain a compact,
        # elongated keychain candidate there, but explicitly forbid grasp use.
        if touches_boundary and not (max(a, b) / min(a, b) >= 1.6
                                     and area / (a * b) >= .65):
            continue
        moments = cv2.moments(contour)
        center = [moments['m10'] / moments['m00'], moments['m01'] / moments['m00']]
        objects.append({'center_px': center, 'area_px': area,
                        'box_px': cv2.boxPoints(rect).tolist(),
                        'detector': 'dark_object_contour',
                        'near_board_boundary': touches_boundary,
                        'grasp_verified': False})
    return sorted(objects, key=lambda obj: obj['area_px'], reverse=True)


def stable_target(detections, max_shift=8):
    if len(detections) < 3 or any(len(frame) != 1 for frame in detections):
        return None
    points = np.array([frame[0]['center_px'] for frame in detections])
    if np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)) > max_shift:
        return None
    return points.mean(axis=0).tolist()


def annotate(frame, objects):
    result = frame.copy()
    for obj in objects:
        cv2.polylines(result, [np.int32(obj['box_px'])], True, (0, 255, 0), 3)
        x, y = map(round, obj['center_px'])
        cv2.drawMarker(result, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)
        cv2.putText(result, f'object ({x}, {y})', (x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765/snapshot.jpg')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frames, detections, corners = [], [], None
    for _ in range(3):
        health = json.load(urllib.request.urlopen(args.url.rsplit('/', 1)[0] + '/health', timeout=5))
        if not health.get('live') or health.get('frame_age_seconds', 99) > 1:
            raise RuntimeError('Camera frame is not live')
        data = urllib.request.urlopen(args.url, timeout=5).read()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError('Invalid camera image')
        if corners is None:
            corners = detect_cardboard_corners(frame)
        frames.append(frame)
        detections.append(detect_objects(frame, corners))
    target = stable_target(detections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), annotate(frames[-1], detections[-1])):
        raise RuntimeError('Cannot save annotated image')
    print(json.dumps({'objects': detections[-1], 'stable_center_px': target,
                      'motion_ready': False,
                      'blocked_reason': 'camera_to_robot_mapping_missing',
                      'motor_commands_sent': 0}, indent=2))


if __name__ == '__main__':
    main()
