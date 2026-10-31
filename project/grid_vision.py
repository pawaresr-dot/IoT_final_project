"""Locate a 2x2 board and compare its cells with an empty-board photo.

Place ArUco 4x4_50 markers 0, 1, 2, 3 outside the board at its top-left,
top-right, bottom-right and bottom-left corners. The marker corner nearest the
board must touch the corresponding board corner. Without markers, the program
tries to detect a light cardboard board against a dark background. Alternatively,
pass a JSON file containing four image points in TL, TR, BR, BL order.

This program only processes images. It never sends commands to the robot.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CELL_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")
WARP_SIZE = 600
MARKER_INNER_CORNER = {0: 2, 1: 3, 2: 0, 3: 1}


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def load_manual_corners(path: Path) -> np.ndarray:
    """Load [top-left, top-right, bottom-right, bottom-left] image points."""
    points = np.asarray(json.loads(path.read_text()), dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Corner JSON must contain four finite [x, y] points")
    return points


def detect_board_corners(image: np.ndarray) -> np.ndarray:
    if not hasattr(cv2, "aruco"):
        raise ValueError("This OpenCV build has no ArUco support; use cardboard mode or --corners")
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    marker_corners, ids, _ = detector.detectMarkers(image)
    found = {} if ids is None else {
        int(marker_id): corners.reshape(4, 2)
        for marker_id, corners in zip(ids.flatten(), marker_corners)
    }
    missing = sorted(set(MARKER_INNER_CORNER) - set(found))
    if missing:
        raise ValueError(
            f"Board markers missing: {missing}. Need ArUco 4x4_50 IDs 0,1,2,3 "
            "or pass --corners with four image points."
        )
    return np.asarray(
        [found[marker_id][MARKER_INNER_CORNER[marker_id]] for marker_id in range(4)],
        dtype=np.float32,
    )


def detect_cardboard_corners(image: np.ndarray) -> np.ndarray:
    """Find the outer edge of one large light board against darker surroundings."""
    height, width = image.shape[:2]
    # Tan cardboard can sit on a light blue/gray surface. In that case a
    # brightness mask merges the board with its background, but Lab's yellow
    # channel still separates them.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    yellow_threshold, _ = cv2.threshold(
        lab[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if yellow_threshold >= 130:
        color_mask = np.uint8(
            (lab[:, :, 2] > max(138, yellow_threshold)) & (lab[:, :, 0] > 100)
        ) * 255
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)
        )
        color_contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if color_contours:
            color_board = max(color_contours, key=cv2.contourArea)
            color_area = cv2.contourArea(color_board) / (width * height)
            hull = cv2.convexHull(color_board)
            color_corners = cv2.approxPolyDP(
                hull, 0.03 * cv2.arcLength(hull, True), True
            )
            if 0.2 <= color_area <= 0.95 and len(color_corners) == 4:
                return order_corners(color_corners.reshape(4, 2).astype(np.float32))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_size = max(5, round(min(width, height) / 100) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((kernel_size, kernel_size), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Could not find a board outline")
    board = max(contours, key=cv2.contourArea)
    area_fraction = cv2.contourArea(board) / (width * height)
    if not 0.2 <= area_fraction <= 0.95:
        raise ValueError(
            f"Light board covers {area_fraction:.0%} of image; expected 20–95%. "
            "Try markers or --corners."
        )
    approximate = cv2.approxPolyDP(board, 0.025 * cv2.arcLength(board, True), True)
    if len(approximate) != 4:
        hull = cv2.convexHull(board)
        approximate = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)
    if len(approximate) != 4:
        raise ValueError("Board edge is not a quadrilateral; try markers or --corners")
    return order_corners(approximate.reshape(4, 2).astype(np.float32))


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral points in TL, TR, BR, BL order."""
    top = points[np.argsort(points[:, 1])[:2]]
    bottom = points[np.argsort(points[:, 1])[2:]]
    top_left, top_right = top[np.argsort(top[:, 0])]
    bottom_left, bottom_right = bottom[np.argsort(bottom[:, 0])]
    return np.asarray([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)


def locate_board(image: np.ndarray, mode: str) -> tuple[np.ndarray, str]:
    if mode == "markers":
        return detect_board_corners(image), "markers"
    if mode == "cardboard":
        return detect_cardboard_corners(image), "cardboard"
    try:
        return detect_board_corners(image), "markers"
    except ValueError:
        return detect_cardboard_corners(image), "cardboard"


def validate_corners(corners: np.ndarray, image: np.ndarray) -> None:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Invalid board corners")
    height, width = image.shape[:2]
    if np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
        raise ValueError("Board corners lie outside the image")
    # OpenCV's polygon area is zero for a collapsed or crossed quadrilateral.
    if cv2.contourArea(corners) < width * height * 0.01:
        raise ValueError("Board corners form a very small or invalid quadrilateral")
    if not cv2.isContourConvex(corners):
        raise ValueError("Board corners must form a convex quadrilateral")


def rectify(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    validate_corners(corners, image)
    destination = np.float32([
        [0, 0], [WARP_SIZE - 1, 0],
        [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1],
    ])
    transform = cv2.getPerspectiveTransform(corners, destination)
    return cv2.warpPerspective(image, transform, (WARP_SIZE, WARP_SIZE))


def find_dividers(empty_board: np.ndarray) -> dict:
    """Estimate the drawn center lines; use geometric midlines if unavailable."""
    gray = cv2.cvtColor(empty_board, cv2.COLOR_BGR2GRAY)
    center = gray[45:-45, 45:-45]
    ink_threshold = int(np.clip(np.median(center) - 55, 45, 145))
    dark = np.uint8(gray < ink_threshold) * 255
    dark[:30] = dark[-30:] = 0
    dark[:, :30] = dark[:, -30:] = 0
    detected = cv2.HoughLinesP(dark, 1, np.pi / 180, threshold=70,
                               minLineLength=150, maxLineGap=35)
    segments = [] if detected is None else [line[0] for line in detected]

    def fit_divider(orientation: str):
        candidates = []
        for x1, y1, x2, y2 in segments:
            dx, dy = x2 - x1, y2 - y1
            is_vertical = abs(dy) > 3 * abs(dx)
            is_horizontal = abs(dx) > 3 * abs(dy)
            midpoint = (x1 + x2) / 2 if orientation == "vertical" else (y1 + y2) / 2
            if (0.3 * WARP_SIZE <= midpoint <= 0.7 * WARP_SIZE
                    and (is_vertical if orientation == "vertical" else is_horizontal)):
                candidates.append((x1, y1, x2, y2))
        if not candidates:
            return None
        longest = max(candidates, key=lambda line: (line[2] - line[0]) ** 2
                      + (line[3] - line[1]) ** 2)
        target = ((longest[0] + longest[2]) / 2 if orientation == "vertical"
                  else (longest[1] + longest[3]) / 2)
        nearby = [line for line in candidates if abs(
            ((line[0] + line[2]) / 2 if orientation == "vertical"
             else (line[1] + line[3]) / 2) - target) <= 25]
        points = np.float32([(x1, y1) for x1, y1, _, _ in nearby]
                            + [(x2, y2) for _, _, x2, y2 in nearby])
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if orientation == "vertical":
            if abs(vy) < 0.1:
                return None
            slope = float(vx / vy)
            return slope, float(x0 - slope * y0)  # x = slope*y + intercept
        if abs(vx) < 0.1:
            return None
        slope = float(vy / vx)
        return slope, float(y0 - slope * x0)  # y = slope*x + intercept

    vertical = fit_divider("vertical")
    horizontal = fit_divider("horizontal")
    if vertical is None or horizontal is None:
        # Thin pencil lines may be too faint for the binary Hough mask.
        # Their average darkness across the board is still measurable.
        smoothed = cv2.GaussianBlur(gray, (5, 5), 0)
        interior = smoothed[40:-40, 40:-40]

        def projection_min(axis: int) -> int | None:
            profile = interior.mean(axis=axis)
            start = int(0.3 * WARP_SIZE) - 40
            end = int(0.7 * WARP_SIZE) - 40
            index = start + int(np.argmin(profile[start:end]))
            local = profile[max(0, index - 30):min(len(profile), index + 31)]
            if np.median(local) - profile[index] < 2.5:
                return None
            return index + 40

        x = projection_min(axis=0)
        y = projection_min(axis=1)
        if x is not None and y is not None:
            return {
                "vertical": ((x, 0), (x, WARP_SIZE - 1)),
                "horizontal": ((0, y), (WARP_SIZE - 1, y)),
                "cross": (x, y),
                "method": "projection",
            }
        return {
            "vertical": ((WARP_SIZE // 2, 0), (WARP_SIZE // 2, WARP_SIZE - 1)),
            "horizontal": ((0, WARP_SIZE // 2), (WARP_SIZE - 1, WARP_SIZE // 2)),
            "cross": (WARP_SIZE // 2, WARP_SIZE // 2),
            "method": "midpoint",
        }
    vslope, vintercept = vertical
    hslope, hintercept = horizontal
    cross_x = (vslope * hintercept + vintercept) / (1 - vslope * hslope)
    cross_y = hslope * cross_x + hintercept
    x_top, x_bottom = vintercept, vslope * (WARP_SIZE - 1) + vintercept
    y_left, y_right = hintercept, hslope * (WARP_SIZE - 1) + hintercept
    values = (cross_x, cross_y, x_top, x_bottom, y_left, y_right)
    if not all(0.25 * WARP_SIZE < value < 0.75 * WARP_SIZE for value in values):
        raise ValueError("Detected divider lines are outside the center of the board")
    return {
        "vertical": ((round(x_top), 0), (round(x_bottom), WARP_SIZE - 1)),
        "horizontal": ((0, round(y_left)), (WARP_SIZE - 1, round(y_right))),
        "cross": (round(cross_x), round(cross_y)),
        "method": "ink",
    }


def cell_polygons(dividers: dict) -> dict:
    top, bottom = dividers["vertical"]
    left, right = dividers["horizontal"]
    cross = dividers["cross"]
    edge = WARP_SIZE - 1
    return {
        "top_left": np.int32([(0, 0), top, cross, left]),
        "top_right": np.int32([top, (edge, 0), right, cross]),
        "bottom_left": np.int32([left, cross, bottom, (0, edge)]),
        "bottom_right": np.int32([cross, right, (edge, edge), bottom]),
    }


def analyze_cells(empty_board: np.ndarray, current_board: np.ndarray,
                  pixel_threshold: int, occupied_fraction: float,
                  dividers: dict) -> dict:
    """Return changed-pixel fraction and occupancy for the four cells."""
    if empty_board.shape != current_board.shape:
        raise ValueError("Reference and current board must have equal dimensions")
    reference = cv2.GaussianBlur(cv2.cvtColor(empty_board, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    current = cv2.GaussianBlur(cv2.cvtColor(current_board, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    changed = cv2.absdiff(reference, current) >= pixel_threshold
    changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN,
                               np.ones((5, 5), np.uint8))
    result = {}
    for name, polygon in cell_polygons(dividers).items():
        mask = np.zeros((WARP_SIZE, WARP_SIZE), np.uint8)
        cv2.fillConvexPoly(mask, polygon, 255)
        # Exclude physical grid lines, board edges and small alignment errors.
        mask = cv2.erode(mask, np.ones((35, 35), np.uint8))
        fraction = float(changed[mask > 0].mean())
        result[name] = {
            "occupied": fraction >= occupied_fraction,
            "changed_fraction": round(fraction, 4),
        }
    return result


def make_preview(board: np.ndarray, cells: dict, dividers: dict) -> np.ndarray:
    preview = board.copy()
    cv2.line(preview, *dividers["vertical"], (255, 255, 255), 3)
    cv2.line(preview, *dividers["horizontal"], (255, 255, 255), 3)
    for name, polygon in cell_polygons(dividers).items():
        cell = cells[name]
        color = (0, 0, 255) if cell["occupied"] else (0, 200, 0)
        x, y, _, _ = cv2.boundingRect(polygon)
        cv2.putText(preview, name, (x + 18, y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        cv2.putText(preview, f'{cell["changed_fraction"]:.1%}',
                    (x + 18, y + 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return preview


def validate_move(cells: dict, source: str, destination: str) -> dict:
    if source == destination:
        raise ValueError("Source and destination must be different cells")
    if not cells[source]["occupied"]:
        raise ValueError(f"Source cell is empty: {source}")
    if cells[destination]["occupied"]:
        raise ValueError(f"Destination cell is occupied: {destination}")
    return {"source": source, "destination": destination}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True,
                        help="photo of the empty board from the fixed camera")
    parser.add_argument("--image", type=Path, required=True,
                        help="photo of the current board")
    parser.add_argument("--corners", type=Path,
                        help="JSON with board corners TL, TR, BR, BL; overrides automatic detection")
    parser.add_argument("--board-mode", choices=("auto", "markers", "cardboard"),
                        default="auto", help="auto tries ArUco markers, then a light board")
    parser.add_argument("--pixel-threshold", type=int, default=35)
    parser.add_argument("--occupied-fraction", type=float, default=0.04)
    parser.add_argument("--preview", type=Path, default=Path("grid_preview.png"))
    parser.add_argument("--from-cell", choices=CELL_NAMES)
    parser.add_argument("--to-cell", choices=CELL_NAMES)
    args = parser.parse_args()
    if not 1 <= args.pixel_threshold <= 255:
        parser.error("--pixel-threshold must be between 1 and 255")
    if not 0 < args.occupied_fraction < 1:
        parser.error("--occupied-fraction must be between 0 and 1")
    if bool(args.from_cell) != bool(args.to_cell):
        parser.error("--from-cell and --to-cell must be supplied together")

    empty_image = load_image(args.reference)
    current_image = load_image(args.image)
    if args.corners:
        if empty_image.shape != current_image.shape:
            raise ValueError("Manual corners require both images to have the same size and camera position")
        corners = load_manual_corners(args.corners)
        empty_corners = current_corners = corners
        board_detection = "manual"
    else:
        empty_corners, empty_method = locate_board(empty_image, args.board_mode)
        current_corners, current_method = locate_board(current_image, args.board_mode)
        board_detection = {"reference": empty_method, "current": current_method}

    empty_board = rectify(empty_image, empty_corners)
    current_board = rectify(current_image, current_corners)
    dividers = find_dividers(empty_board)
    cells = analyze_cells(empty_board, current_board,
                          args.pixel_threshold, args.occupied_fraction, dividers)
    try:
        move = validate_move(cells, args.from_cell, args.to_cell) if args.from_cell else None
    except ValueError as exc:
        parser.error(str(exc))
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.preview), make_preview(current_board, cells, dividers)):
        raise OSError(f"Cannot save preview: {args.preview}")
    result = {"board_detection": board_detection, "divider_detection": dividers["method"],
              "cells": cells,
              "preview": str(args.preview)}
    if move:
        result["move"] = move
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
