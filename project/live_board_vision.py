"""Board calibration and frame comparison; this module never controls motors."""

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from grid_vision import (
    WARP_SIZE, analyze_cells, find_dividers, make_preview, rectify,
    validate_corners,
)


class BoardVision:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.lock = threading.RLock()
        self.corners = None
        self.frame_size = None
        self.reference = None
        self.reference_path = None
        self.dividers = None
        self.cache_sequence = None
        self.cache = None
        self.preview = None
        config = self.directory / "board_config.json"
        if config.exists():
            data = json.loads(config.read_text())
            self.corners = np.asarray(data["corners"], np.float32)
            self.frame_size = tuple(data["frame_size"])
        # A new camera session always needs a fresh operator-confirmed reference.

    def _invalidate(self):
        self.cache_sequence = None
        self.cache = None
        self.preview = None

    def clear_reference(self):
        """A reconnected camera needs a fresh empty-board confirmation."""
        with self.lock:
            self.reference = self.dividers = self.reference_path = None
            self._invalidate()

    def _validate_frame(self, frame):
        if self.corners is None:
            raise ValueError("Select the four board corners first.")
        if self.frame_size != (frame.shape[1], frame.shape[0]):
            raise ValueError("Camera resolution changed; select board corners again.")
        validate_corners(self.corners, frame)

    def set_corners(self, points, frame):
        corners = np.asarray(points, np.float32)
        validate_corners(corners, frame)
        # TL, TR, BR, BL must describe the visible board orientation.
        if not (corners[0, 0] < corners[1, 0] and corners[3, 0] < corners[2, 0]
                and corners[0, 1] < corners[3, 1] and corners[1, 1] < corners[2, 1]):
            raise ValueError("Click corners in TL, TR, BR, BL order.")
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.corners = corners
            self.frame_size = (frame.shape[1], frame.shape[0])
            data = {"corners": corners.tolist(), "frame_size": self.frame_size}
            (self.directory / "board_config.json").write_text(
                json.dumps(data, indent=2) + "\n"
            )
            self.reference = self.dividers = self.reference_path = None
            self._invalidate()

    def capture_reference(self, frame):
        with self.lock:
            self._validate_frame(frame)
            board = rectify(frame, self.corners)
            dividers = find_dividers(board)
            self.directory.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            path = self.directory / ("empty_" + stamp + ".jpg")
            if not cv2.imwrite(str(path), frame):
                raise OSError("Could not save the empty-board image.")
            self.reference = board
            self.dividers = dividers
            self.reference_path = str(path)
            self._invalidate()
            return self.reference_path

    def analyze(self, frame, sequence):
        with self.lock:
            self._validate_frame(frame)
            if sequence == self.cache_sequence and self.cache is not None:
                return self.cache, self.preview
            board = rectify(frame, self.corners)
            if self.reference is None:
                cells = None
                dividers = {
                    "vertical": ((WARP_SIZE // 2, 0), (WARP_SIZE // 2, WARP_SIZE - 1)),
                    "horizontal": ((0, WARP_SIZE // 2), (WARP_SIZE - 1, WARP_SIZE // 2)),
                    "cross": (WARP_SIZE // 2, WARP_SIZE // 2),
                    "method": "pending_reference",
                }
                preview = board
            else:
                dividers = self.dividers
                cells = analyze_cells(self.reference, board, 35, 0.04, dividers)
                preview = make_preview(board, cells, dividers)
            destination = np.float32([
                [0, 0], [WARP_SIZE - 1, 0],
                [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1],
            ])
            inverse = cv2.getPerspectiveTransform(destination, self.corners)
            lines = []
            for key in ("vertical", "horizontal"):
                points = np.float32(dividers[key]).reshape(1, 2, 2)
                lines.append(cv2.perspectiveTransform(points, inverse)[0].tolist())
            success, encoded = cv2.imencode(".jpg", preview)
            if not success:
                raise OSError("Could not encode board preview.")
            self.cache = {
                "calibrated": True,
                "reference_ready": self.reference is not None,
                "reference_path": self.reference_path,
                "corners": self.corners.tolist(),
                "lines": lines,
                "cells": cells,
                "divider_method": dividers["method"],
                "frame_size": list(self.frame_size),
                "sequence": sequence,
            }
            self.preview = encoded.tobytes()
            self.cache_sequence = sequence
            return self.cache, self.preview
