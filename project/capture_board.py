"""Capture one board photo from a Raspberry Pi Camera Module or USB camera."""

import argparse
import time
from pathlib import Path


def capture_pi(output: Path, width: int, height: int, warmup: float) -> None:
    try:
        from picamera2 import Picamera2
    except ImportError as exc:
        raise RuntimeError(
            "Picamera2 is unavailable. On Raspberry Pi OS install python3-picamera2 "
            "and run this script with the system python3."
        ) from exc

    camera = Picamera2()
    try:
        config = camera.create_still_configuration(main={"size": (width, height)})
        camera.configure(config)
        camera.start()
        time.sleep(warmup)
        camera.capture_file(str(output))
    finally:
        camera.stop()
        camera.close()


def capture_usb(output: Path, device: int, width: int, height: int,
                warmup: float) -> None:
    import cv2

    camera = cv2.VideoCapture(device)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open USB camera index {device}")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        deadline = time.monotonic() + warmup
        frame = None
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a camera frame")
            if time.monotonic() >= deadline:
                break
        if not cv2.imwrite(str(output), frame):
            raise OSError(f"Cannot save photo: {output}")
    finally:
        camera.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pi", "usb"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0, help="USB camera index")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--warmup", type=float, default=2.0)
    args = parser.parse_args()
    if args.width < 320 or args.height < 240 or args.warmup < 0:
        parser.error("Width must be >=320, height >=240, and warmup >=0")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "pi":
        capture_pi(args.output, args.width, args.height, args.warmup)
    else:
        capture_usb(args.output, args.device, args.width, args.height, args.warmup)
    print(args.output)


if __name__ == "__main__":
    main()
