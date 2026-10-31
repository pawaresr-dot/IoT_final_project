"""Show the Raspberry Pi camera locally through SSH; no recordings or public port."""

import json
import selectors
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from live_board_vision import BoardVision

HOST = "pawacha@iot.local"
PORT = 8765
condition = threading.Condition()
latest = None
frame_count = 0
last_frame_time = 0.0
camera_error = None
camera_generation = 0
camera_stop = threading.Event()
ROOT = Path(__file__).resolve().parent
vision = BoardVision(ROOT / "vision_state")


def current_frame():
    with condition:
        if camera_error or latest is None or time.monotonic() - last_frame_time > 2:
            raise ValueError("ภาพกล้องไม่พร้อมหรือค้างอยู่ กรุณาตรวจการเชื่อมต่อ Pi")
        jpeg, sequence = latest, frame_count
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("อ่านภาพจากกล้องไม่ได้")
    return jpeg, frame, sequence


def receive_frames(process):
    global latest, frame_count, last_frame_time, camera_error
    buffer = bytearray()
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    last_received = time.monotonic()
    received_frame = False
    try:
        while not camera_stop.is_set():
            if not selector.select(timeout=0.5):
                limit = 6 if received_frame else 20
                if time.monotonic() - last_received > limit:
                    raise TimeoutError(f"No fresh camera frame for {limit} seconds; reconnecting to Pi.")
                continue
            chunk = process.stdout.read1(65536)
            if not chunk:
                raise ConnectionError("Camera SSH connection ended; reconnecting to Pi.")
            buffer.extend(chunk)
            while True:
                start = buffer.find(b"\xff\xd8")
                if start < 0:
                    buffer[:] = buffer[-1:]
                    break
                end = buffer.find(b"\xff\xd9", start + 2)
                if end < 0:
                    if start:
                        del buffer[:start]
                    break
                frame = bytes(buffer[start:end + 2])
                received_frame = True
                del buffer[:end + 2]
                with condition:
                    latest = frame
                    frame_count += 1
                    last_frame_time = time.monotonic()
                    last_received = last_frame_time
                    camera_error = None
                    condition.notify_all()
            if len(buffer) > 16 * 1024 * 1024:
                raise RuntimeError("Camera stream did not contain a complete JPEG frame")
    except Exception as exc:
        with condition:
            camera_error = str(exc)
    finally:
        selector.close()
        with condition:
            condition.notify_all()


def camera_manager():
    global latest, camera_error, camera_generation
    while not camera_stop.is_set():
        vision.clear_reference()
        with condition:
            latest = None
            camera_generation += 1
            camera_error = "กำลังเชื่อมต่อ Pi — Mac และ Pi ต้องอยู่เครือข่ายเดียวกัน"
            condition.notify_all()
        process = None
        try:
            process = subprocess.Popen([
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=3",
                "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=yes", HOST,
                "exec rpicam-vid --nopreview --timeout 0 --width 1280 --height 720 "
                "--framerate 10 --codec mjpeg --quality 40 --flush "
                "--autofocus-mode continuous --denoise cdn_off --output -",
            ], stdout=subprocess.PIPE)
            receive_frames(process)
        except Exception as exc:
            with condition:
                camera_error = str(exc)
                condition.notify_all()
        finally:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdout.close()
        camera_stop.wait(2)


class Handler(BaseHTTPRequestHandler):
    def reply(self, body, content_type="application/json", status=200):
        if isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        try:
            origin = self.headers.get("Origin")
            if origin and origin not in ("http://127.0.0.1:8765", "http://localhost:8765"):
                self.reply({"error": "Invalid origin"}, status=403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096 or self.headers.get_content_type() != "application/json":
                self.reply({"error": "Expected a small JSON request"}, status=400)
                return
            data = json.loads(self.rfile.read(length))
            _, frame, _ = current_frame()
            if self.path == "/api/corners":
                vision.set_corners(data["corners"], frame)
                self.reply({"ok": True})
            elif self.path == "/api/reference":
                if data.get("confirm_empty") is not True:
                    raise ValueError("Confirm that the board is empty and unobstructed first.")
                path = vision.capture_reference(frame)
                self.reply({"ok": True, "reference_path": path})
            else:
                self.reply({"error": "Not found"}, status=404)
        except (ValueError, TypeError, KeyError, cv2.error) as exc:
            self.reply({"error": str(exc)}, status=400)
        except OSError as exc:
            self.reply({"error": str(exc)}, status=500)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/api/board", "/board.jpg", "/snapshot.jpg"):
            try:
                jpeg, frame, sequence = current_frame()
                if path == "/snapshot.jpg":
                    self.reply(jpeg, "image/jpeg")
                    return
                state, preview = vision.analyze(frame, sequence)
                state = dict(state, camera_generation=camera_generation)
                self.reply(state if path == "/api/board" else preview,
                           "application/json" if path == "/api/board" else "image/jpeg")
            except (ValueError, cv2.error) as exc:
                self.reply({"error": str(exc)}, status=503)
            return
        if path == "/":
            body = (ROOT / "pi_camera_view.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/health":
            with condition:
                data = {
                    "source": HOST,
                    "frames": frame_count,
                    "frame_age_seconds": round(time.monotonic() - last_frame_time, 3) if latest else None,
                    "jpeg_bytes": len(latest) if latest else 0,
                    "error": camera_error,
                    "live": bool(latest is not None and not camera_error
                                 and time.monotonic() - last_frame_time <= 2),
                    "camera_generation": camera_generation,
                }
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            previous = -1
            generation = camera_generation
            try:
                while True:
                    with condition:
                        condition.wait_for(
                            lambda: frame_count != previous or camera_error or camera_stop.is_set(),
                            timeout=0.5,
                        )
                        if (camera_error or camera_stop.is_set()
                                or generation != camera_generation
                                or latest is None or time.monotonic() - last_frame_time > 2):
                            break
                        if frame_count == previous:
                            continue
                        frame, previous = latest, frame_count
                    if frame is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    manager = threading.Thread(target=camera_manager, daemon=True)
    manager.start()
    print(f"Pi camera preview: http://127.0.0.1:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        camera_stop.set()
        with condition:
            condition.notify_all()
        manager.join(timeout=5)
        server.server_close()
