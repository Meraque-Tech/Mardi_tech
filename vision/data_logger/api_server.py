"""
REST API to start / stop frame_logger and video_logger C++ binaries.

Endpoints:
  POST /start/frame   body: { device, output, fps, max_frames, show }
  POST /start/video   body: { device, output, fps, duration, show }
  POST /stop
  GET  /status
"""

import os
import signal
import subprocess
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
BINS = {
    "frame": os.path.join(BASE, "build", "frame_logger"),
    "video": os.path.join(BASE, "build", "video_logger"),
}

_proc: subprocess.Popen = None
_logger_type: str = None


def _status():
    running = _proc is not None and _proc.poll() is None
    return {
        "running": running,
        "type": _logger_type if running else None,
        "pid": _proc.pid if running else None,
    }


@app.get("/status")
def status():
    return jsonify(_status())


@app.post("/start/frame")
def start_frame():
    global _proc, _logger_type
    if _proc and _proc.poll() is None:
        return jsonify({"error": "A logger is already running. POST /stop first."}), 409

    body = request.get_json(silent=True) or {}
    device = str(body.get("device", 0))
    output = str(body.get("output", "logs/frames"))
    fps    = str(body.get("fps", 5.0))
    show   = bool(body.get("show", False))
    max_f  = body.get("max_frames")

    cmd = [BINS["frame"], "--device", device, "--output", output, "--fps", fps]
    if max_f is not None:
        cmd += ["--max-frames", str(max_f)]
    if show:
        cmd.append("--show")

    _proc = subprocess.Popen(cmd)
    _logger_type = "frame"
    return jsonify({"started": "frame_logger", "pid": _proc.pid, "cmd": cmd})


@app.post("/start/video")
def start_video():
    global _proc, _logger_type
    if _proc and _proc.poll() is None:
        return jsonify({"error": "A logger is already running. POST /stop first."}), 409

    body = request.get_json(silent=True) or {}
    device   = str(body.get("device", 2))
    output   = str(body.get("output", "logs/videos"))
    fps      = str(body.get("fps", 30.0))
    show     = bool(body.get("show", False))
    duration = body.get("duration")

    cmd = [BINS["video"], "--device", device, "--output", output, "--fps", fps]
    if duration is not None:
        cmd += ["--duration", str(duration)]
    if show:
        cmd.append("--show")

    _proc = subprocess.Popen(cmd)
    _logger_type = "video"
    return jsonify({"started": "video_logger", "pid": _proc.pid, "cmd": cmd})


@app.post("/stop")
def stop():
    global _proc, _logger_type
    if _proc is None or _proc.poll() is not None:
        _proc = None
        _logger_type = None
        return jsonify({"stopped": False, "reason": "No logger running."})

    _proc.send_signal(signal.SIGINT)
    _proc.wait()
    pid = _proc.pid
    _proc = None
    _logger_type = None
    return jsonify({"stopped": True, "pid": pid})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
