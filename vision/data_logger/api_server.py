"""
REST API to start / stop frame_logger and video_logger C++ binaries.

Endpoints:
  POST /start/frame         body: { device, output, fps, max_frames, show }
  POST /start/video         body: { device, output, fps, duration, show }
  POST /stop
  GET  /status
  GET  /files?type=frames|videos   list saved files
  GET  /download/<type>/<filename> download a file
  DELETE /files/<type>/<filename>  delete a file
"""

import os
import signal
import subprocess
from flask import Flask, request, jsonify, send_from_directory, send_file, abort

app = Flask(__name__, static_folder="ui")

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


@app.get("/")
def index():
    return send_from_directory("ui", "index.html")


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


LOGS = {
    "frames": os.path.join(BASE, "logs", "frames"),
    "videos": os.path.join(BASE, "logs", "videos"),
}


@app.get("/files")
def list_files():
    ftype = request.args.get("type", "videos")
    if ftype not in LOGS:
        return jsonify({"error": "type must be 'frames' or 'videos'"}), 400
    folder = LOGS[ftype]
    if not os.path.isdir(folder):
        return jsonify({"type": ftype, "files": []})
    files = sorted(os.listdir(folder), reverse=True)
    result = []
    for f in files:
        path = os.path.join(folder, f)
        result.append({
            "name": f,
            "size_mb": round(os.path.getsize(path) / 1024 / 1024, 2),
            "url": f"/download/{ftype}/{f}",
        })
    return jsonify({"type": ftype, "files": result})


@app.get("/download/<ftype>/<filename>")
def download_file(ftype, filename):
    if ftype not in LOGS:
        abort(404)
    folder = LOGS[ftype]
    safe = os.path.basename(filename)   # prevent path traversal
    path = os.path.join(folder, safe)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, as_attachment=True, download_name=safe)


@app.delete("/files/<ftype>/<filename>")
def delete_file(ftype, filename):
    if ftype not in LOGS:
        abort(404)
    folder = LOGS[ftype]
    safe = os.path.basename(filename)
    path = os.path.join(folder, safe)
    if not os.path.isfile(path):
        abort(404)
    os.remove(path)
    return jsonify({"deleted": safe})


# ── Network endpoints ──────────────────────────────────────────────────────────

@app.get("/network/status")
def network_status():
    try:
        devices = subprocess.check_output(
            ["nmcli", "-t", "-f", "DEVICE,TYPE,STATE,CONNECTION", "device", "status"],
            universal_newlines=True
        ).strip()
        ips = subprocess.check_output(["hostname", "-I"], universal_newlines=True).strip()
        return jsonify({"devices": devices, "ips": ips.split()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/network/wifi/list")
def wifi_list():
    try:
        # Force a fresh scan — required on Jetson/embedded; ignored gracefully if it fails
        subprocess.call(["nmcli", "device", "wifi", "rescan"],
                        stderr=subprocess.DEVNULL)
        out = subprocess.check_output(
            ["nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY,IN-USE", "device", "wifi", "list"],
            universal_newlines=True
        ).strip()
        networks = []
        for line in out.splitlines():
            parts = line.split(":")
            if len(parts) >= 4 and parts[0]:
                networks.append({
                    "ssid":     parts[0],
                    "signal":   parts[1],
                    "security": parts[2],
                    "in_use":   parts[3] == "*",
                })
        return jsonify({"networks": networks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/network/wifi/connect")
def wifi_connect():
    body = request.get_json(silent=True) or {}
    ssid     = body.get("ssid", "").strip()
    password = body.get("password", "").strip()
    if not ssid:
        return jsonify({"error": "ssid is required"}), 400
    try:
        cmd = ["nmcli", "device", "wifi", "connect", ssid]
        if password:
            cmd += ["password", password]
        out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.STDOUT)
        ips = subprocess.check_output(["hostname", "-I"], universal_newlines=True).strip()
        return jsonify({"connected": True, "ssid": ssid, "output": out.strip(), "ips": ips.split()})
    except subprocess.CalledProcessError as e:
        return jsonify({"connected": False, "error": e.output.strip()}), 500


@app.post("/system/shutdown")
def system_shutdown():
    try:
        subprocess.Popen(["shutdown", "-h", "now"])
        return jsonify({"shutdown": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)
