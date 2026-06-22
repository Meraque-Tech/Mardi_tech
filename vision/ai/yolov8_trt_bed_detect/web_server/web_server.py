#!/usr/bin/env python3
"""
YOLOv8 TRT Bed Detect — Flask Web API + WebSocket Server
"""

import datetime
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import urllib.request
from typing import Optional, List, Dict
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8, Float32
from std_srvs.srv import Trigger

from flask import Flask, jsonify, send_from_directory, request
from flask_sock import Sock

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR   = os.environ.get("SAVE_DIR",   "/saved_frames")
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8080"))
API_PORT   = int(os.environ.get("API_PORT",   "8090"))
STATIC_DIR = Path(__file__).parent / "static"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "counts":       {},
    "bed_status":   0,
    "conf":         0.0,
    "detecting":    False,
    "last_updated": None,
}
state_lock = threading.Lock()
ws_clients = []  # type: List
ws_lock    = threading.Lock()


def broadcast(payload: str):
    with ws_lock:
        dead = []
        for ws in ws_clients:
            try:
                ws.send(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)


# ── ROS 2 node ────────────────────────────────────────────────────────────────
class BridgeNode(Node):
    def __init__(self):
        super().__init__("web_bridge")
        self.create_subscription(String,  "/class_counts",         self._counts_cb, 10)
        self.create_subscription(UInt8,   "/bed_detection_status", self._status_cb, 10)
        self.create_subscription(Float32, "/conf",                  self._conf_cb,   10)
        self._start_cli = self.create_client(Trigger, "/bed_detection")

    def _counts_cb(self, msg):
        counts = {}
        for part in msg.data.strip().split():
            if ":" in part:
                k, v = part.split(":")
                counts[k.replace("class", "")] = int(v)
        with state_lock:
            state["counts"]       = counts
            state["last_updated"] = datetime.datetime.now().isoformat()
            snap = dict(state)
        broadcast(json.dumps({
            "type":    "counts",
            "counts":  snap["counts"],
            "bed":     snap["bed_status"],
            "conf":    snap["conf"],
            "time":    snap["last_updated"],
        }))

    def _status_cb(self, msg):
        with state_lock:
            state["bed_status"] = int(msg.data)

    def _conf_cb(self, msg):
        with state_lock:
            state["conf"] = round(float(msg.data), 4)

    def call_start(self):
        if not self._start_cli.wait_for_service(timeout_sec=2.0):
            return False, "service not available"
        future = self._start_cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result():
            with state_lock:
                state["detecting"] = True
            return True, future.result().message
        return False, "no response"


ros_node = None


def _ros_spin():
    global ros_node
    rclpy.init(args=None)
    ros_node = BridgeNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    rclpy.shutdown()


# ── Snapshot from MJPEG ───────────────────────────────────────────────────────
def _grab_frame_bytes():
    # type: () -> Optional[bytes]
    """
    Read the MJPEG stream until we find one complete JPEG frame.
    Returns raw JPEG bytes — no cv2 needed.
    """
    try:
        url = f"http://127.0.0.1:{MJPEG_PORT}/"
        with urllib.request.urlopen(url, timeout=3) as resp:
            buf = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buf += chunk
                # JPEG starts with FF D8 and ends with FF D9
                start = buf.find(b"\xff\xd8")
                end   = buf.find(b"\xff\xd9")
                if start != -1 and end != -1 and end > start:
                    return buf[start:end + 2]
    except Exception:
        return None


# ── Flask app ─────────────────────────────────────────────────────────────────
app  = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
sock = Sock(app)


@app.route("/")
def dashboard():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/saved/<path:filename>")
def saved_file(filename):
    return send_from_directory(SAVE_DIR, filename)


@sock.route("/ws")
def websocket(ws):
    with ws_lock:
        ws_clients.append(ws)
    with state_lock:
        snap = dict(state)
    try:
        ws.send(json.dumps({
            "type":    "counts",
            "counts":  snap["counts"],
            "bed":     snap["bed_status"],
            "conf":    snap["conf"],
            "time":    snap["last_updated"],
        }))
        while True:
            ws.receive(timeout=30)   # keep alive
    except Exception:
        pass
    finally:
        with ws_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)


@app.route("/api/counts")
def get_counts():
    with state_lock:
        return jsonify(state["counts"])


@app.route("/api/status")
def get_status():
    with state_lock:
        return jsonify({
            "detecting":    state["detecting"],
            "bed_status":   state["bed_status"],
            "conf":         state["conf"],
            "last_updated": state["last_updated"],
            "mjpeg_url":    f"http://{{host}}:{MJPEG_PORT}/",
        })


@app.route("/api/start", methods=["POST"])
def start_detection():
    if ros_node is None:
        return jsonify({"success": False, "message": "ROS node not ready"}), 503
    ok, msg = ros_node.call_start()
    if ok:
        return jsonify({"success": True, "message": msg})
    return jsonify({"success": False, "message": msg}), 500


@app.route("/api/stop", methods=["POST"])
def stop_detection():
    with state_lock:
        state["detecting"] = False
    return jsonify({"success": True, "message": "detection stopped"})


@app.route("/api/save", methods=["POST"])
def save_frame():
    jpeg = _grab_frame_bytes()
    if jpeg is None:
        return jsonify({"success": False, "message": "could not grab frame"}), 500
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{ts}.jpg"
    path     = os.path.join(SAVE_DIR, filename)
    with open(path, "wb") as f:
        f.write(jpeg)
    return jsonify({"success": True, "filename": filename})


@app.route("/api/set_track", methods=["POST"])
def set_track():
    data    = request.get_json(force=True)
    enabled = str(data.get("enabled", False)).lower()
    subprocess.Popen(["ros2", "param", "set", "/yolov8_trt", "is_track", enabled])
    return jsonify({"success": True, "is_track": enabled})


@app.route("/api/reset_tracker", methods=["POST"])
def reset_tracker():
    broadcast(json.dumps({"type": "tracker_reset"}))
    return jsonify({"success": True})


@app.route("/api/images")
def list_images():
    files = sorted(Path(SAVE_DIR).glob("*.jpg"), reverse=True)
    return jsonify([{
        "filename": f.name,
        "url":      f"/saved/{f.name}",
        "size":     f.stat().st_size,
        "time":     datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
    } for f in files])


@app.route("/api/images/<filename>", methods=["DELETE"])
def delete_image(filename):
    path = Path(SAVE_DIR) / filename
    if not path.exists():
        return jsonify({"success": False}), 404
    path.unlink()
    return jsonify({"success": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=_ros_spin, daemon=True).start()
    time.sleep(1.0)
    app.run(host="0.0.0.0", port=API_PORT, threaded=True)
