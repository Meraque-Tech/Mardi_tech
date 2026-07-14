#!/usr/bin/env python3
"""YOLOv8 TRT bed detection dashboard, REST API, and ROS 2 bridge."""

import datetime
import json
import os
import sqlite3
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float32, Int32, String, UInt8
from std_srvs.srv import SetBool, Trigger

from flask import Flask, jsonify, request, send_from_directory
from flask_sock import Sock


# Configuration
SAVE_DIR = os.environ.get("SAVE_DIR", "/saved_frames")
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8080"))
API_PORT = int(os.environ.get("API_PORT", "8090"))
HISTORY_DB = os.environ.get("HISTORY_DB", os.path.join(SAVE_DIR, "count_history.db"))
AUTO_SAVE_INTERVAL = max(0.5, float(os.environ.get("AUTO_SAVE_INTERVAL", "0.5")))
STATIC_DIR = Path(__file__).parent / "static"

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_DB) or ".", exist_ok=True)


# Shared live state
state = {
    "counts": {},
    "bed_status": 0,
    "conf": 0.0,
    "detecting": False,
    "is_track": False,
    "auto_save": False,
    "last_updated": None,
    "camera_index": None,
    "infer_ms": 0.0,
}
state_lock = threading.Lock()
auto_save_wakeup = threading.Event()
ws_clients = []  # type: List
ws_lock = threading.Lock()


# Persistent count history
db_lock = threading.Lock()
storage_lock = threading.Lock()


def _db_connect():
    connection = sqlite3.connect(HISTORY_DB, timeout=5.0)
    connection.row_factory = sqlite3.Row
    return connection


def _init_db():
    with db_lock, _db_connect() as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS count_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recorded_at TEXT NOT NULL,
                counts_json TEXT NOT NULL,
                total INTEGER NOT NULL,
                bed_status INTEGER NOT NULL,
                confidence REAL NOT NULL,
                tracking INTEGER NOT NULL,
                frame_filename TEXT
            )
            """
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(count_history)")}
        if "frame_filename" not in columns:
            db.execute("ALTER TABLE count_history ADD COLUMN frame_filename TEXT")


def _store_history(snapshot, frame_filename=None):
    with db_lock, _db_connect() as db:
        cursor = db.execute(
            """
            INSERT INTO count_history
                (recorded_at, counts_json, total, bed_status, confidence, tracking, frame_filename)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["last_updated"],
                json.dumps(snapshot["counts"], separators=(",", ":"), sort_keys=True),
                sum(snapshot["counts"].values()),
                snapshot["bed_status"],
                snapshot["conf"],
                int(snapshot["is_track"]),
                frame_filename,
            ),
        )
        return cursor.lastrowid


def _history_rows(limit, offset):
    with db_lock, _db_connect() as db:
        total = db.execute("SELECT COUNT(*) FROM count_history").fetchone()[0]
        rows = db.execute(
            """
            SELECT id, recorded_at, counts_json, total, bed_status, confidence, tracking,
                   frame_filename
            FROM count_history ORDER BY id DESC LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    items = [
        {
            "id": row["id"],
            "time": row["recorded_at"],
            "counts": json.loads(row["counts_json"]),
            "total": row["total"],
            "bed": row["bed_status"],
            "conf": row["confidence"],
            "tracking": bool(row["tracking"]),
            "frame": row["frame_filename"],
        }
        for row in rows
    ]
    return total, items


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


def _snapshot_payload(message_type="snapshot"):
    with state_lock:
        snapshot = dict(state)
        snapshot["counts"] = dict(state["counts"])
    snapshot["type"] = message_type
    return snapshot


def broadcast_state(message_type="snapshot"):
    broadcast(json.dumps(_snapshot_payload(message_type)))


# ROS 2 bridge
class BridgeNode(Node):
    def __init__(self):
        super().__init__("web_bridge")
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(String, "/class_counts", self._counts_cb, 10)
        self.create_subscription(UInt8, "/bed_detection_status", self._status_cb, 10)
        self.create_subscription(UInt8, "/detection_active", self._active_cb, state_qos)
        self.create_subscription(UInt8, "/tracking_enabled", self._tracking_cb, state_qos)
        self.create_subscription(Float32, "/conf", self._conf_cb, 10)
        self.create_subscription(Int32, "/camera_index", self._camera_index_cb, state_qos)
        self.create_subscription(Float32, "/infer_ms", self._infer_ms_cb, 10)

        self._start_cli = self.create_client(Trigger, "/bed_detection")
        self._stop_cli = self.create_client(Trigger, "/bed_detection_stop")
        self._reset_cli = self.create_client(Trigger, "/reset_tracker")
        self._track_cli = self.create_client(SetBool, "/set_tracking")
        self._control_lock = threading.Lock()

    def _counts_cb(self, msg):
        counts = {}
        try:
            for part in msg.data.strip().split():
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                counts[key.replace("class", "")] = int(value)
        except (TypeError, ValueError):
            self.get_logger().warning("Ignored malformed /class_counts payload: %r" % msg.data)
            return

        with state_lock:
            state["counts"] = counts
            state["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            snapshot = dict(state)
            snapshot["counts"] = dict(counts)

        broadcast_state()

    def _status_cb(self, msg):
        with state_lock:
            state["bed_status"] = int(msg.data)

    def _conf_cb(self, msg):
        with state_lock:
            state["conf"] = round(float(msg.data), 4)

    def _camera_index_cb(self, msg):
        with state_lock:
            state["camera_index"] = int(msg.data)
        broadcast_state("status")

    def _infer_ms_cb(self, msg):
        with state_lock:
            state["infer_ms"] = round(float(msg.data), 2)

    def _active_cb(self, msg):
        with state_lock:
            state["detecting"] = bool(msg.data)
        auto_save_wakeup.set()
        broadcast_state("status")

    def _tracking_cb(self, msg):
        with state_lock:
            state["is_track"] = bool(msg.data)
        broadcast_state("status")

    def _call(self, client, request_message):
        # The node is already spinning in _ros_spin. Waiting on an Event here
        # avoids trying to add it to a second executor from a Flask thread.
        with self._control_lock:
            if not client.wait_for_service(timeout_sec=2.0):
                return False, "ROS service is not available"
            future = client.call_async(request_message)
            completed = threading.Event()
            future.add_done_callback(lambda _future: completed.set())
            if not completed.wait(timeout=4.0):
                return False, "ROS service timed out"
            try:
                result = future.result()
            except Exception as exc:
                return False, "ROS service failed: %s" % exc
            if result is None:
                return False, "ROS service returned no response"
            return bool(result.success), result.message

    def call_start(self):
        result = self._call(self._start_cli, Trigger.Request())
        if result[0]:
            with state_lock:
                state["detecting"] = True
            broadcast_state("status")
        return result

    def call_stop(self):
        result = self._call(self._stop_cli, Trigger.Request())
        if result[0]:
            with state_lock:
                state["detecting"] = False
            broadcast_state("status")
        return result

    def call_reset(self):
        return self._call(self._reset_cli, Trigger.Request())

    def call_set_tracking(self, enabled):
        message = SetBool.Request()
        message.data = bool(enabled)
        ok, detail = self._call(self._track_cli, message)
        if ok:
            with state_lock:
                state["is_track"] = bool(enabled)
            broadcast_state("status")
        return ok, detail


ros_node = None


def _ros_spin():
    global ros_node
    rclpy.init(args=None)
    ros_node = BridgeNode()
    try:
        rclpy.spin(ros_node)
    finally:
        ros_node.destroy_node()
        ros_node = None
        rclpy.shutdown()


# Snapshot extraction from the native MJPEG server
def _grab_frame_bytes():
    # type: () -> Optional[bytes]
    try:
        url = "http://127.0.0.1:%d/" % MJPEG_PORT
        with urllib.request.urlopen(url, timeout=3) as response:
            buffer = b""
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start != -1 and end != -1:
                    return buffer[start:end + 2]
    except Exception:
        return None
    return None


def _write_frame(jpeg):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = "frame_%s.jpg" % timestamp
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "wb") as output:
        output.write(jpeg)
    return filename, path


def _auto_save_loop():
    """Save one matching frame and count while detection and auto-save are active."""
    while True:
        auto_save_wakeup.wait(timeout=AUTO_SAVE_INTERVAL)
        auto_save_wakeup.clear()

        with state_lock:
            should_save = state["auto_save"] and state["detecting"]
        if not should_save:
            continue

        jpeg = _grab_frame_bytes()
        if jpeg is None:
            app.logger.warning("Auto-save skipped: MJPEG frame unavailable")
            continue

        # Re-check after the blocking frame read so Stop prevents a late save.
        with state_lock:
            if not state["auto_save"] or not state["detecting"] or state["last_updated"] is None:
                continue
            snapshot = dict(state)
            snapshot["counts"] = dict(state["counts"])

        snapshot["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        frame_path = None
        try:
            with storage_lock:
                with state_lock:
                    if not state["auto_save"] or not state["detecting"]:
                        continue
                frame_filename, frame_path = _write_frame(jpeg)
                _store_history(snapshot, frame_filename)
        except (OSError, sqlite3.Error) as exc:
            if frame_path and os.path.exists(frame_path):
                os.unlink(frame_path)
            app.logger.error("Auto-save failed: %s", exc)


# Flask app
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
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
    try:
        ws.send(json.dumps(_snapshot_payload()))
        while True:
            ws.receive(timeout=30)
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


@app.route("/api/history")
def get_history():
    try:
        limit = min(max(int(request.args.get("limit", 100)), 1), 5000)
        offset = max(int(request.args.get("offset", 0)), 0)
    except ValueError:
        return jsonify({"success": False, "message": "limit and offset must be integers"}), 400
    total, items = _history_rows(limit, offset)
    return jsonify({"total": total, "limit": limit, "offset": offset, "items": items})


@app.route("/api/save_count", methods=["POST"])
def save_count():
    with state_lock:
        if state["last_updated"] is None:
            return jsonify({"success": False, "message": "no live count has been received yet"}), 409
        snapshot = dict(state)
        snapshot["counts"] = dict(state["counts"])

    source_updated = snapshot["last_updated"]
    snapshot["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with storage_lock:
            record_id = _store_history(snapshot)
    except sqlite3.Error as exc:
        return jsonify({"success": False, "message": "could not save count: %s" % exc}), 500

    return jsonify({
        "success": True,
        "message": "current count saved",
        "id": record_id,
        "saved_at": snapshot["last_updated"],
        "source_updated": source_updated,
        "counts": snapshot["counts"],
        "total": sum(snapshot["counts"].values()),
    })


@app.route("/api/auto_save", methods=["POST"])
def set_auto_save():
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify({"success": False, "message": "enabled must be a boolean"}), 400
    with state_lock:
        state["auto_save"] = enabled
    auto_save_wakeup.set()
    broadcast_state("status")
    return jsonify({
        "success": True,
        "message": "automatic saving enabled" if enabled else "automatic saving disabled",
        "auto_save": enabled,
        "interval_seconds": AUTO_SAVE_INTERVAL,
    })


@app.route("/api/data", methods=["DELETE"])
def delete_all_data():
    with state_lock:
        state["auto_save"] = False
    auto_save_wakeup.set()

    deleted_images = 0
    try:
        with storage_lock:
            with db_lock, _db_connect() as db:
                deleted_records = db.execute("SELECT COUNT(*) FROM count_history").fetchone()[0]
                db.execute("DELETE FROM count_history")
                db.execute("DELETE FROM sqlite_sequence WHERE name = 'count_history'")
            for image_path in Path(SAVE_DIR).glob("*.jpg"):
                image_path.unlink()
                deleted_images += 1
    except (OSError, sqlite3.Error) as exc:
        return jsonify({"success": False, "message": "could not delete all data: %s" % exc}), 500

    broadcast_state("status")
    return jsonify({
        "success": True,
        "message": "all saved counts and images deleted",
        "deleted_records": deleted_records,
        "deleted_images": deleted_images,
        "auto_save": False,
    })


@app.route("/api/status")
def get_status():
    payload = _snapshot_payload()
    payload.pop("type", None)
    payload["mjpeg_port"] = MJPEG_PORT
    return jsonify(payload)


def _control_response(method_name):
    bridge = ros_node
    if bridge is None:
        return jsonify({"success": False, "message": "ROS bridge is not ready"}), 503
    ok, message = getattr(bridge, method_name)()
    return jsonify({"success": ok, "message": message}), (200 if ok else 503)


@app.route("/api/start", methods=["POST"])
def start_detection():
    return _control_response("call_start")


@app.route("/api/stop", methods=["POST"])
def stop_detection():
    return _control_response("call_stop")


@app.route("/api/reset_tracker", methods=["POST"])
def reset_tracker():
    return _control_response("call_reset")


@app.route("/api/set_track", methods=["POST"])
def set_track():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", False))
    if ros_node is None:
        return jsonify({"success": False, "message": "ROS bridge is not ready"}), 503
    ok, message = ros_node.call_set_tracking(enabled)
    return jsonify({"success": ok, "message": message, "is_track": enabled}), (200 if ok else 503)


@app.route("/api/save", methods=["POST"])
def save_frame():
    with storage_lock:
        jpeg = _grab_frame_bytes()
        if jpeg is None:
            return jsonify({"success": False, "message": "could not grab frame"}), 500
        filename, _path = _write_frame(jpeg)
    return jsonify({"success": True, "filename": filename})


@app.route("/api/images")
def list_images():
    files = sorted(Path(SAVE_DIR).glob("*.jpg"), reverse=True)
    return jsonify([
        {
            "filename": item.name,
            "url": "/saved/%s" % item.name,
            "size": item.stat().st_size,
            "time": datetime.datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
        }
        for item in files
    ])


@app.route("/api/images/<filename>", methods=["DELETE"])
def delete_image(filename):
    if Path(filename).name != filename:
        return jsonify({"success": False, "message": "invalid filename"}), 400
    path = Path(SAVE_DIR) / filename
    with storage_lock:
        if not path.exists():
            return jsonify({"success": False, "message": "image not found"}), 404
        path.unlink()
    return jsonify({"success": True})


if __name__ == "__main__":
    _init_db()
    threading.Thread(target=_ros_spin, daemon=True).start()
    threading.Thread(target=_auto_save_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=API_PORT, threaded=True, use_reloader=False)
