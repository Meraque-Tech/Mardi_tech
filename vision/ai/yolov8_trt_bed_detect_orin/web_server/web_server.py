#!/usr/bin/env python3
"""YOLOv8 TRT bed detection dashboard, REST API, and ROS 2 bridge."""

import csv
import datetime
import io
import json
import math
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32, Int32, String, UInt8
from std_srvs.srv import SetBool, Trigger

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_sock import Sock


# Configuration
SAVE_DIR = os.environ.get("SAVE_DIR", "/saved_frames")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8080"))
API_PORT = int(os.environ.get("API_PORT", "8090"))
HISTORY_DB = os.environ.get("HISTORY_DB", os.path.join(SAVE_DIR, "count_history.db"))
AUTO_SAVE_INTERVAL = max(0.5, float(os.environ.get("AUTO_SAVE_INTERVAL", "0.5")))
GNSS_FIX_TOPIC = os.environ.get("GNSS_FIX_TOPIC", "/receiver/fix")
GNSS_STALE_TIMEOUT = float(os.environ.get("GNSS_STALE_TIMEOUT", "3.0"))
FORWARD_TOPIC = os.environ.get("FORWARD_TOPIC", "/gnss/is_forward")
BACKWARD_TOPIC = os.environ.get("BACKWARD_TOPIC", "/gnss/is_backward")
DIRECTION_STALE_TIMEOUT = float(os.environ.get("DIRECTION_STALE_TIMEOUT", "3.0"))
STATIC_DIR = Path(__file__).parent / "static"

if not math.isfinite(GNSS_STALE_TIMEOUT) or GNSS_STALE_TIMEOUT <= 0:
    raise ValueError("GNSS_STALE_TIMEOUT must be a finite number greater than zero")
if not math.isfinite(DIRECTION_STALE_TIMEOUT) or DIRECTION_STALE_TIMEOUT <= 0:
    raise ValueError("DIRECTION_STALE_TIMEOUT must be a finite number greater than zero")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
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
    "latitude": None,
    "longitude": None,
    "gnss_valid": False,
    "gnss_received_at": None,
    "is_forward": None,
    "is_backward": None,
    "direction_received_at": None,
}
state_lock = threading.Lock()
gnss_last_monotonic = None
forward_last_monotonic = None
backward_last_monotonic = None
auto_save_wakeup = threading.Event()
ws_clients = []  # type: List
ws_lock = threading.Lock()


# Persistent count history
db_lock = threading.Lock()
storage_lock = threading.Lock()
frame_filename_counters = {}


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
                frame_filename TEXT,
                latitude REAL,
                longitude REAL,
                gnss_recorded_at TEXT
            )
            """
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(count_history)")}
        migrations = {
            "frame_filename": "TEXT",
            "latitude": "REAL",
            "longitude": "REAL",
            "gnss_recorded_at": "TEXT",
        }
        for column, column_type in migrations.items():
            if column not in columns:
                db.execute(
                    "ALTER TABLE count_history ADD COLUMN %s %s" % (column, column_type)
                )


def _store_history(snapshot, frame_filename=None):
    with db_lock, _db_connect() as db:
        cursor = db.execute(
            """
            INSERT INTO count_history
                (recorded_at, counts_json, total, bed_status, confidence, tracking,
                 frame_filename, latitude, longitude, gnss_recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot["last_updated"],
                json.dumps(snapshot["counts"], separators=(",", ":"), sort_keys=True),
                sum(snapshot["counts"].values()),
                snapshot["bed_status"],
                snapshot["conf"],
                int(snapshot["is_track"]),
                frame_filename,
                snapshot["latitude"],
                snapshot["longitude"],
                snapshot["gnss_received_at"],
            ),
        )
        return cursor.lastrowid


def _history_rows(limit, offset):
    with db_lock, _db_connect() as db:
        total = db.execute("SELECT COUNT(*) FROM count_history").fetchone()[0]
        rows = db.execute(
            """
            SELECT id, recorded_at, counts_json, total, bed_status, confidence, tracking,
                   frame_filename, latitude, longitude, gnss_recorded_at
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
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "gnss_time": row["gnss_recorded_at"],
        }
        for row in rows
    ]
    return total, items


def _image_history_rows():
    with db_lock, _db_connect() as db:
        rows = db.execute(
            """
            SELECT id, recorded_at, counts_json, total, bed_status, confidence, tracking,
                   frame_filename, latitude, longitude, gnss_recorded_at
            FROM count_history
            WHERE frame_filename IS NOT NULL
            ORDER BY id DESC
            """
        ).fetchall()

    history_by_filename = {}
    for row in rows:
        history_by_filename.setdefault(row["frame_filename"], row)
    return history_by_filename


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


def _state_snapshot():
    now_monotonic = time.monotonic()
    with state_lock:
        snapshot = dict(state)
        snapshot["counts"] = dict(state["counts"])

        gnss_is_fresh = (
            state["gnss_valid"]
            and gnss_last_monotonic is not None
            and now_monotonic - gnss_last_monotonic <= GNSS_STALE_TIMEOUT
        )
        direction_is_fresh = (
            forward_last_monotonic is not None
            and backward_last_monotonic is not None
            and now_monotonic - forward_last_monotonic <= DIRECTION_STALE_TIMEOUT
            and now_monotonic - backward_last_monotonic <= DIRECTION_STALE_TIMEOUT
        )

    if not gnss_is_fresh:
        snapshot["latitude"] = None
        snapshot["longitude"] = None
        snapshot["gnss_valid"] = False
        snapshot["gnss_received_at"] = None
    snapshot["direction_valid"] = direction_is_fresh
    if not direction_is_fresh:
        snapshot["is_forward"] = None
        snapshot["is_backward"] = None
        snapshot["direction_received_at"] = None
    return snapshot


def _direction_allows_auto_save(snapshot):
    return (
        bool(snapshot["direction_valid"])
        and snapshot["is_forward"] is True
        and snapshot["is_backward"] is False
    )


def _snapshot_payload(message_type="snapshot"):
    snapshot = _state_snapshot()
    snapshot["auto_save_direction_eligible"] = _direction_allows_auto_save(snapshot)
    snapshot["auto_save_active"] = (
        snapshot["auto_save"]
        and snapshot["detecting"]
        and snapshot["auto_save_direction_eligible"]
    )
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
        self.create_subscription(NavSatFix, GNSS_FIX_TOPIC, self._gnss_fix_cb, 10)
        self.create_subscription(Bool, FORWARD_TOPIC, self._forward_cb, 10)
        self.create_subscription(Bool, BACKWARD_TOPIC, self._backward_cb, 10)

        self._start_cli = self.create_client(Trigger, "/bed_detection")
        self._stop_cli = self.create_client(Trigger, "/bed_detection_stop")
        self._reset_cli = self.create_client(Trigger, "/reset_tracker")
        self._track_cli = self.create_client(SetBool, "/set_tracking")
        self._control_lock = threading.Lock()
        self._last_direction_broadcast = (None, None, False)
        self.get_logger().info(
            "Listening for GNSS fixes on %s (stale after %.1f s)"
            % (GNSS_FIX_TOPIC, GNSS_STALE_TIMEOUT)
        )
        self.get_logger().info(
            "Listening for direction flags on %s and %s (stale after %.1f s)"
            % (FORWARD_TOPIC, BACKWARD_TOPIC, DIRECTION_STALE_TIMEOUT)
        )

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

    def _gnss_fix_cb(self, msg):
        global gnss_last_monotonic

        latitude = float(msg.latitude)
        longitude = float(msg.longitude)
        valid = (
            msg.status.status >= 0
            and math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90.0 <= latitude <= 90.0
            and -180.0 <= longitude <= 180.0
        )

        with state_lock:
            if valid:
                state["latitude"] = latitude
                state["longitude"] = longitude
                state["gnss_valid"] = True
                state["gnss_received_at"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                gnss_last_monotonic = time.monotonic()
            else:
                state["latitude"] = None
                state["longitude"] = None
                state["gnss_valid"] = False
                state["gnss_received_at"] = None
                gnss_last_monotonic = None

    def _forward_cb(self, msg):
        self._direction_cb("is_forward", bool(msg.data))

    def _backward_cb(self, msg):
        self._direction_cb("is_backward", bool(msg.data))

    def _direction_cb(self, state_key, value):
        global forward_last_monotonic, backward_last_monotonic

        now_monotonic = time.monotonic()
        with state_lock:
            state[state_key] = value
            state["direction_received_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            if state_key == "is_forward":
                forward_last_monotonic = now_monotonic
            else:
                backward_last_monotonic = now_monotonic

        snapshot = _state_snapshot()
        direction_state = (
            snapshot["is_forward"],
            snapshot["is_backward"],
            snapshot["direction_valid"],
        )
        if direction_state != self._last_direction_broadcast:
            self._last_direction_broadcast = direction_state
            broadcast_state("status")

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
        clocks_ok, clocks_message = _run_jetson_clocks()
        if not clocks_ok:
            print("jetson_clocks warning: %s" % clocks_message)

        result = self._call(self._start_cli, Trigger.Request())
        if result[0]:
            with state_lock:
                state["detecting"] = True
            broadcast_state("status")
            if not clocks_ok:
                return result[0], "%s (jetson_clocks warning: %s)" % (result[1], clocks_message)
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


def _coordinate_filename_token(value):
    if value is None:
        return "unknown"
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if not math.isfinite(coordinate):
        return "unknown"
    return "%.8f" % coordinate


def _write_frame(jpeg, latitude=None, longitude=None):
    gmt_plus_8 = datetime.timezone(datetime.timedelta(hours=8))
    timestamp_token = datetime.datetime.now(gmt_plus_8).strftime(
        "%d_%m_%Y_GMT+8_%H_%M_%S"
    )
    latitude_token = _coordinate_filename_token(latitude)
    longitude_token = _coordinate_filename_token(longitude)
    filename_stem = "%s_%s_%s" % (
        timestamp_token,
        latitude_token,
        longitude_token,
    )
    sequence = frame_filename_counters.get(filename_stem, 1)

    while True:
        suffix = "" if sequence == 1 else "_%d" % sequence
        filename = "%s%s.jpg" % (filename_stem, suffix)
        path = os.path.join(SAVE_DIR, filename)
        try:
            with open(path, "xb") as output:
                output.write(jpeg)
            frame_filename_counters[filename_stem] = sequence + 1
            return filename, path
        except FileExistsError:
            sequence += 1


def _auto_save_loop():
    """Auto-save only while detection and fresh forward-only motion are active."""
    while True:
        auto_save_wakeup.wait(timeout=AUTO_SAVE_INTERVAL)
        auto_save_wakeup.clear()

        snapshot = _state_snapshot()
        if (
            not snapshot["auto_save"]
            or not snapshot["detecting"]
            or not _direction_allows_auto_save(snapshot)
        ):
            continue

        jpeg = _grab_frame_bytes()
        if jpeg is None:
            app.logger.warning("Auto-save skipped: MJPEG frame unavailable")
            continue

        # Re-check after the blocking frame read so state changes prevent a late save.
        snapshot = _state_snapshot()
        if (
            not snapshot["auto_save"]
            or not snapshot["detecting"]
            or snapshot["last_updated"] is None
            or not _direction_allows_auto_save(snapshot)
        ):
            continue

        frame_path = None
        try:
            with storage_lock:
                snapshot = _state_snapshot()
                if (
                    not snapshot["auto_save"]
                    or not snapshot["detecting"]
                    or snapshot["last_updated"] is None
                    or not _direction_allows_auto_save(snapshot)
                ):
                    continue
                snapshot["last_updated"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                frame_filename, frame_path = _write_frame(
                    jpeg, snapshot["latitude"], snapshot["longitude"]
                )
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
    snapshot = _state_snapshot()

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
        "latitude": snapshot["latitude"],
        "longitude": snapshot["longitude"],
        "gnss_time": snapshot["gnss_received_at"],
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
            frame_filename_counters.clear()
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


def _run_jetson_clocks():
    # Same nsenter-into-host-PID-1 trick as /api/shutdown below: pid: host
    # in docker-compose puts this container in the host's PID namespace, so
    # PID 1 here *is* the host's init. jetson_clocks/nvpmodel live on the
    # Jetson host, not in this container, so we have to run them there.
    # No sudo needed: this container runs as root, and nsenter carries that
    # UID into the host's namespaces -- same reason /api/shutdown's
    # `nsenter -- shutdown -h now` below needs no sudo either.
    try:
        result = subprocess.run(
            ["nsenter", "--target", "1", "--mount", "--uts", "--ipc", "--net", "--pid",
             "--", "jetson_clocks"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return False, "jetson_clocks failed: %s" % (result.stderr.strip() or result.stdout.strip())
        return True, "jetson_clocks applied"
    except subprocess.TimeoutExpired:
        return False, "jetson_clocks timed out"
    except OSError as exc:
        return False, "could not run jetson_clocks: %s" % exc


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


@app.route("/api/shutdown", methods=["POST"])
def shutdown_device():
    data = request.get_json(silent=True) or {}
    if data.get("confirm") != "shutdown":
        return jsonify({"success": False, "message": "confirmation required"}), 400

    # pid: host in docker-compose puts this container in the host's PID
    # namespace, so PID 1 here *is* the host's init -- nsenter into its
    # other namespaces to run shutdown against the actual Jetson, not
    # the container's own (otherwise inert) view of the system.
    try:
        subprocess.Popen([
            "nsenter", "--target", "1", "--mount", "--uts", "--ipc", "--net", "--pid",
            "--", "shutdown", "-h", "now",
        ])
    except OSError as exc:
        return jsonify({"success": False, "message": "could not initiate shutdown: %s" % exc}), 500

    return jsonify({"success": True, "message": "Jetson is shutting down now"})


@app.route("/api/save", methods=["POST"])
def save_frame():
    with storage_lock:
        jpeg = _grab_frame_bytes()
        if jpeg is None:
            return jsonify({"success": False, "message": "could not grab frame"}), 500
        snapshot = _state_snapshot()
        filename, _path = _write_frame(
            jpeg, snapshot["latitude"], snapshot["longitude"]
        )
    return jsonify({"success": True, "filename": filename})


@app.route("/api/models/upload", methods=["POST"])
def upload_model():
    if "model" not in request.files:
        return jsonify({"success": False, "message": "no file uploaded"}), 400
    file = request.files["model"]
    filename = Path(file.filename or "").name  # strip any path components
    if not filename.lower().endswith(".pt"):
        return jsonify({"success": False, "message": "only .pt files are accepted"}), 400
    if filename in ("", ".pt"):
        return jsonify({"success": False, "message": "invalid filename"}), 400
    dest_path = Path(WEIGHTS_DIR) / filename
    with storage_lock:
        file.save(str(dest_path))
    return jsonify({
        "success": True,
        "message": "model uploaded",
        "filename": filename,
        "size": dest_path.stat().st_size,
    })


@app.route("/api/models")
def list_models():
    files = sorted(Path(WEIGHTS_DIR).glob("*.pt"), reverse=True)
    return jsonify([
        {
            "filename": item.name,
            "size": item.stat().st_size,
            "time": datetime.datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
        }
        for item in files
    ])


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


@app.route("/api/images/download")
def download_images():
    archive_buffer = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)

    try:
        with storage_lock:
            image_paths = sorted(Path(SAVE_DIR).glob("*.jpg"))
            if not image_paths:
                archive_buffer.close()
                return jsonify({
                    "success": False,
                    "message": "no saved images are available to download",
                }), 404

            history_by_filename = _image_history_rows()
            manifest_buffer = io.StringIO(newline="")
            fieldnames = [
                "filename",
                "file_size_bytes",
                "captured_at",
                "history_id",
                "detection_recorded_at",
                "class_counts_json",
                "total_objects",
                "object_detected",
                "confidence",
                "tracking_enabled",
                "latitude",
                "longitude",
                "gnss_recorded_at",
            ]
            writer = csv.DictWriter(manifest_buffer, fieldnames=fieldnames)
            writer.writeheader()

            with zipfile.ZipFile(archive_buffer, mode="w", allowZip64=True) as archive:
                for image_path in image_paths:
                    file_stat = image_path.stat()
                    history = history_by_filename.get(image_path.name)
                    writer.writerow({
                        "filename": image_path.name,
                        "file_size_bytes": file_stat.st_size,
                        "captured_at": datetime.datetime.fromtimestamp(
                            file_stat.st_mtime, datetime.timezone.utc
                        ).isoformat(),
                        "history_id": history["id"] if history else "",
                        "detection_recorded_at": history["recorded_at"] if history else "",
                        "class_counts_json": history["counts_json"] if history else "",
                        "total_objects": history["total"] if history else "",
                        "object_detected": history["bed_status"] if history else "",
                        "confidence": history["confidence"] if history else "",
                        "tracking_enabled": history["tracking"] if history else "",
                        "latitude": history["latitude"] if history else "",
                        "longitude": history["longitude"] if history else "",
                        "gnss_recorded_at": history["gnss_recorded_at"] if history else "",
                    })
                    archive.write(
                        image_path,
                        arcname="images/%s" % image_path.name,
                        compress_type=zipfile.ZIP_STORED,
                    )

                archive.writestr(
                    "images_manifest.csv",
                    manifest_buffer.getvalue().encode("utf-8"),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
    except (OSError, sqlite3.Error, zipfile.BadZipFile) as exc:
        archive_buffer.close()
        app.logger.error("Could not prepare image archive: %s", exc)
        return jsonify({
            "success": False,
            "message": "could not prepare image archive: %s" % exc,
        }), 500

    archive_buffer.seek(0)
    archive_name = "saved_images_%s.zip" % datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    response = send_file(
        archive_buffer,
        mimetype="application/zip",
        as_attachment=True,
        download_name=archive_name,
    )
    response.call_on_close(archive_buffer.close)
    return response


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
