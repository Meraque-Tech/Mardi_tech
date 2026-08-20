#!/usr/bin/env python3
"""YOLOv8 TRT bed detection dashboard, REST API, and ROS 2 bridge."""

import collections
import csv
import datetime
import io
import json
import logging
import math
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32, Int32, String, UInt8
from std_srvs.srv import SetBool, Trigger
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters

from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_sock import Sock


# Configuration
SAVE_DIR = os.environ.get("SAVE_DIR", "/saved_frames")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
VIDEO_DIR = os.environ.get("VIDEO_DIR", "/uploaded_videos")
# Name of the yolov8_trt node whose "video_source" parameter the dashboard
# flips between "camera" and an uploaded file path (see bed_detect.launch.py).
TRT_NODE_NAME = os.environ.get("TRT_NODE_NAME", "yolov8_trt")
CONVERT_IMAGE = os.environ.get("CONVERT_IMAGE", "meraquetech/tensorrt-yolov8:ultralytics")
# docker run below talks to the *host* daemon via the mounted docker.sock, so
# its -v bind mounts must be host paths, not paths inside this container --
# same reason engine_file_build.sh uses $(pwd) rather than an in-container path.
HOST_WEIGHTS_DIR = os.environ.get("HOST_WEIGHTS_DIR")
HOST_YOLOV8_DIR = os.environ.get("HOST_YOLOV8_DIR")
# meraquetech/tensorrt-yolov8:ultralytics is an x86-only image (no arm64
# build) -- docker run rejects it with a platform-mismatch error on Jetson.
# Gate the .pt -> .wts conversion feature to x86 hosts only.
CONVERSION_SUPPORTED = platform.machine() in ("x86_64", "AMD64")
SYSTEM_TYPE = platform.machine() or "unknown"
# jetson_clocks/nvpmodel only exist on Jetson (aarch64) hosts; skip the call
# entirely on x86 rather than attempting it and reporting a spurious failure.
IS_JETSON = platform.machine() in ("aarch64", "arm64")
TRT_PARAMS_FILE = os.environ.get(
    "TRT_PARAMS_FILE",
    "/ros2_ws/install/yolov8_trt_bed_detect_orin/share/yolov8_trt_bed_detect_orin/config/trt_params.yaml",
)
# Same host-path requirement as HOST_WEIGHTS_DIR/HOST_YOLOV8_DIR: `docker
# compose -f ...` below talks to the host daemon via the mounted docker.sock,
# so it needs the compose file's *host* path, and the one-shot serialize
# service name differs per target (see docker-compose.yolov8-trt-bed-detect-
# orin-nano.yml / -x86.ros2.jazzy.yml).
HOST_COMPOSE_FILE = os.environ.get("HOST_COMPOSE_FILE")
SERIALIZE_SERVICE_NAME = os.environ.get("SERIALIZE_SERVICE_NAME")
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8080"))
API_PORT = int(os.environ.get("API_PORT", "8090"))
HISTORY_DB = os.environ.get("HISTORY_DB", os.path.join(SAVE_DIR, "count_history.db"))
AUTO_SAVE_INTERVAL = max(0.5, float(os.environ.get("AUTO_SAVE_INTERVAL", "0.5")))
GNSS_FIX_TOPIC = os.environ.get("GNSS_FIX_TOPIC", "/receiver/fix")
GNSS_STALE_TIMEOUT = float(os.environ.get("GNSS_STALE_TIMEOUT", "3.0"))
MOTION_STATE_TOPIC = os.environ.get(
    "MOTION_STATE_TOPIC", "/gnss_imu_eskf/motion_state_raw_gnss"
)
MOTION_POS_DEADBAND_TOPIC = os.environ.get(
    "MOTION_POS_DEADBAND_TOPIC", "/gnss_imu_eskf/motion_pos_deadband"
)
MOTION_POS_DEADBAND_M = float(os.environ.get("MOTION_POS_DEADBAND_M", "0.3"))
# QA report thresholds (see get_report() near the bottom of this file).
REPORT_LOW_CONFIDENCE = float(os.environ.get("REPORT_LOW_CONFIDENCE", "0.5"))
REPORT_NEAR_DUPLICATE_M = float(
    os.environ.get("REPORT_NEAR_DUPLICATE_M", str(MOTION_POS_DEADBAND_M))
)
REPORT_NO_DETECTION_RUN_LENGTH = int(
    os.environ.get("REPORT_NO_DETECTION_RUN_LENGTH", "3")
)
# Local ENU position of the raw GNSS fix (nav_msgs/Odometry.pose.pose.position)
# -- the auto-save minimum-distance gate and the "distance from previous frame"
# history column are both computed from this rather than lat/lon, since it's
# the same flat-Earth frame gnss_imu_eskf_node itself uses for its motion
# classification (no haversine/earth-radius approximation needed).
GNSS_ONLY_ODOM_TOPIC = os.environ.get(
    "GNSS_ONLY_ODOM_TOPIC", "/gnss_imu_eskf/gnss_only_odom"
)
DIRECTION_STALE_TIMEOUT = float(os.environ.get("DIRECTION_STALE_TIMEOUT", "3.0"))
STATIC_DIR = Path(__file__).parent / "static"

if not math.isfinite(GNSS_STALE_TIMEOUT) or GNSS_STALE_TIMEOUT <= 0:
    raise ValueError("GNSS_STALE_TIMEOUT must be a finite number greater than zero")
if not math.isfinite(DIRECTION_STALE_TIMEOUT) or DIRECTION_STALE_TIMEOUT <= 0:
    raise ValueError("DIRECTION_STALE_TIMEOUT must be a finite number greater than zero")
if not math.isfinite(MOTION_POS_DEADBAND_M) or MOTION_POS_DEADBAND_M <= 0:
    raise ValueError("MOTION_POS_DEADBAND_M must be a finite number greater than zero")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_DB) or ".", exist_ok=True)


# Shared live state
state = {
    "counts": {},
    "bed_status": 0,
    "conf": 0.0,
    "detecting": False,
    "is_track": False,
    "auto_save": False,
    "video_source": "camera",
    "motion_pos_deadband_m": MOTION_POS_DEADBAND_M,
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
motion_state_last_monotonic = None
# Local ENU position from /gnss_imu_eskf/gnss_only_odom (see
# GNSS_ONLY_ODOM_TOPIC above) -- None until the first odom message arrives.
gnss_odom_x = None
gnss_odom_y = None
last_save_x = None
last_save_y = None
# Bounded trace of recent ENU path points for the UI's small path plot, each
# {"x", "y", "captured": bool} -- "captured" marks the point nearest a saved
# frame. Path-only, not persisted: it exists to show recent motion, not as a
# historical record (count_history/DB already covers that per saved frame).
PATH_POINTS_MAXLEN = 2000
# Only append a new path point once the robot has moved at least this far
# from the last recorded point, so a stationary robot doesn't fill the trace
# with a stack of overlapping points at native GNSS PVT rate.
PATH_POINT_MIN_SPACING_M = 0.05
path_points = collections.deque(maxlen=PATH_POINTS_MAXLEN)
path_points_lock = threading.Lock()
auto_save_wakeup = threading.Event()
ws_clients = []  # type: List
ws_lock = threading.Lock()

convert_job = {"running": False, "pt_filename": None, "message": None, "ok": None}
convert_job_lock = threading.Lock()

engine_build_job = {
    "running": False, "wts_filename": None, "engine_filename": None,
    "message": None, "ok": None,
}
engine_build_job_lock = threading.Lock()

launch_job = {"running": False, "mode": None, "message": None, "ok": None}
launch_job_lock = threading.Lock()
launch_process = None
launch_log = collections.deque(maxlen=500)
launch_log_lock = threading.Lock()


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
                gnss_recorded_at TEXT,
                distance_from_prev_m REAL
            )
            """
        )
        columns = {row["name"] for row in db.execute("PRAGMA table_info(count_history)")}
        migrations = {
            "frame_filename": "TEXT",
            "latitude": "REAL",
            "longitude": "REAL",
            "gnss_recorded_at": "TEXT",
            "distance_from_prev_m": "REAL",
        }
        for column, column_type in migrations.items():
            if column not in columns:
                db.execute(
                    "ALTER TABLE count_history ADD COLUMN %s %s" % (column, column_type)
                )


def _store_history(snapshot, frame_filename=None, distance_from_prev_m=None):
    with db_lock, _db_connect() as db:
        cursor = db.execute(
            """
            INSERT INTO count_history
                (recorded_at, counts_json, total, bed_status, confidence, tracking,
                 frame_filename, latitude, longitude, gnss_recorded_at,
                 distance_from_prev_m)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                distance_from_prev_m,
            ),
        )
        return cursor.lastrowid


def _history_rows(limit, offset):
    with db_lock, _db_connect() as db:
        total = db.execute("SELECT COUNT(*) FROM count_history").fetchone()[0]
        rows = db.execute(
            """
            SELECT id, recorded_at, counts_json, total, bed_status, confidence, tracking,
                   frame_filename, latitude, longitude, gnss_recorded_at,
                   distance_from_prev_m
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
            "distance_from_prev_m": row["distance_from_prev_m"],
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


def _report_rows():
    """All saved-frame rows in capture order (oldest first), for QA report
    generation -- distinct from _image_history_rows() (which keys by
    filename for the ZIP manifest) and _history_rows() (which paginates for
    the live UI table).
    """
    with db_lock, _db_connect() as db:
        rows = db.execute(
            """
            SELECT id, recorded_at, counts_json, total, bed_status, confidence,
                   frame_filename, latitude, longitude, distance_from_prev_m
            FROM count_history
            WHERE frame_filename IS NOT NULL
            ORDER BY id ASC
            """
        ).fetchall()
    return [dict(row) for row in rows]


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
            motion_state_last_monotonic is not None
            and now_monotonic - motion_state_last_monotonic <= DIRECTION_STALE_TIMEOUT
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
    snapshot["auto_save_distance_eligible"] = not _below_min_save_distance(snapshot)
    snapshot["auto_save_active"] = (
        snapshot["auto_save"]
        and snapshot["detecting"]
        and snapshot["auto_save_direction_eligible"]
        and snapshot["auto_save_distance_eligible"]
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
        self.create_subscription(String, MOTION_STATE_TOPIC, self._motion_state_cb, 10)
        self.create_subscription(
            Odometry, GNSS_ONLY_ODOM_TOPIC, self._gnss_only_odom_cb, 10
        )
        self._motion_pos_deadband_pub = self.create_publisher(
            Float32, MOTION_POS_DEADBAND_TOPIC, 10
        )

        self._start_cli = self.create_client(Trigger, "/bed_detection")
        self._stop_cli = self.create_client(Trigger, "/bed_detection_stop")
        self._reset_cli = self.create_client(Trigger, "/reset_tracker")
        self._track_cli = self.create_client(SetBool, "/set_tracking")
        self._set_params_cli = self.create_client(
            SetParameters, "/%s/set_parameters" % TRT_NODE_NAME
        )
        self._control_lock = threading.Lock()
        self._last_direction_broadcast = (None, None, False)
        self.get_logger().info(
            "Listening for GNSS fixes on %s (stale after %.1f s)"
            % (GNSS_FIX_TOPIC, GNSS_STALE_TIMEOUT)
        )
        self.get_logger().info(
            "Listening for motion state on %s (stale after %.1f s)"
            % (MOTION_STATE_TOPIC, DIRECTION_STALE_TIMEOUT)
        )
        self.get_logger().info(
            "Publishing motion deadband updates on %s" % MOTION_POS_DEADBAND_TOPIC
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
        
        print("GNSS fix received: lat=%f, lon=%f, status=%d" % (latitude, longitude, msg.status.status))

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

    def _gnss_only_odom_cb(self, msg):
        # Local ENU position of the raw GNSS fix, same frame
        # gnss_imu_eskf_node uses internally -- source of truth for the
        # auto-save minimum-distance gate, the history table's "distance
        # from previous frame" column, and the UI's path trace below.
        global gnss_odom_x, gnss_odom_y
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        with state_lock:
            gnss_odom_x = x
            gnss_odom_y = y

        with path_points_lock:
            if (
                not path_points
                or math.hypot(
                    x - path_points[-1]["x"], y - path_points[-1]["y"]
                ) >= PATH_POINT_MIN_SPACING_M
            ):
                path_points.append({"x": x, "y": y, "captured": False})

    def _motion_state_cb(self, msg):
        global motion_state_last_monotonic

        # /gnss_imu_eskf/motion_state_raw_gnss publishes one of:
        # "idle", "forward", "backward".
        motion_state = (msg.data or "").strip()
        is_forward = motion_state == "forward"
        is_backward = motion_state == "backward"

        now_monotonic = time.monotonic()
        with state_lock:
            state["is_forward"] = is_forward
            state["is_backward"] = is_backward
            state["direction_received_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()
            motion_state_last_monotonic = now_monotonic

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

    def _call_raw(self, client, request_message):
        # The node is already spinning in _ros_spin. Waiting on an Event here
        # avoids trying to add it to a second executor from a Flask thread.
        # Returns (response_or_None, error_message_or_None).
        with self._control_lock:
            if not client.wait_for_service(timeout_sec=2.0):
                return None, "ROS service is not available"
            future = client.call_async(request_message)
            completed = threading.Event()
            future.add_done_callback(lambda _future: completed.set())
            if not completed.wait(timeout=4.0):
                return None, "ROS service timed out"
            try:
                result = future.result()
            except Exception as exc:
                return None, "ROS service failed: %s" % exc
            if result is None:
                return None, "ROS service returned no response"
            return result, None

    def _call(self, client, request_message):
        # For Trigger/SetBool-style services that respond with .success/.message.
        result, error = self._call_raw(client, request_message)
        if result is None:
            return False, error
        return bool(result.success), result.message

    def call_start(self):
        if IS_JETSON:
            clocks_ok, clocks_message = _run_jetson_clocks()
        else:
            clocks_ok, clocks_message = True, None
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

    def call_set_video_source(self, source):
        # "camera" reverts the yolov8_trt node to the live camera; any other
        # value is a video file path it opens with cv::VideoCapture instead
        # (see main.cpp's video_source parameter and open_video_file()).
        param = Parameter(
            name="video_source",
            value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(source)),
        )
        request_message = SetParameters.Request(parameters=[param])
        response, error = self._call_raw(self._set_params_cli, request_message)
        if response is None:
            return False, error
        result = response.results[0]
        ok = bool(result.successful)
        detail = result.reason or ("video source set to %s" % source)
        if ok:
            with state_lock:
                state["video_source"] = str(source)
            broadcast_state("status")
        return ok, detail

    def call_set_class_labels(self, labels):
        # Pushes the dashboard's "Class Labels" mapping (id -> name) down to
        # the yolov8_trt node so the MJPEG overlay's cv::putText calls can
        # draw names instead of raw class indices (see main.cpp's
        # class_labels_json parameter and parse_class_labels_json()).
        payload = json.dumps({str(k): str(v) for k, v in labels.items()})
        param = Parameter(
            name="class_labels_json",
            value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=payload),
        )
        request_message = SetParameters.Request(parameters=[param])
        response, error = self._call_raw(self._set_params_cli, request_message)
        if response is None:
            return False, error
        result = response.results[0]
        ok = bool(result.successful)
        detail = result.reason or "class labels updated"
        return ok, detail

    def set_motion_pos_deadband(self, value):
        message = Float32()
        message.data = float(value)
        with self._control_lock:
            self._motion_pos_deadband_pub.publish(message)
        with state_lock:
            state["motion_pos_deadband_m"] = float(value)
        broadcast_state("status")
        self.get_logger().info("Motion position deadband set to %.3f m" % value)


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


def _gnss_odom_xy():
    with state_lock:
        return gnss_odom_x, gnss_odom_y


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


def _mark_last_path_point_captured():
    with path_points_lock:
        if path_points:
            path_points[-1]["captured"] = True


def _distance_since_last_save_m(snapshot):
    """Euclidean distance (m) in the ESKF's local ENU frame from the last
    saved position to the current /gnss_imu_eskf/gnss_only_odom position, or
    None if either the current or last-saved position is unknown.
    """
    x, y = _gnss_odom_xy()
    if x is None or y is None:
        return None
    if last_save_x is None or last_save_y is None:
        return None
    return math.hypot(x - last_save_x, y - last_save_y)


def _below_min_save_distance(snapshot):
    """True if an odom position exists but hasn't moved far enough to save yet.

    Distance is only enforced when both the current and last-saved position
    are known -- with no prior save, or no odom data at all, the threshold
    can't be evaluated, so auto-save falls back to the interval-only cadence.
    """
    distance_m = _distance_since_last_save_m(snapshot)
    if distance_m is None:
        return False
    return distance_m < snapshot["motion_pos_deadband_m"]


def _auto_save_loop():
    """Auto-save only while detection, fresh forward-only motion, and the
    minimum travel distance since the last save are all satisfied."""
    global last_save_x, last_save_y

    while True:
        auto_save_wakeup.wait(timeout=AUTO_SAVE_INTERVAL)
        auto_save_wakeup.clear()

        snapshot = _state_snapshot()
        if (
            not snapshot["auto_save"]
            or not snapshot["detecting"]
            or not _direction_allows_auto_save(snapshot)
            or _below_min_save_distance(snapshot)
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
            or _below_min_save_distance(snapshot)
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
                    or _below_min_save_distance(snapshot)
                ):
                    continue
                distance_from_prev_m = _distance_since_last_save_m(snapshot)
                snapshot["last_updated"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat()
                frame_filename, frame_path = _write_frame(
                    jpeg, snapshot["latitude"], snapshot["longitude"]
                )
                _store_history(snapshot, frame_filename, distance_from_prev_m)
                x, y = _gnss_odom_xy()
                if x is not None and y is not None:
                    last_save_x = x
                    last_save_y = y
                _mark_last_path_point_captured()
        except (OSError, sqlite3.Error) as exc:
            if frame_path and os.path.exists(frame_path):
                os.unlink(frame_path)
            app.logger.error("Auto-save failed: %s", exc)


# Flask app
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.logger.setLevel(logging.INFO)
sock = Sock(app)


@app.route("/")
def dashboard():
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/saved/<path:filename>")
def saved_file(filename):
    return send_from_directory(SAVE_DIR, filename)


@sock.route("/ws")
def websocket(ws):
    peer = request.remote_addr or "unknown"
    app.logger.info("Dashboard WebSocket connected remote=%s", peer)
    with ws_lock:
        ws_clients.append(ws)
    try:
        ws.send(json.dumps(_snapshot_payload()))
        while True:
            message = ws.receive(timeout=15)
            if message is None:
                break
            if message == "ping":
                ws.send(json.dumps({
                    "type": "pong",
                    "server_time": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }))
    except Exception as exc:
        app.logger.info(
            "Dashboard WebSocket disconnected remote=%s reason=%s", peer, exc
        )
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


@app.route("/api/path")
def get_path():
    with path_points_lock:
        points = list(path_points)
    return jsonify({"points": points})


@app.route("/api/save_count", methods=["POST"])
def save_count():
    global last_save_x, last_save_y

    with state_lock:
        if state["last_updated"] is None:
            return jsonify({"success": False, "message": "no live count has been received yet"}), 409
    snapshot = _state_snapshot()

    source_updated = snapshot["last_updated"]
    snapshot["last_updated"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        with storage_lock:
            distance_from_prev_m = _distance_since_last_save_m(snapshot)
            record_id = _store_history(snapshot, distance_from_prev_m=distance_from_prev_m)
            x, y = _gnss_odom_xy()
            if x is not None and y is not None:
                last_save_x = x
                last_save_y = y
            _mark_last_path_point_captured()
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
    global last_save_x, last_save_y

    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        return jsonify({"success": False, "message": "enabled must be a boolean"}), 400
    with state_lock:
        state["auto_save"] = enabled
        if enabled:
            last_save_x = None
            last_save_y = None
    auto_save_wakeup.set()
    broadcast_state("status")
    return jsonify({
        "success": True,
        "message": "automatic saving enabled" if enabled else "automatic saving disabled",
        "auto_save": enabled,
        "interval_seconds": AUTO_SAVE_INTERVAL,
    })


@app.route("/api/motion_pos_deadband", methods=["POST"])
def set_motion_pos_deadband():
    data = request.get_json(silent=True) or {}
    try:
        deadband_m = float(data.get("deadband_m"))
    except (TypeError, ValueError):
        return jsonify({
            "success": False, "message": "deadband_m must be a number",
        }), 400
    if not math.isfinite(deadband_m) or deadband_m <= 0:
        return jsonify({
            "success": False,
            "message": "deadband_m must be a finite number greater than zero",
        }), 400

    bridge = ros_node
    if bridge is None:
        return jsonify({"success": False, "message": "ROS bridge is not ready"}), 503
    bridge.set_motion_pos_deadband(deadband_m)
    return jsonify({
        "success": True,
        "message": "motion deadband set to %.3f m" % deadband_m,
        "motion_pos_deadband_m": deadband_m,
        "topic": MOTION_POS_DEADBAND_TOPIC,
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

    action = {
        "call_start": "start",
        "call_stop": "stop",
        "call_reset": "reset_tracker",
    }.get(method_name, method_name)
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    user_agent = request.headers.get("User-Agent", "")[:240]
    log = app.logger.warning if action == "stop" else app.logger.info
    log(
        "Detection control requested action=%s request_id=%s remote=%s "
        "forwarded_for=%s user_agent=%s",
        action,
        request_id,
        request.remote_addr or "unknown",
        forwarded_for or "-",
        user_agent or "-",
    )
    ok, message = getattr(bridge, method_name)()
    log(
        "Detection control completed action=%s request_id=%s success=%s message=%s",
        action,
        request_id,
        ok,
        message,
    )
    return jsonify({
        "success": ok,
        "message": message,
        "request_id": request_id,
    }), (200 if ok else 503)


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


def _save_upload(directory, extension, field="model"):
    if field not in request.files:
        return None, (jsonify({"success": False, "message": "no file uploaded"}), 400)
    file = request.files[field]
    filename = Path(file.filename or "").name  # strip any path components
    extensions = (extension,) if isinstance(extension, str) else tuple(extension)
    if not filename.lower().endswith(extensions):
        return None, (jsonify({"success": False, "message": "only %s files are accepted" % ", ".join(extensions)}), 400)
    if filename in ("", *extensions):
        return None, (jsonify({"success": False, "message": "invalid filename"}), 400)
    dest_path = Path(directory) / filename
    with storage_lock:
        file.save(str(dest_path))
    return dest_path, None


@app.route("/api/models/upload", methods=["POST"])
def upload_model():
    dest_path, error = _save_upload(WEIGHTS_DIR, ".wts")
    if error:
        return error
    return jsonify({
        "success": True,
        "message": "model uploaded",
        "filename": dest_path.name,
        "size": dest_path.stat().st_size,
    })


@app.route("/api/models/upload_pt", methods=["POST"])
def upload_pt_model():
    dest_path, error = _save_upload(WEIGHTS_DIR, ".pt")
    if error:
        return error
    return jsonify({
        "success": True,
        "message": "model uploaded",
        "filename": dest_path.name,
        "size": dest_path.stat().st_size,
    })


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@app.route("/api/video/upload", methods=["POST"])
def upload_video():
    dest_path, error = _save_upload(VIDEO_DIR, VIDEO_EXTENSIONS, field="video")
    if error:
        return error
    return jsonify({
        "success": True,
        "message": "video uploaded",
        "filename": dest_path.name,
        "size": dest_path.stat().st_size,
    })


@app.route("/api/video/list")
def list_videos():
    return jsonify(_list_dir(VIDEO_DIR, "*"))


@app.route("/api/video/source", methods=["POST"])
def set_video_source():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if mode not in ("camera", "file"):
        return jsonify({"success": False, "message": "mode must be 'camera' or 'file'"}), 400

    if mode == "camera":
        source = "camera"
    else:
        filename = Path(str(data.get("filename") or "")).name
        if not filename:
            return jsonify({"success": False, "message": "filename is required for file mode"}), 400
        candidate = Path(VIDEO_DIR) / filename
        if not candidate.is_file():
            return jsonify({"success": False, "message": "no such uploaded video: %s" % filename}), 404
        source = str(candidate)

    if ros_node is None:
        return jsonify({"success": False, "message": "ROS bridge is not ready"}), 503
    ok, message = ros_node.call_set_video_source(source)
    return jsonify({"success": ok, "message": message, "video_source": mode}), (200 if ok else 503)


@app.route("/api/class_labels", methods=["POST"])
def set_class_labels():
    # The id -> name mapping is edited entirely client-side (the dashboard's
    # "Class Labels" panel, persisted in localStorage) -- this just forwards
    # the current mapping to the yolov8_trt node so its video overlay can use
    # it too. Not stored server-side; the frontend is the source of truth.
    data = request.get_json(silent=True) or {}
    labels = data.get("labels")
    if not isinstance(labels, dict):
        return jsonify({"success": False, "message": "labels must be an object"}), 400

    if ros_node is None:
        return jsonify({"success": False, "message": "ROS bridge is not ready"}), 503
    ok, message = ros_node.call_set_class_labels(labels)
    return jsonify({"success": ok, "message": message}), (200 if ok else 503)


def _list_dir(directory, pattern):
    files = sorted(Path(directory).glob(pattern), reverse=True)
    return [
        {
            "filename": f.name,
            "size": f.stat().st_size,
            "time": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        }
        for f in files
    ]


@app.route("/api/models")
def list_models():
    return jsonify({
        "pt": _list_dir(WEIGHTS_DIR, "*.pt"),
        "wts": _list_dir(WEIGHTS_DIR, "*.wts"),
        "engine": _list_dir(WEIGHTS_DIR, "*.engine"),
        "conversion_supported": CONVERSION_SUPPORTED,
    })


_MODEL_KIND_EXTENSIONS = {"pt", "wts", "engine"}


@app.route("/api/models/<kind>/<filename>/download")
def download_model(kind, filename):
    if kind not in _MODEL_KIND_EXTENSIONS:
        return jsonify({"success": False, "message": "unknown file type"}), 400
    if Path(filename).name != filename or not filename.lower().endswith("." + kind):
        return jsonify({"success": False, "message": "invalid filename"}), 400
    path = Path(WEIGHTS_DIR) / filename
    if not path.is_file():
        return jsonify({"success": False, "message": "file not found"}), 404
    return send_file(str(path), as_attachment=True, download_name=filename)


@app.route("/api/models/<kind>/<filename>", methods=["DELETE"])
def delete_model(kind, filename):
    if kind not in _MODEL_KIND_EXTENSIONS:
        return jsonify({"success": False, "message": "unknown file type"}), 400
    if Path(filename).name != filename or not filename.lower().endswith("." + kind):
        return jsonify({"success": False, "message": "invalid filename"}), 400
    path = Path(WEIGHTS_DIR) / filename
    with storage_lock:
        if not path.is_file():
            return jsonify({"success": False, "message": "file not found"}), 404
        path.unlink()
    return jsonify({"success": True})


def _run_conversion(pt_filename):
    # Mirrors engine_file_build.sh's Step 1 exactly: docker run against the
    # ultralytics image, mounting the *host* yolov8/ source dir (for
    # gen_wts.py) and the host weights dir. This container only has the
    # docker CLI + docker.sock -- it talks to the host daemon, so these -v
    # paths must be host paths (HOST_YOLOV8_DIR / HOST_WEIGHTS_DIR), not
    # paths inside this container.
    wts_filename = os.path.splitext(pt_filename)[0] + ".wts"
    if not HOST_YOLOV8_DIR or not HOST_WEIGHTS_DIR:
        with convert_job_lock:
            convert_job.update(
                running=False, ok=False,
                message="HOST_YOLOV8_DIR / HOST_WEIGHTS_DIR not configured for this container",
            )
        return
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--net=host",
             "--runtime", "nvidia", "--gpus", "all", "--privileged",
             "-v", f"{HOST_WEIGHTS_DIR}:/workspace/yolov8/build/weights",
             "-v", f"{HOST_YOLOV8_DIR}:/yolov8",
             CONVERT_IMAGE,
             "bash", "-c",
             f"cd /yolov8 && python3 gen_wts.py "
             f"-w /workspace/yolov8/build/weights/{pt_filename} "
             f"-o /workspace/yolov8/build/weights/{wts_filename} -t detect"],
            capture_output=True, text=True, timeout=600,
        )
        ok = result.returncode == 0
        message = "conversion complete" if ok else (result.stderr.strip() or "conversion failed")
    except subprocess.TimeoutExpired:
        ok, message = False, "conversion timed out"
    except OSError as exc:
        ok, message = False, "could not start conversion: %s" % exc
    with convert_job_lock:
        convert_job.update(running=False, message=message, ok=ok)


@app.route("/api/models/convert", methods=["POST"])
def convert_model():
    if not CONVERSION_SUPPORTED:
        return jsonify({
            "success": False,
            "message": ".pt to .wts conversion is only supported on x86 hosts "
                       "(meraquetech/tensorrt-yolov8:ultralytics has no arm64 build)",
        }), 400
    data = request.get_json(silent=True) or {}
    pt_filename = Path(data.get("filename", "")).name
    if not pt_filename.lower().endswith(".pt"):
        return jsonify({"success": False, "message": "filename must be a .pt file"}), 400
    if not (Path(WEIGHTS_DIR) / pt_filename).is_file():
        return jsonify({"success": False, "message": "file not found"}), 404
    with convert_job_lock:
        if convert_job["running"]:
            return jsonify({"success": False, "message": "a conversion is already running"}), 409
        convert_job.update(running=True, pt_filename=pt_filename, message=None, ok=None)
    threading.Thread(target=_run_conversion, args=(pt_filename,), daemon=True).start()
    return jsonify({"success": True, "message": "conversion started"})


@app.route("/api/models/convert/status")
def convert_status():
    with convert_job_lock:
        return jsonify(dict(convert_job))


def _set_yaml_scalar(lines, key, value, quote=True):
    # Surgical line replacement instead of yaml.safe_load/safe_dump: this
    # file is hand-maintained with extensive comments and commented-out
    # alternatives (see trt_prams_main.yaml) that a full YAML parse/dump
    # round-trip would silently discard. Only the first *uncommented*
    # "key: ..." line is replaced, preserving indentation and every comment.
    pattern = re.compile(r'^(\s*)%s:(?!\S)' % re.escape(key))
    formatted = '"%s"' % value if quote else str(value)
    for i, line in enumerate(lines):
        match = pattern.match(line)
        if match:
            lines[i] = '%s%s: %s\n' % (match.group(1), key, formatted)
            return True
    return False


def _update_trt_params(**scalars):
    with open(TRT_PARAMS_FILE) as f:
        lines = f.readlines()
    shutil.copy2(TRT_PARAMS_FILE, TRT_PARAMS_FILE + ".bak")
    for key, (value, quote) in scalars.items():
        if not _set_yaml_scalar(lines, key, value, quote=quote):
            raise ValueError("%s key not found in %s" % (key, TRT_PARAMS_FILE))
    with open(TRT_PARAMS_FILE, "w") as f:
        f.writelines(lines)


@app.route("/api/models/build_engine", methods=["POST"])
def build_engine():
    data = request.get_json(silent=True) or {}
    wts_filename = Path(data.get("filename", "")).name
    if not wts_filename.lower().endswith(".wts"):
        return jsonify({"success": False, "message": "filename must be a .wts file"}), 400
    if not (Path(WEIGHTS_DIR) / wts_filename).is_file():
        return jsonify({"success": False, "message": "file not found"}), 404
    try:
        num_class = int(data.get("num_class"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "num_class must be a positive integer"}), 400
    if num_class <= 0:
        return jsonify({"success": False, "message": "num_class must be a positive integer"}), 400
    stem = Path(wts_filename).stem
    engine_filename = f"{stem}_{SYSTEM_TYPE}.engine"
    wts_path = f"{WEIGHTS_DIR}/{wts_filename}"
    engine_path = f"{WEIGHTS_DIR}/{engine_filename}"
    try:
        _update_trt_params(
            wts_name=(wts_path, True),
            engine_name=(engine_path, True),
            num_class=(num_class, False),
        )
    except (OSError, ValueError) as exc:
        return jsonify({"success": False, "message": "could not update trt_params.yaml: %s" % exc}), 500
    return jsonify({
        "success": True,
        "message": "trt_params.yaml updated — restart the detection service to build %s" % engine_filename,
        "engine_filename": engine_filename,
        "wts_name": wts_path,
        "engine_name": engine_path,
        "num_class": num_class,
    })


@app.route("/api/models/select_engine", methods=["POST"])
def select_engine():
    data = request.get_json(silent=True) or {}
    engine_filename = Path(data.get("filename", "")).name
    if not engine_filename.lower().endswith(".engine"):
        return jsonify({"success": False, "message": "filename must be a .engine file"}), 400
    if not (Path(WEIGHTS_DIR) / engine_filename).is_file():
        return jsonify({"success": False, "message": "file not found"}), 404
    try:
        num_class = int(data.get("num_class"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "num_class must be a positive integer"}), 400
    if num_class <= 0:
        return jsonify({"success": False, "message": "num_class must be a positive integer"}), 400
    engine_path = f"{WEIGHTS_DIR}/{engine_filename}"
    try:
        _update_trt_params(
            engine_name=(engine_path, True),
            num_class=(num_class, False),
        )
    except (OSError, ValueError) as exc:
        return jsonify({"success": False, "message": "could not update trt_params.yaml: %s" % exc}), 500
    return jsonify({
        "success": True,
        "message": "trt_params.yaml now points at %s — deserialize to run detection with it" % engine_filename,
        "engine_name": engine_path,
        "num_class": num_class,
    })


_LAUNCH_FILES = {
    "serialize": "serialize_engine.launch.py",
    "deserialize": "bed_detect.launch.py",
}


def _read_engine_path():
    try:
        with open(TRT_PARAMS_FILE) as f:
            lines = f.readlines()
    except OSError:
        return None
    pattern = re.compile(r'^\s*engine_name:\s*"?([^"\s]+)"?\s*$')
    for line in lines:
        if line.lstrip().startswith("#"):
            continue
        match = pattern.match(line)
        if match:
            return match.group(1)
    return None


def _stop_launch_process(timeout=10.0):
    """Stop the running launch process, if any. Caller must hold launch_job_lock."""
    process = launch_process
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def _log_line(line):
    with launch_log_lock:
        launch_log.append(line.rstrip("\n"))


def _run_launch(mode):
    global launch_process
    launch_file = _LAUNCH_FILES[mode]
    with launch_log_lock:
        launch_log.clear()
    _log_line("$ ros2 launch yolov8_trt_bed_detect_orin %s" % launch_file)
    try:
        process = subprocess.Popen(
            ["ros2", "launch", "yolov8_trt_bed_detect_orin", launch_file],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
    except OSError as exc:
        _log_line("could not start ros2 launch: %s" % exc)
        with launch_job_lock:
            launch_job.update(running=False, message="could not start ros2 launch: %s" % exc, ok=False)
        return

    with launch_job_lock:
        launch_process = process

    for line in process.stdout:
        _log_line(line)
    process.wait()

    ok = process.returncode == 0
    stopped = process.returncode < 0
    if stopped:
        message = "%s stopped" % mode
    elif ok:
        message = "%s complete" % mode
    else:
        message = "%s failed (exit %d) — see log" % (mode, process.returncode)
    _log_line(message)
    with launch_job_lock:
        launch_process = None
        launch_job.update(running=False, message=message, ok=(None if stopped else ok))


def _start_launch(mode, restart_same_mode=False):
    with launch_job_lock:
        if launch_job["running"]:
            if launch_job["mode"] == mode and not restart_same_mode:
                return jsonify({
                    "success": False,
                    "message": "a %s launch is already running" % mode,
                }), 409
            _stop_launch_process()
        launch_job.update(running=True, mode=mode, message=None, ok=None)
    threading.Thread(target=_run_launch, args=(mode,), daemon=True).start()
    return jsonify({"success": True, "message": "%s started" % mode})


@app.route("/api/serialize/model", methods=["POST"])
def serialize_model():
    engine_path = _read_engine_path()
    if engine_path and os.path.isfile(engine_path):
        with launch_job_lock:
            already_running = launch_job["running"] and launch_job["mode"] == "serialize"
        if not already_running:
            return jsonify({
                "success": True,
                "message": "engine already exists at %s, skipping serialize" % engine_path,
                "engine_name": engine_path,
                "skipped": True,
            })
    return _start_launch("serialize")


@app.route("/api/deserialize/model", methods=["POST"])
def deserialize_model():
    return _start_launch("deserialize", restart_same_mode=True)


@app.route("/api/models/launch/status")
def launch_status():
    with launch_job_lock:
        return jsonify(dict(launch_job))


@app.route("/api/models/launch/log")
def launch_log_route():
    with launch_log_lock:
        return jsonify({"lines": list(launch_log)})


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


def _class_totals(rows):
    """Sum each class's count across all saved-frame rows.

    Returns (totals, grand_total) where totals is a dict of
    class_id -> summed count, ordered numerically where class IDs are
    numeric (falling back to string order otherwise), and grand_total is
    the sum across all classes.
    """
    totals = collections.defaultdict(int)
    for row in rows:
        try:
            counts = json.loads(row["counts_json"])
        except (TypeError, ValueError):
            continue
        for class_id, count in counts.items():
            try:
                totals[class_id] += int(count)
            except (TypeError, ValueError):
                continue

    def _class_sort_key(class_id):
        try:
            return (0, int(class_id))
        except ValueError:
            return (1, class_id)

    ordered = {
        class_id: totals[class_id]
        for class_id in sorted(totals, key=_class_sort_key)
    }
    return ordered, sum(ordered.values())


def _build_qa_report(rows):
    """Flag data-quality issues across all saved-frame rows (oldest first).

    Four checks, each answering "what should the operator double-check or
    re-survey": low-confidence detections, frames saved without a GNSS fix,
    near-duplicate captures (little/no travel since the previous save --
    likely redundant), and runs of consecutive frames with zero detections
    (possible coverage gap or missed object).
    """
    low_confidence = [
        row for row in rows
        if row["confidence"] is not None and row["confidence"] < REPORT_LOW_CONFIDENCE
    ]
    missing_gnss = [
        row for row in rows
        if row["latitude"] is None or row["longitude"] is None
    ]
    near_duplicates = [
        row for row in rows
        if row["distance_from_prev_m"] is not None
        and row["distance_from_prev_m"] < REPORT_NEAR_DUPLICATE_M
    ]

    no_detection_runs = []
    current_run = []
    for row in rows:
        if row["total"] == 0:
            current_run.append(row)
            continue
        if len(current_run) >= REPORT_NO_DETECTION_RUN_LENGTH:
            no_detection_runs.append(current_run)
        current_run = []
    if len(current_run) >= REPORT_NO_DETECTION_RUN_LENGTH:
        no_detection_runs.append(current_run)

    return {
        "low_confidence": low_confidence,
        "missing_gnss": missing_gnss,
        "near_duplicates": near_duplicates,
        "no_detection_runs": no_detection_runs,
    }


def _report_html(rows, flags, class_labels=None):
    class_labels = class_labels or {}

    def esc(value):
        return (
            str(value)
            .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    def class_label(class_id):
        name = class_labels.get(str(class_id))
        return "%s (%s)" % (name, class_id) if name else "Class %s" % class_id

    def row_line(row, extra=""):
        gnss = (
            "%.6f, %.6f" % (row["latitude"], row["longitude"])
            if row["latitude"] is not None and row["longitude"] is not None
            else "no fix"
        )
        return (
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                esc(row["frame_filename"]), esc(row["recorded_at"]),
                esc("%.3f" % row["confidence"] if row["confidence"] is not None else "—"),
                esc(gnss), esc(extra),
            )
        )

    def section(title, description, table_rows):
        if not table_rows:
            return (
                "<h2>%s</h2><p class='desc'>%s</p><p class='ok'>None found.</p>"
                % (esc(title), esc(description))
            )
        return (
            "<h2>%s <span class='count'>(%d)</span></h2>"
            "<p class='desc'>%s</p>"
            "<table><thead><tr><th>Frame</th><th>Captured</th>"
            "<th>Confidence</th><th>GNSS</th><th>Note</th></tr></thead>"
            "<tbody>%s</tbody></table>"
            % (esc(title), len(table_rows), esc(description), "".join(table_rows))
        )

    total_frames = len(rows)
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    class_totals, class_grand_total = _class_totals(rows)
    if class_totals:
        class_total_rows = "".join(
            "<tr><td>%s</td><td>%d</td></tr>" % (esc(class_label(class_id)), count)
            for class_id, count in class_totals.items()
        )
        class_totals_section = (
            "<h2>Class totals <span class='count neutral'>(%d classes)</span></h2>"
            "<p class='desc'>Summed detection count per class across all saved frames.</p>"
            "<table><thead><tr><th>Class</th><th>Total count</th></tr></thead>"
            "<tbody>%s<tr class='grand-total'><td>All classes</td><td>%d</td></tr></tbody>"
            "</table>"
            % (len(class_totals), class_total_rows, class_grand_total)
        )
    else:
        class_totals_section = (
            "<h2>Class totals <span class='count neutral'>(0 classes)</span></h2>"
            "<p class='desc'>Summed detection count per class across all saved frames.</p>"
            "<p class='ok'>No class counts recorded.</p>"
        )

    low_conf_section = section(
        "Low-confidence detections",
        "Confidence below %.2f -- verify these against the saved frame." % REPORT_LOW_CONFIDENCE,
        [row_line(row) for row in flags["low_confidence"]],
    )
    missing_gnss_section = section(
        "Frames saved without a GNSS fix",
        "No latitude/longitude recorded -- location can't be traced back for these.",
        [row_line(row) for row in flags["missing_gnss"]],
    )
    near_dup_section = section(
        "Near-duplicate captures",
        "Less than %.2f m of travel since the previous save -- likely redundant." % REPORT_NEAR_DUPLICATE_M,
        [
            row_line(row, "%.3f m from previous" % row["distance_from_prev_m"])
            for row in flags["near_duplicates"]
        ],
    )
    no_detection_lines = []
    for run in flags["no_detection_runs"]:
        no_detection_lines.append(
            "<tr><td colspan='5' class='run-header'>Run of %d consecutive frames with no detections "
            "(%s → %s)</td></tr>"
            % (len(run), esc(run[0]["recorded_at"]), esc(run[-1]["recorded_at"]))
        )
        no_detection_lines.extend(row_line(row) for row in run)
    no_detection_section = (
        "<h2>No-detection stretches <span class='count'>(%d runs)</span></h2>"
        "<p class='desc'>%d or more consecutive saved frames with zero objects detected -- "
        "possible coverage gap or missed target.</p>"
        % (len(flags["no_detection_runs"]), REPORT_NO_DETECTION_RUN_LENGTH)
    )
    if flags["no_detection_runs"]:
        no_detection_section += (
            "<table><thead><tr><th>Frame</th><th>Captured</th>"
            "<th>Confidence</th><th>GNSS</th><th>Note</th></tr></thead>"
            "<tbody>%s</tbody></table>" % "".join(no_detection_lines)
        )
    else:
        no_detection_section += "<p class='ok'>None found.</p>"

    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Data Quality Report</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;margin:32px auto;padding:0 16px;color:#1a1d24;background:#fff}
h1{margin-bottom:4px}
.meta{color:#666;font-size:13px;margin-bottom:28px}
.summary{display:flex;gap:16px;margin-bottom:32px;flex-wrap:wrap}
.stat{border:1px solid #ddd;border-radius:8px;padding:10px 16px;min-width:140px}
.stat .n{font-size:22px;font-weight:700}
.stat .l{font-size:12px;color:#666}
h2{margin-top:36px;border-top:1px solid #eee;padding-top:20px}
.count{color:#c0392b;font-weight:600}
.count.neutral{color:#3a5fc9}
.desc{color:#555;font-size:13px;margin:4px 0 12px}
.ok{color:#1e8e4e;font-weight:600}
table{width:100%%;border-collapse:collapse;font-size:12.5px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid #eee}
th{color:#666;font-weight:600}
.run-header{background:#fff6e5;font-weight:600;color:#8a5a00}
.grand-total td{border-top:2px solid #ccc;font-weight:700}
</style></head>
<body>
<h1>Data Quality Report</h1>
<div class="meta">Generated %s &middot; %d saved frames analyzed</div>
<div class="summary">
  <div class="stat"><div class="n">%d</div><div class="l">Distinct classes</div></div>
  <div class="stat"><div class="n">%d</div><div class="l">All-class total</div></div>
  <div class="stat"><div class="n">%d</div><div class="l">Low-confidence</div></div>
  <div class="stat"><div class="n">%d</div><div class="l">Missing GNSS</div></div>
  <div class="stat"><div class="n">%d</div><div class="l">Near-duplicates</div></div>
  <div class="stat"><div class="n">%d</div><div class="l">No-detection runs</div></div>
</div>
%s
%s
%s
%s
%s
</body></html>""" % (
        esc(generated_at), total_frames,
        len(class_totals), class_grand_total,
        len(flags["low_confidence"]), len(flags["missing_gnss"]),
        len(flags["near_duplicates"]), len(flags["no_detection_runs"]),
        class_totals_section, low_conf_section, missing_gnss_section,
        near_dup_section, no_detection_section,
    )


@app.route("/api/report")
def get_report():
    rows = _report_rows()
    if not rows:
        return jsonify({
            "success": False,
            "message": "no saved frames are available to report on",
        }), 404

    # Class ID -> human name is only known client-side (the dashboard's
    # "Class Labels" list, stored in localStorage) -- the frontend passes its
    # current mapping along so the report can show names instead of raw IDs.
    class_labels = {}
    try:
        parsed = json.loads(request.args.get("labels", "{}"))
        if isinstance(parsed, dict):
            class_labels = {str(k): str(v) for k, v in parsed.items()}
    except (TypeError, ValueError):
        class_labels = {}

    flags = _build_qa_report(rows)
    html = _report_html(rows, flags, class_labels)
    report_name = "quality_report_%s.html" % datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    return app.response_class(
        html, mimetype="text/html",
        headers={"Content-Disposition": "attachment; filename=%s" % report_name},
    )


@app.route("/api/images/download")
def download_images():
    archive_buffer = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)

    # Same client-side "Class Labels" mapping /api/report accepts (labels are
    # only known in the browser's localStorage) -- lets the manifest's column
    # headers and JSON blobs show names instead of raw class indices.
    class_labels = {}
    try:
        parsed = json.loads(request.args.get("labels", "{}"))
        if isinstance(parsed, dict):
            class_labels = {str(k): str(v) for k, v in parsed.items()}
    except (TypeError, ValueError):
        class_labels = {}

    def _class_column_name(class_id):
        name = class_labels.get(str(class_id))
        # CSV headers must stay stable/ASCII-safe identifiers, so a label is
        # appended alongside the index rather than replacing it outright.
        return "class_%s_%s" % (class_id, name) if name else "class_%s" % class_id

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

            # Parse each row's counts_json once up front so the CSV header can
            # include every class that appears across all images (each image
            # only has the classes it actually detected in its own JSON blob).
            counts_by_filename = {}
            all_class_ids = set()
            for image_path in image_paths:
                history = history_by_filename.get(image_path.name)
                if not history or not history["counts_json"]:
                    continue
                try:
                    counts = json.loads(history["counts_json"])
                except (TypeError, ValueError):
                    continue
                counts_by_filename[image_path.name] = counts
                all_class_ids.update(counts.keys())

            def _class_sort_key(class_id):
                try:
                    return (0, int(class_id))
                except ValueError:
                    return (1, class_id)

            sorted_class_ids = sorted(all_class_ids, key=_class_sort_key)
            class_columns = [_class_column_name(class_id) for class_id in sorted_class_ids]

            manifest_buffer = io.StringIO(newline="")
            fieldnames = [
                "filename",
                "file_size_bytes",
                "captured_at",
                "history_id",
                "detection_recorded_at",
                "class_counts_json",
                "class_names_json",
                *class_columns,
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
                    counts = counts_by_filename.get(image_path.name, {})
                    names_json = (
                        json.dumps({
                            class_id: class_labels[str(class_id)]
                            for class_id in counts
                            if str(class_id) in class_labels
                        })
                        if history else ""
                    )
                    row = {
                        "filename": image_path.name,
                        "file_size_bytes": file_stat.st_size,
                        "captured_at": datetime.datetime.fromtimestamp(
                            file_stat.st_mtime, datetime.timezone.utc
                        ).isoformat(),
                        "history_id": history["id"] if history else "",
                        "detection_recorded_at": history["recorded_at"] if history else "",
                        "class_counts_json": history["counts_json"] if history else "",
                        "class_names_json": names_json,
                        "total_objects": history["total"] if history else "",
                        "object_detected": history["bed_status"] if history else "",
                        "confidence": history["confidence"] if history else "",
                        "tracking_enabled": history["tracking"] if history else "",
                        "latitude": history["latitude"] if history else "",
                        "longitude": history["longitude"] if history else "",
                        "gnss_recorded_at": history["gnss_recorded_at"] if history else "",
                    }
                    for class_id, column in zip(sorted_class_ids, class_columns):
                        row[column] = counts.get(class_id, 0) if history else ""
                    writer.writerow(row)
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
