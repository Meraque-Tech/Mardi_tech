#!/usr/bin/env python3
"""
YOLOv8 TRT Bed Detect — Web API + WebSocket Server
Bridges ROS 2 topics/services to a browser dashboard.

Endpoints:
  GET  /                      → dashboard UI
  WS   /ws                    → real-time class counts (JSON)
  GET  /api/counts            → current counts snapshot (JSON)
  GET  /api/status            → node status (JSON)
  POST /api/start             → start bed detection (calls ROS service)
  POST /api/stop              → stop bed detection
  POST /api/save              → save current annotated frame to disk
  GET  /api/images            → list saved images (JSON)
  GET  /api/images/{filename} → serve a saved image
"""

import asyncio
import datetime
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, UInt8, Float32
from std_srvs.srv import Trigger

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR   = os.environ.get("SAVE_DIR",   "/saved_frames")
MJPEG_PORT = int(os.environ.get("MJPEG_PORT", "8080"))
API_PORT   = int(os.environ.get("API_PORT",   "8090"))
STATIC_DIR = Path(__file__).parent / "static"

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Shared state ──────────────────────────────────────────────────────────────
state: Dict = {
    "counts":       {},          # {"0": 2, "1": 1, ...}
    "bed_status":   0,           # 0 or 1
    "conf":         0.0,
    "detecting":    False,
    "last_updated": None,
}
state_lock = threading.Lock()

ws_clients: List[WebSocket] = []
ws_lock = asyncio.Lock()

latest_frame: np.ndarray | None = None
frame_lock = threading.Lock()


# ── ROS 2 node ────────────────────────────────────────────────────────────────
class BridgeNode(Node):
    def __init__(self):
        super().__init__("web_bridge")
        self.create_subscription(String,  "/class_counts",         self._counts_cb,     10)
        self.create_subscription(UInt8,   "/bed_detection_status", self._status_cb,     10)
        self.create_subscription(Float32, "/conf",                  self._conf_cb,       10)
        self._start_cli = self.create_client(Trigger, "/bed_detection")

    def _counts_cb(self, msg: String):
        global state
        counts = {}
        for part in msg.data.strip().split():
            if ":" in part:
                k, v = part.split(":")
                counts[k.replace("class", "")] = int(v)
        with state_lock:
            state["counts"]       = counts
            state["last_updated"] = datetime.datetime.now().isoformat()
        asyncio.run_coroutine_threadsafe(_broadcast_counts(), loop)

    def _status_cb(self, msg: UInt8):
        with state_lock:
            state["bed_status"] = int(msg.data)

    def _conf_cb(self, msg: Float32):
        with state_lock:
            state["conf"] = round(float(msg.data), 4)

    def call_start(self):
        if not self._start_cli.wait_for_service(timeout_sec=2.0):
            return False, "service not available"
        req = Trigger.Request()
        future = self._start_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)
        if future.result():
            with state_lock:
                state["detecting"] = True
            return True, future.result().message
        return False, "no response"


ros_node: BridgeNode | None = None
loop: asyncio.AbstractEventLoop | None = None


def _ros_spin():
    rclpy.init(args=None)
    global ros_node
    ros_node = BridgeNode()
    rclpy.spin(ros_node)
    ros_node.destroy_node()
    rclpy.shutdown()


async def _broadcast_counts():
    with state_lock:
        payload = json.dumps({
            "type":    "counts",
            "counts":  state["counts"],
            "bed":     state["bed_status"],
            "conf":    state["conf"],
            "time":    state["last_updated"],
        })
    async with ws_lock:
        dead = []
        for ws in ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)


# ── Snapshot helper ───────────────────────────────────────────────────────────
def _grab_mjpeg_frame() -> np.ndarray | None:
    """Grab a single frame from the local MJPEG stream."""
    try:
        cap = cv2.VideoCapture(f"http://127.0.0.1:{MJPEG_PORT}/")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except Exception:
        return None


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="YOLOv8 TRT Bed Detect API")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/saved",  StaticFiles(directory=SAVE_DIR), name="saved")


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    index = STATIC_DIR / "index.html"
    return HTMLResponse(index.read_text())


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    async with ws_lock:
        ws_clients.append(ws)
    try:
        # send current state immediately on connect
        with state_lock:
            snap = dict(state)
        await ws.send_text(json.dumps({
            "type":   "counts",
            "counts": snap["counts"],
            "bed":    snap["bed_status"],
            "conf":   snap["conf"],
            "time":   snap["last_updated"],
        }))
        while True:
            await ws.receive_text()   # keep alive; client may send pings
    except WebSocketDisconnect:
        pass
    finally:
        async with ws_lock:
            if ws in ws_clients:
                ws_clients.remove(ws)


@app.get("/api/counts")
async def get_counts():
    with state_lock:
        return JSONResponse(state["counts"])


@app.get("/api/status")
async def get_status():
    with state_lock:
        return JSONResponse({
            "detecting":    state["detecting"],
            "bed_status":   state["bed_status"],
            "conf":         state["conf"],
            "last_updated": state["last_updated"],
            "mjpeg_url":    f"http://{{host}}:{MJPEG_PORT}/",
        })


@app.post("/api/start")
async def start_detection():
    if ros_node is None:
        raise HTTPException(503, "ROS node not ready")
    ok, msg = await asyncio.get_event_loop().run_in_executor(None, ros_node.call_start)
    if ok:
        return JSONResponse({"success": True, "message": msg})
    raise HTTPException(500, msg)


@app.post("/api/stop")
async def stop_detection():
    with state_lock:
        state["detecting"] = False
    # No ROS stop service exists yet — just flip the flag for UI feedback
    return JSONResponse({"success": True, "message": "detection stopped (UI only)"})


@app.post("/api/save")
async def save_frame():
    frame = await asyncio.get_event_loop().run_in_executor(None, _grab_mjpeg_frame)
    if frame is None:
        raise HTTPException(500, "Could not grab frame from MJPEG stream")
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"frame_{ts}.jpg"
    path     = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return JSONResponse({"success": True, "filename": filename, "path": path})


@app.get("/api/images")
async def list_images():
    files = sorted(Path(SAVE_DIR).glob("*.jpg"), reverse=True)
    return JSONResponse([
        {"filename": f.name, "url": f"/saved/{f.name}",
         "size": f.stat().st_size, "time": datetime.datetime.fromtimestamp(f.stat().st_mtime).isoformat()}
        for f in files
    ])


@app.delete("/api/images/{filename}")
async def delete_image(filename: str):
    path = Path(SAVE_DIR) / filename
    if not path.exists():
        raise HTTPException(404, "not found")
    path.unlink()
    return JSONResponse({"success": True})


@app.post("/api/reset_tracker")
async def reset_tracker():
    # Signal the ROS node via a param set (ros_node runs in separate thread)
    # The C++ node checks is_track; resetting is done by setting a transient param.
    # For now broadcast a reset event to connected WebSocket clients.
    async with ws_lock:
        for ws in ws_clients:
            try:
                await ws.send_text(json.dumps({"type": "tracker_reset"}))
            except Exception:
                pass
    return JSONResponse({"success": True})


class TrackBody(dict):
    pass

@app.post("/api/set_track")
async def set_track(body: dict):
    enabled = body.get("enabled", False)
    # ros2 param set via subprocess inside container
    import subprocess
    subprocess.Popen(
        ["ros2", "param", "set", "/yolov8_trt", "is_track", str(enabled).lower()],
        env={**os.environ, "ROS_DOMAIN_ID": os.environ.get("ROS_DOMAIN_ID", "0")}
    )
    return JSONResponse({"success": True, "is_track": enabled})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ros_thread = threading.Thread(target=_ros_spin, daemon=True)
    ros_thread.start()
    time.sleep(1.0)   # let ROS init before accepting requests

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    config = uvicorn.Config(app, host="0.0.0.0", port=API_PORT, loop="asyncio", log_level="info")
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
