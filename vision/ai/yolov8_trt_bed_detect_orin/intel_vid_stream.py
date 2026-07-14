#!/usr/bin/env python3
"""Web MJPEG stream for a user-selectable cv2.VideoCapture index."""

# pip install flask opencv-python

import time
import threading

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

DEFAULT_CAMERA_INDEX = 6
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_FPS = 30
HTTP_PORT = 8081

app = Flask(__name__)


class CameraManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._camera = None
        self._index = None

    def switch(self, index):
        with self._lock:
            if self._index == index and self._camera is not None and self._camera.isOpened():
                return True, None

            new_camera = cv2.VideoCapture(index)
            if not new_camera.isOpened():
                new_camera.release()
                return False, "Could not open camera %d" % index

            new_camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            new_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            new_camera.set(cv2.CAP_PROP_FPS, FRAME_FPS)

            old_camera = self._camera
            self._camera = new_camera
            self._index = index
            if old_camera is not None:
                old_camera.release()
            return True, None

    def current_index(self):
        with self._lock:
            return self._index

    def read(self):
        with self._lock:
            if self._camera is None:
                return None
            ok, frame = self._camera.read()
        return frame if ok else None


camera_manager = CameraManager()


def _mjpeg_generator():
    while True:
        frame = camera_manager.read()
        if frame is None:
            time.sleep(0.1)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )


PAGE = """
<!doctype html>
<html>
<head>
<title>Camera Stream</title>
<style>
  body { font-family: sans-serif; background: #111; color: #eee; text-align: center; }
  #stream { max-width: 90vw; border: 2px solid #444; margin-top: 1em; }
  #status { margin-top: 0.5em; min-height: 1.2em; }
  input, button { font-size: 1em; padding: 0.3em; }
</style>
</head>
<body>
  <h2>Camera Stream</h2>
  <div>
    Camera index:
    <input id="camIndex" type="number" min="0" value="{{ current_index }}" style="width: 4em;">
    <button onclick="connect()">Connect</button>
  </div>
  <div id="status"></div>
  <img id="stream" src="/video_feed">

<script>
function connect() {
  const index = parseInt(document.getElementById("camIndex").value, 10);
  const status = document.getElementById("status");
  status.textContent = "Connecting to camera " + index + "...";
  fetch("/api/camera", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({index: index}),
  })
    .then((response) => response.json())
    .then((data) => {
      status.textContent = data.message || (data.success ? "Connected" : "Failed");
      if (data.success) {
        document.getElementById("stream").src = "/video_feed?ts=" + Date.now();
      }
    })
    .catch((error) => {
      status.textContent = "Request failed: " + error;
    });
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE, current_index=camera_manager.current_index() or 0)


@app.route("/api/camera", methods=["POST"])
def set_camera():
    data = request.get_json(silent=True) or {}
    try:
        index = int(data.get("index"))
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "index must be an integer"}), 400

    ok, message = camera_manager.switch(index)
    if not ok:
        return jsonify({"success": False, "message": message}), 500
    return jsonify({"success": True, "message": "connected to camera %d" % index, "index": index})


@app.route("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    ok, message = camera_manager.switch(DEFAULT_CAMERA_INDEX)
    if not ok:
        print(message)
    app.run(host="0.0.0.0", port=HTTP_PORT, threaded=True)
