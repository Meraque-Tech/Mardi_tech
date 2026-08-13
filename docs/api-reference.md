# API Reference — Bed Detection Dashboard (`yolov8-trt-bed-detect-orin`)

Base URL: `http://<jetson-ip>:8090`
Raw MJPEG stream (no auth, no JSON): `http://<jetson-ip>:8080`

All JSON error responses follow `{"success": false, "message": "..."}`. Common status codes: `400` invalid input, `404` not found, `409` conflict, `500` I/O/DB failure, `503` ROS bridge/service unavailable.

---

## Pages / static

| Method | Route | Description |
|---|---|---|
| GET | `/` | Dashboard HTML single-page app |
| GET | `/saved/<path:filename>` | Serve a saved JPEG frame |

## Live counts & history

| Method | Route | Description |
|---|---|---|
| GET | `/api/counts` | Current live per-class detection counts |
| GET | `/api/history` | Paginated saved-count history. Query params: `limit` (≤5000), `offset` |
| GET | `/api/path` | Recent ENU path-point trace for the UI map/plot |
| POST | `/api/save_count` | Save one snapshot of live counts to SQLite. `409` if no live count yet |
| GET | `/api/status` | Full live-state snapshot including `mjpeg_port` |

## Auto-save & motion tuning

| Method | Route | Body | Description |
|---|---|---|---|
| POST | `/api/auto_save` | `{"enabled": bool}` | Toggle forward-only auto-save |
| POST | `/api/motion_pos_deadband` | `{"deadband_m": float>0}` | Push new deadband to the ESKF node via ROS |

## Detection control

| Method | Route | Body | Description |
|---|---|---|---|
| POST | `/api/start` | — | Calls `bed_detection` service (Trigger); also runs `jetson_clocks` if on Jetson |
| POST | `/api/stop` | — | Calls `bed_detection_stop` service (Trigger) |
| POST | `/api/reset_tracker` | — | Calls `reset_tracker` service (Trigger) |
| POST | `/api/set_track` | `{"enabled": bool}` | Calls `set_tracking` service (SetBool) |
| POST | `/api/save` | — | Grab current MJPEG frame and save as JPEG immediately |

## Model management

| Method | Route | Description |
|---|---|---|
| POST | `/api/models/upload` | Multipart upload of a `.wts` file into `WEIGHTS_DIR` |
| POST | `/api/models/upload_pt` | Multipart upload of a `.pt` file |
| GET | `/api/models` | List `.pt`/`.wts`/`.engine` files; includes `conversion_supported` flag |
| GET | `/api/models/<kind>/<filename>/download` | Download a model file (`kind` ∈ `pt`, `wts`, `engine`) |
| DELETE | `/api/models/<kind>/<filename>` | Delete a model file |
| POST | `/api/models/convert` | `{"filename": "x.pt"}` — background `.pt`→`.wts` conversion via `docker run` (**x86 only**, 400 on Jetson) |
| GET | `/api/models/convert/status` | Poll conversion job status |
| POST | `/api/models/build_engine` | `{"filename","num_class"}` — writes `wts_name`/`engine_name`/`num_class` into `trt_params.yaml`. 400 if `num_class` missing/non-positive |
| POST | `/api/models/select_engine` | `{"filename","num_class"}` — writes `engine_name`/`num_class` only |
| POST | `/api/serialize/model` | Runs `serialize_engine.launch.py` in background (skips if engine file already exists) |
| POST | `/api/deserialize/model` | Runs `bed_detect.launch.py` in background (stops any running serialize first; restarts detection if already running) |
| GET | `/api/models/launch/status` | `{"running","mode","message","ok"}` |
| GET | `/api/models/launch/log` | Last 500 lines of the current/last `ros2 launch` subprocess output |

## Saved images & reports

| Method | Route | Description |
|---|---|---|
| GET | `/api/images` | List saved JPEGs with metadata |
| GET | `/api/report` | Generate and download an HTML QA report (low confidence, missing GNSS, near-duplicate, no-detection-run flags) |
| GET | `/api/images/download` | Stream a ZIP of all saved JPEGs + `images_manifest.csv` |
| DELETE | `/api/images/<filename>` | Delete one saved JPEG |
| DELETE | `/api/data` | **Destructive.** Disables auto-save, deletes all history rows and all saved JPEGs |

## System

| Method | Route | Body | Description |
|---|---|---|---|
| POST | `/api/shutdown` | `{"confirm": "shutdown"}` | `nsenter`s into host PID 1 and runs `shutdown -h now` on the **physical host** (works via `pid: host`) |

---

## WebSocket — `/ws`

On connect: server sends a full state snapshot (`{"type": "snapshot", ...}`).

Push events:
- `{"type": "status", ...}` — on detection/tracking/direction/camera-index change
- `{"type": "snapshot", ...}` — on every `/class_counts` update

Client should send a `ping` text frame roughly every 20s to keep the connection alive (server uses a 30s receive timeout).

**State snapshot fields**: `counts`, `bed_status`, `conf`, `detecting`, `is_track`, `auto_save`, `motion_pos_deadband_m`, `last_updated`, `camera_index`, `infer_ms`, `latitude`, `longitude`, `gnss_valid`, `gnss_received_at`, `is_forward`, `is_backward`, `direction_received_at`, plus derived `direction_valid`, `auto_save_direction_eligible`, `auto_save_distance_eligible`, `auto_save_active`.

---

## ROS 2 service proxying

These REST endpoints are thin wrappers around ROS 2 services exposed by the `yolov8_trt` detection node:

| REST endpoint | ROS service | ROS type |
|---|---|---|
| `POST /api/start` | `/bed_detection` | `std_srvs/Trigger` |
| `POST /api/stop` | `/bed_detection_stop` | `std_srvs/Trigger` |
| `POST /api/reset_tracker` | `/reset_tracker` | `std_srvs/Trigger` |
| `POST /api/set_track` | `/set_tracking` | `std_srvs/SetBool` |

If the detection node isn't running (before `/api/deserialize/model` has been called), these return `503`.

---

## curl examples

```bash
# Start detection
curl -X POST http://<jetson-ip>:8090/api/start

# Check live status
curl http://<jetson-ip>:8090/api/status

# Toggle auto-save on
curl -X POST http://<jetson-ip>:8090/api/auto_save \
  -H 'Content-Type: application/json' \
  -d '{"enabled": true}'

# Update motion deadband to 0.5m
curl -X POST http://<jetson-ip>:8090/api/motion_pos_deadband \
  -H 'Content-Type: application/json' \
  -d '{"deadband_m": 0.5}'

# Download a QA report
curl -o report.html http://<jetson-ip>:8090/api/report

# Download all saved frames as a zip
curl -o frames.zip http://<jetson-ip>:8090/api/images/download
```

---

## Related ROS 2 topics (not HTTP, but relevant to API consumers building their own tooling)

| Topic | Type | Source node |
|---|---|---|
| `/receiver/fix` | `sensor_msgs/NavSatFix` | `gnss_imu_eskf_node` (republished) |
| `/gnss_imu_eskf/gnss_only_odom` | `nav_msgs/Odometry` | `gnss_imu_eskf_node` |
| `/gnss_imu_eskf/motion_state_raw_gnss` | `std_msgs/String` | `gnss_imu_eskf_node` |
| `/gnss_imu_eskf/motion_pos_deadband` | `std_msgs/Float32` | written by dashboard, read by `gnss_imu_eskf_node` |
| `/conf`, `/bed_detection_status`, `/class_counts`, `/detection_active`, `/tracking_enabled`, `/camera_index`, `/infer_ms` | various | `yolov8_trt` detection node |

See [developer-manual.md](developer-manual.md) for full topic/parameter tables across all three services.
