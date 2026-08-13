# Developer Manual — Sensor Fusion & Bed Detection Stack

Scope: the three services launched together on the Jetson (`run_ai.sh`, `aarch64` branch):

- `imu_gnss_raw-aarch64` — `rtk_localization` package, node `gnss_imu_eskf_node`
- `sensor_data_raw-aarch64` — `base_receiver` + `bwt901ble_imu` packages
- `yolov8-trt-bed-detect-orin` — `yolov8_trt_bed_detect_orin` package (C++ TensorRT node + Python web server)

All three ship in **two** Docker images:

| Image | Built from | Contains | Compose services |
|---|---|---|---|
| `race_nav:sf.humble.v3-aarch64` | `vision/sf/Dockerfile.sf.humble` | `bwt901ble_imu`, `prime_msgs`, `base_receiver`, `rtk_localization` | `imu_gnss_raw-aarch64`, `sensor_data_raw-aarch64` |
| `race_nav:yolov8-trt-bed-detect-orin-nano.v5` | `vision/ai/Dockerfile.yolov8_trt_bed_detect_jetson_orien_nano` | `yolov8_trt_bed_detect_orin` | `yolov8-trt-bed-detect-orin` |

`imu_gnss_raw-*` and `sensor_data_raw-*` are **the same image** — they only differ in which `ros2 launch` command runs (`command:` in `docker-compose.yml`).

The x86 siblings (`imu_gnss_raw-x86`, `sensor_data_raw-x86`, `yolov8-trt-bed-detect-x86-jazzy`) are dev/desktop equivalents built against ROS 2 **Jazzy** rather than **Humble**. Jazzy is deliberately used on x86 — mixing Humble and Jazzy DDS participants on one domain previously caused `rmw_cyclonedds` decode failures.

---

## 1. System architecture / data flow

```
┌─────────────────────────────┐
│ sensor_data_raw-aarch64     │
│  (base_receiver +           │
│   bwt901ble_imu)            │
│                              │
│  serial GNSS ──► /gnss/pvt  │──────┐
│  (JSON)      ──► /receiver/fix     │
│                              │      │
│  BLE IMU ──► /imu/data_raw  │      │
│  ──(complementary filter)──►│      │
│           /imu/data         │──┐   │
└──────────────────────────────┘  │   │
                                   ▼   ▼
                    ┌──────────────────────────────┐
                    │ imu_gnss_raw-aarch64          │
                    │  (rtk_localization /          │
                    │   gnss_imu_eskf_node)          │
                    │                                │
                    │  15-state ESKF fusion          │
                    │  ──► /gnss_imu_eskf/odom       │
                    │  ──► /gnss_imu_eskf/gnss_only_odom
                    │  ──► /gnss_imu_eskf/motion_state_raw_gnss
                    │  ──► /receiver/fix (republished)
                    │  ◄── /gnss_imu_eskf/motion_pos_deadband
                    └───────────────┬────────────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │ yolov8-trt-bed-detect-orin    │
                    │  (web_server.py BridgeNode +  │
                    │   on-demand TensorRT node)    │
                    │                                │
                    │  camera ──► detections         │
                    │  GNSS/motion context ──► dashboard,
                    │  auto-save gating              │
                    │  Dashboard :8090, MJPEG :8080  │
                    └────────────────────────────────┘
```

**Chain summary:** `sensor_data_raw` (serial GNSS + BLE IMU) → `imu_gnss_raw` (ESKF fusion) → `yolov8-trt-bed-detect-orin` (dashboard / auto-save context). The three containers only talk to each other over ROS 2 topics (DDS on loopback, `network_mode: host`) — there is no direct process/API coupling between them except the shared `docker.sock` used for model conversion.

---

## 2. Service: `imu_gnss_raw-aarch64` (`rtk_localization`)

### 2.1 Launch
`ros2 launch rtk_localization gnss_imu_eskf.launch.py` → starts a single node, `gnss_imu_eskf_node`, parameterized from `config/gnss_imu_eskf.yaml` (launch arg `config`, overridable).

This launch file assumes `/imu/data` and `/gnss/pvt` are already being published by `sensor_data_raw-*` — it does not start any driver itself.

### 2.2 Node: `gnss_imu_eskf_node`

**Subscriptions**

| Topic | Type | QoS | Purpose |
|---|---|---|---|
| `/imu/data` | `sensor_msgs/Imu` | SensorData | Strapdown prediction, ~100 Hz |
| `/gnss/pvt` | `std_msgs/String` (JSON) | SystemDefaults | Position update |
| `/gnss/rtk_status` | `std_msgs/Bool` | SystemDefaults | RTK-active flag (logged/state only) |
| `/gnss_imu_eskf/motion_pos_deadband` (param `motion_pos_deadband_topic`) | `std_msgs/Float32` | depth 10 | Runtime deadband update; rejects ≤0 or non-finite |

**Publications**

| Topic | Type | Notes |
|---|---|---|
| `/receiver/fix` | `sensor_msgs/NavSatFix` | Republishes every accepted `/gnss/pvt` fix |
| `/gnss_imu_eskf/odom` | `nav_msgs/Odometry` | Fused ESKF output, `map`→`gnss_base_link` |
| `/gnss_imu_eskf/path` | `nav_msgs/Path` | Fused path trail |
| `/gnss_imu_eskf/raw_gnss_markers` | `visualization_msgs/MarkerArray` | RViz marker per accepted raw fix |
| `/gnss_imu_eskf/gnss_only_path` | `nav_msgs/Path` | Raw-GNSS-only path (no IMU) |
| `/gnss_imu_eskf/imu_only_path` | `nav_msgs/Path` | IMU-only dead-reckoning debug path |
| `/gnss_imu_eskf/gnss_only_odom` | `nav_msgs/Odometry` | Position = raw fix, orientation = course heading |
| `/gnss_imu_eskf/imu_only_odom` | `nav_msgs/Odometry` | Position = raw fix, orientation = IMU-only dead reckoning |
| `/gnss_imu_eskf/motion_state` | `std_msgs/String` | `"idle"` / `"forward"` / `"backward"`, fused |
| `/gnss_imu_eskf/motion_state_raw_gnss` | `std_msgs/String` | Same classification from raw GNSS only |
| TF `map → gnss_base_link` | — | Broadcast per IMU tick once origin is set, if `publish_tf` |

No services or actions.

**Parameters** (declared in `gnss_imu_eskf_node.cpp`)

| Parameter | Default | Meaning |
|---|---|---|
| `map_frame` | `map` | Parent TF frame |
| `base_frame` | `gnss_base_link` | Child TF frame |
| `min_fix_type` | `3` | Minimum `/gnss/pvt` `fix` value accepted |
| `acc_noise_density` | `0.05` | m/s²/√Hz |
| `gyro_noise_density` | `0.005` | rad/s/√Hz |
| `acc_bias_rw` | `0.001` | Accel bias random walk |
| `gyro_bias_rw` | `0.0001` | Gyro bias random walk |
| `publish_tf` | `true` | Broadcast TF |
| `gnss_lever_arm` | `[0,0,0]` | Antenna offset from body origin (m) |
| `default_hacc` / `default_vacc` | `1.0` / `2.0` | Fallback accuracy (m) when receiver omits it |
| `min_heading_dist` | `0.1` | Min displacement (m) before recomputing course heading |
| `gnss_heading_std_deg` | `15.0` (code default) | Heading measurement std — **confirm against runtime `ros2 param list`, see §6** |
| `motion_pos_deadband_topic` | `/gnss_imu_eskf/motion_pos_deadband` | Runtime deadband update topic |
| `motion_pos_deadband` | `0.3` | m |
| `motion_idle_hold_sec` | `1.0` | s — hysteresis hold before reverting to idle |

### 2.3 Algorithm notes
15-state ESKF: nominal state `[p, v, q, ab, gb]`, quaternion strapdown integration each IMU tick (midpoint), Joseph-form covariance correction on GNSS position updates and single-antenna course-heading updates. Initialization requires 100 stationary IMU samples to estimate gyro/accel bias and initial attitude (falls back to gravity alignment if no IMU orientation is available). Position integration doesn't start until the first accepted GNSS fix latches the ENU origin.

Motion-state classification uses a hysteresis state machine: enters `forward`/`backward` only after `motion_pos_deadband` meters of displacement from a stationary anchor, and only reverts to `idle` after `motion_idle_hold_sec` seconds within the deadband of a fixed reference. The `_raw_gnss` variant runs identical logic from consecutive `/gnss/pvt` fixes alone (no IMU) and can diverge from the fused result.

### 2.4 Environment
No environment variables — configured purely via ROS parameters and launch args.

---

## 3. Service: `sensor_data_raw-aarch64` (`base_receiver` + `bwt901ble_imu`)

### 3.1 Launch
`ros2 launch base_receiver imu_gps_raw.launch.py`, which:
1. Includes `base_receiver/launch/receiver.launch.py` (serial GNSS bridge)
2. Starts `bwt901ble_imu`/`imu_publisher` (BLE IMU driver)
3. Starts `imu_complementary_filter`/`complementary_filter_node` (fuses accel+gyro → orientation)
4. Has an **RTK localization node inclusion commented out** — not started here; that's the separate `imu_gnss_raw-*` service
5. Has `imu_filter_madgwick` commented out (disabled by default)

Launch args: `device_address` (default `F7:75:A3:1D:9C:AB`), `device_name_filter` (`WT`), `topic` (`imu/data_raw`), `mag_topic` (`imu/mag`), `frame_id` (`imu_link`), `publish_tf` (`false`), `tf_parent_frame` (`imu_reference`), `magnetic_declination_radians` / `yaw_offset_radians` (`0.0`).

### 3.2 Node: `base_receiver` (class `RoverGnssNode`)
Reads newline-delimited JSON from a serial GNSS receiver in a background thread, decoupled from ROS publishing via a bounded queue drained by a 50 Hz timer.

**Parameters**: `port` (default `""` → USB auto-detect), `baud` (`115200`), `frame_id` (`gps`), `stale_timeout` (`3.0`), `reconnect_interval` (`2.0`), `fix_topic` (`/receiver/fix`), `pvt_topic` (`/gnss/pvt`), `rtk_status_topic` (`/gnss/rtk_status`).

Deployed `config/receiver.yaml` pins `port: "/dev/gnss_rtk"` (see §7, udev rule).

**Publications**

| Topic | Type | Description |
|---|---|---|
| `/receiver/fix` | `sensor_msgs/NavSatFix` | lat/lon/alt/status/covariance |
| `/gnss/pvt` | `std_msgs/String` | Enriched PVT JSON (`rtkState`, `fixLabel`, `corrAgeLabel`, `timestamp`) |
| `/gnss/rtk_status` | `std_msgs/Bool` | `true` iff `rtkState` ∈ {RTK_FIXED, RTK_FLOAT} |

Auto-detected USB VID/PIDs: `(0x10C4,0xEA60)` CP210x, `(0x1A86,0x7523)` CH340, `(0x0403,0x6001)` FTDI, `(0x303A,0x1001)` Espressif USB.

`fix` codes: `0=NO_FIX 1=DEAD_RECKONING 2=2D 3=3D 4=GNSS_DR 5=TIME_ONLY` (valid for position: 2/3/4). `carr`: `0=none 1=RTK_FLOAT 2=RTK_FIXED`.

### 3.3 Node: `gnss_enu` (class `GnssEnuNode`)
Subscribes `NavSatFix` on `fix_topic` (default `/receiver/fix`), publishes `geometry_msgs/PointStamped` on `output_topic` (default `/gps/enu_position`). Origin latched on first valid fix and held until node restart. `x=East, y=North, z=Up`.

### 3.4 Node: `bwt901ble_imu`/`imu_publisher`
Connects to a BlueLink WT-series BLE IMU (default MAC `F7:75:A3:1D:9C:AB`) via `bleak` over the host's D-Bus/BlueZ socket. Publishes `sensor_msgs/Imu` on `topic` (`/imu/data_raw` as launched) and `sensor_msgs/MagneticField` on `mag_topic` (`/imu/mag`). This is why `/run/dbus/system_bus_socket` is mounted into the container.

### 3.5 Environment
Pure ROS params — no environment variables consumed by `base_receiver` or `bwt901ble_imu` source.

---

## 4. Service: `yolov8-trt-bed-detect-orin`

### 4.1 Launch — important: detection is NOT running by default
`ros2 launch yolov8_trt_bed_detect_orin yolov8_web.launch.py` starts **only** the Python web server (`web_server.py`). It does **not** start the TensorRT/ROS C++ detection node.

The detection node is started **on demand** by the web server via `POST /api/deserialize/model`, which runs `ros2 launch yolov8_trt_bed_detect_orin bed_detect.launch.py` as a background subprocess. Engine serialization similarly runs `serialize_engine.launch.py` via `POST /api/serialize/model`.

### 4.2 Node: `yolov8_trt` (executable `yolov8_trt_bed_detect_orin`, C++)

**Parameters**

| Parameter | Default | Meaning |
|---|---|---|
| `engine_name` | `yolov8n.engine` | Path to serialized TensorRT engine |
| `wts_name` | `yolov8n.wts` | Serialize-mode input weights |
| `model_type` | `n` | YOLOv8 size letter |
| `input_h` / `input_w` | `416` / `416` | Inference resolution (baked into engine) |
| `num_class` | `80` | Class count |
| `precision` | `fp16` | `fp16` or `int8` |
| `cuda_post_process` | `g` | `g`=GPU NMS, `c`=CPU NMS |
| `conf_thresh` | `0.5` | Detection confidence threshold |
| `nms_thresh` | `0.45` | NMS IoU threshold |
| `max_output_bbox` | `100` | Postprocess box cap |
| `mjpeg_port` | `8080` | Raw-socket MJPEG streamer port |
| `camera_index` | `0` | `-1` = auto-scan `/dev/video0..9` |
| `camera_width` / `camera_height` | `1280` / `720` | Capture resolution |
| `conf_score_value` | `0.8` | Gate for `bed_detection_status=1` |
| `is_track` | `false` | Cumulative unique-object tracking (MOSSE) vs per-frame counts |

**⚠ Deployed config caveat** — `config/trt_params.yaml` (mounted for both aarch64 and x86 services) defaults `engine_name` to `/weights/yolov8n_x86_64.engine`, while `config/trt_prams_main.yaml` (note the misspelling) defaults to the aarch64 engine. Both `bed_detect.launch.py` and `yolov8_web.launch.py` default their `params_file` launch arg to `trt_params.yaml`. **Confirm which engine file is actually loaded on the Orin before relying on default behavior** — this looks like a real misconfiguration risk, not just a doc inconsistency.

**Topics**

| Topic | Type | QoS | Direction |
|---|---|---|---|
| `conf` | `std_msgs/Float32` | depth 10 | Pub — max confidence this frame |
| `bed_detection_status` | `std_msgs/UInt8` | depth 10 | Pub — 1 if any detection's conf > `conf_score_value` |
| `class_counts` | `std_msgs/String` | depth 10 | Pub — e.g. `"class0:2 class1:1"` |
| `detection_active` | `std_msgs/UInt8` | reliable + transient_local, depth 1 | Pub — 1 = inference loop running |
| `tracking_enabled` | `std_msgs/UInt8` | reliable + transient_local, depth 1 | Pub — 1 = unique-tracking mode |
| `camera_index` | `std_msgs/Int32` | reliable + transient_local, depth 1 | Pub — actually-opened camera index |
| `infer_ms` | `std_msgs/Float32` | depth 10 | Pub — inference latency per frame |

**Services**

| Service | Type | Behavior |
|---|---|---|
| `bed_detection` | `std_srvs/Trigger` | Enable detection, reset tracker, `detection_active=1` |
| `bed_detection_stop` | `std_srvs/Trigger` | Disable detection, zero out status/conf/counts |
| `reset_tracker` | `std_srvs/Trigger` | Clear cumulative MOSSE tracker counts |
| `set_tracking` | `std_srvs/SetBool` | Set `is_track` param at runtime |

MJPEG streaming is a custom raw-socket HTTP server (`include/mjpeg_server.h`), not Flask-based — non-blocking, skips JPEG encode if no clients connected.

### 4.3 Web server (`web_server/web_server.py`) — the actual container entrypoint

Full REST/WebSocket API is documented separately in [api-reference.md](api-reference.md).

**Environment variables** (set by `yolov8_web.launch.py`, sourced from `docker-compose.yml` `environment:`)

| Variable | Default | Purpose |
|---|---|---|
| `SAVE_DIR` | `/saved_frames` | JPEG + SQLite DB storage |
| `WEIGHTS_DIR` | `/weights` | Model file directory |
| `CONVERT_IMAGE` | `meraquetech/tensorrt-yolov8:ultralytics` | Docker image for `.pt`→`.wts` (x86 only) |
| `HOST_WEIGHTS_DIR` | — | Host path for weights (conversion talks to host daemon via `docker.sock`) |
| `HOST_YOLOV8_DIR` | — | Host path to `yolov8/` source |
| `TRT_PARAMS_FILE` | `.../config/trt_params.yaml` | File the web server edits in place for model selection |
| `MJPEG_PORT` | `8080` | Port the C++ node's MJPEG server listens on |
| `API_PORT` | `8090` | Flask bind port |
| `HISTORY_DB` | `<SAVE_DIR>/count_history.db` | SQLite path |
| `AUTO_SAVE_INTERVAL` | `0.5` (floor) | Auto-save loop wake interval, seconds |
| `GNSS_FIX_TOPIC` | `/receiver/fix` | NavSatFix source for dashboard lat/lon |
| `GNSS_STALE_TIMEOUT` | `3.0` | Seconds before GNSS considered stale |
| `MOTION_STATE_TOPIC` | `/gnss_imu_eskf/motion_state_raw_gnss` | Direction subscription |
| `MOTION_POS_DEADBAND_TOPIC` | `/gnss_imu_eskf/motion_pos_deadband` | Publishes deadband updates back to ESKF node |
| `MOTION_POS_DEADBAND_M` | `0.3` | Initial deadband |
| `REPORT_LOW_CONFIDENCE` | `0.5` | QA report threshold |
| `REPORT_NEAR_DUPLICATE_M` | `= MOTION_POS_DEADBAND_M` | QA report near-duplicate distance |
| `REPORT_NO_DETECTION_RUN_LENGTH` | `3` | QA report min flagged gap run length |
| `GNSS_ONLY_ODOM_TOPIC` | `/gnss_imu_eskf/gnss_only_odom` | ENU source for auto-save distance gating |
| `DIRECTION_STALE_TIMEOUT` | `3.0` | Seconds before direction data considered stale |

`HOST_COMPOSE_FILE` and `SERIALIZE_SERVICE_NAME` are read but not set in the current `docker-compose.yml` — likely vestigial from an earlier compose-based serialize workflow.

### 4.4 Closed-loop GNSS/motion integration
1. `imu_gnss_raw` publishes `/receiver/fix`, `/gnss_imu_eskf/motion_state_raw_gnss`, `/gnss_imu_eskf/gnss_only_odom`; subscribes `/gnss_imu_eskf/motion_pos_deadband`.
2. `web_server.py`'s `BridgeNode` subscribes to those same three topics (via the env-var-configured names) and publishes back to the deadband topic when the dashboard's deadband control (`POST /api/motion_pos_deadband`) is used.
3. Auto-save gating requires `is_forward=true AND is_backward=false` (fresh, non-stale) **and** ≥ `motion_pos_deadband_m` displacement since the last save (Euclidean distance in ENU from `gnss_only_odom`).

### 4.5 Volumes enabling live edits
`docker-compose.yml` bind-mounts `config/`, `launch/`, and `web_server/` directly over the installed share-directory paths inside the container. Host edits to these directories take effect without rebuilding the image — this is also why the web server's in-place YAML edits (`_update_trt_params`) persist across container restarts.

### 4.6 Privileges
`pid: host` + `privileged: true` + `/var/run/docker.sock` mounted grant this container effective root control of the host: `POST /api/shutdown` `nsenter`s into host PID 1 to shut down the physical Jetson, `.pt→.wts` conversion runs `docker run` against the host daemon.

---

## 5. Build

```bash
# aarch64 (Jetson)
docker compose build imu_gnss_raw-aarch64 yolov8-trt-bed-detect-orin sensor_data_raw-aarch64

# x86 (desktop/dev)
docker compose build imu_gnss_raw-x86 yolov8-trt-bed-detect-x86-jazzy sensor_data_raw-x86
```

`run_ai.sh` wraps this per-architecture (`uname -m` dispatch) and additionally calls `./kill_server.sh` first, which force-removes **all** running containers (`docker rm -f $(docker ps -aq)`) — not scoped to this project.

**Asymmetry to be aware of**: the x86 branch of `run_ai.sh` builds `sensor_data_raw-x86` but does not `up` it (that line is commented out); the aarch64 branch starts all three services.

---

## 6. Known drift / gotchas for developers

1. **`gnss_heading_std_deg`**: code default `15.0`, not set in deployed `config/gnss_imu_eskf.yaml`, and rtk_localization's own `docs/readme.md` describes a different value/range. Verify with `ros2 param get /gnss_imu_eskf_node gnss_heading_std_deg` before treating any single number as authoritative.
2. Several parameters present in `config/gnss_imu_eskf.yaml` (`min_carr_soln`, `max_hacc`, `max_vacc`, `position_nis_gate`, `heading_nis_gate`, `resolve_reverse_motion`, `stationary_init_samples`, `initial_yaw_std_deg`, etc.) are **not** `declare_parameter`'d in `gnss_imu_eskf_node.cpp` and are silently ignored by rclcpp unless the node allows undeclared parameters (it does not). Confirm live with `ros2 param list /gnss_imu_eskf_node`.
3. `bwt901ble_imu/README.md` describes an older/aspirational architecture (topics `/gnss/odom`, `/gnss/is_forward`, `/gnss/is_backward`, a node `/rtk_localization` included by `imu_gps_raw.launch.py`) that doesn't match the current wiring (RTK inclusion is commented out there; the real ESKF node lives in the separate `imu_gnss_raw` service and uses `/gnss_imu_eskf/*` topic names).
4. `yolov8_trt_bed_detect_orin/README.md`'s "Prerequisites" section (Jetson Nano, JetPack 4.6.1, L4T r32.7) contradicts the actual Dockerfile chain (Jetson **Orin** Nano, JetPack 6.x, L4T r36.4.0, `sm_87`) — likely copied from the non-Orin sibling package without updating.
5. `config/trt_params.yaml` vs `config/trt_prams_main.yaml` engine-path defaults appear swapped relative to their intended architecture — see §4.2.

---

## 7. Physical device pinning

GNSS receiver is a CH340 USB-serial adapter pinned to a fixed device symlink so USB enumeration order can't swap it:

```
# udev/gnss_rtk.rules
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", KERNELS=="1-2.3", MODE:="0777", SYMLINK+="gnss_rtk"
```

`KERNELS=="1-2.3"` is the physical USB port path (from `udevadm info -a -n /dev/ttyUSBx`). If the receiver moves to a different USB port, regenerate this rule. Install/reload via `udev/install.sh`.

`base_receiver/config/receiver.yaml` points `port: "/dev/gnss_rtk"` at this symlink.
