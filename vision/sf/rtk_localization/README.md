# rtk_localization

ROS2 package for dual-antenna RTK-GPS localization with optional LIO fusion.

---

# Bag Play
```
  cd /media/dj/dj/bags/Mar_13_bag_plantation && ros2 bag play mar13_04 --clock

  ./run_rtk.sh
  ./run_fast_lio.sh
```

## Overview

Uses two RTK antennas (left / right) to compute:
- Accurate vehicle **heading** from the baseline vector between antennas
- **Global position** via UTM coordinates
- **Local odometry** relative to a map origin set on first fix

Optionally fuses with LiDAR-Inertial Odometry (LIO-SAM / DLIO) for drift-free localization in GPS-degraded environments.

---

## Executables

| Executable | Node Name | Purpose |
|---|---|---|
| `rtk_odom` | `dual_rtk_odom_node` | Full RTK odom with TF, path, markers, enable/disable service |
| `rtk_mid` | `rtk_mid_raw_node` | Midpoint NavSatFix + TF/odom/path/markers on `map → rtk_pub` |
| `rtk_mid_raw` | `rtk_mid_raw_node` | Midpoint NavSatFix only (no TF/odom) |
| `rtk_imu_sf` | `dual_rtk_heading_node` | RTK + IMU sensor fusion via ENU origin |
| `ekf_imu_gps_node` | — | EKF fusion of IMU and GPS |
| `rtk_lio_calibration` | — | Auto-calibrates yaw offset between RTK and LIO frames |
| `lio_odom_georef` | — | Georeferences LIO odometry to UTM/global frame |
| `gnss_pvt_enu_odom` | `gnss_pvt_enu_odom_node` | Single-antenna `/gnss/pvt` JSON fix → ENU odometry |
| `gnss_imu_eskf_node` | `gnss_imu_eskf_node` | Error-State Kalman Filter fusing IMU + `/gnss/pvt` → ENU odometry |

---

## Topics

### Subscribed

| Topic | Type | Description |
|---|---|---|
| `/rtk/left/fix` | `sensor_msgs/NavSatFix` | Left antenna RTK fix |
| `/rtk/right/fix` | `sensor_msgs/NavSatFix` | Right antenna RTK fix |
| `/odom` | `nav_msgs/Odometry` | LIO odometry input (rtk_imu_sf) |

### Published (rtk_mid)

| Topic | Type | Description |
|---|---|---|
| `/rtk/mid/fix` | `sensor_msgs/NavSatFix` | ECEF midpoint converted back to LLA |
| `/rtk/pub/odom` | `nav_msgs/Odometry` | Local odom, frame `map`, child `rtk_pub` |
| `/rtk/pub/path` | `nav_msgs/Path` | Trail of RTK midpoint poses in `map` frame |
| `/rtk/pub/points` | `visualization_msgs/Marker` | Point cloud trail for RViz |
| TF `map → rtk_pub` | — | Dynamic transform broadcast each fix |

### Published (rtk_odom)

| Topic | Type | Description |
|---|---|---|
| `/rtk/odom/fix` | `nav_msgs/Odometry` | Raw UTM fix with heading |
| `/rtk/odom` | `nav_msgs/Odometry` | Local odom relative to map origin |
| `/rtk/path` | `nav_msgs/Path` | Path trail |
| `/rtk/points` | `visualization_msgs/Marker` | RViz point trail |

### Subscribed / Published (gnss_pvt_enu_odom)

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/gnss/pvt` | `std_msgs/String` (JSON) | Sub | Single-antenna fix: `{"type":"pvt","lat":..,"lon":..,"alt":..,"fix":..,"sats":..,...}` |
| `/gnss/rtk_status` | `std_msgs/Bool` | Sub | `true` while base-station corrections are being received |
| `/gnss/odom` | `nav_msgs/Odometry` | Pub | Local ENU odometry, frame `map`, child `gnss_base_link` |
| `/gnss/path` | `nav_msgs/Path` | Pub | Trail of GNSS ENU poses in `map` frame |
| TF `map → gnss_base_link` | — | Pub | Dynamic transform broadcast each fix |

Orientation (yaw only) is derived from **consecutive ENU fixes**: `heading = atan2(current.n - prev.n, current.e - prev.e)`, only updated once the displacement from the last heading-fix exceeds `min_heading_dist` (default `0.1` m) to avoid noisy heading while stationary or moving slowly. Until the first such update, yaw is `0` with covariance `1e6` (unobserved); roll/pitch are always unobserved (`1e6`). Positional covariance scales with the reported `fix` type (RTK fixed < RTK float < plain 3D fix).

ENU origin is latched on the **first fix** with `fix >= min_fix_type` (default `3`).

### Subscribed / Published (gnss_imu_eskf_node)

| Topic | Type | Direction | Description |
|---|---|---|---|
| `/imu/data` | `sensor_msgs/Imu` | Sub | Raw IMU (linear acceleration + angular velocity), used for high-rate strapdown prediction |
| `/gnss/pvt` | `std_msgs/String` (JSON) | Sub | Same PVT payload as `gnss_pvt_enu_odom`; consumed as a position update |
| `/gnss/rtk_status` | `std_msgs/Bool` | Sub | `true` while base-station corrections are being received (logged only) |
| `/gnss_imu_eskf/motion_pos_deadband` | `std_msgs/Float32` | Sub | Runtime motion-classification deadband update in metres; must be finite and greater than zero |
| `/gnss_imu_eskf/odom` | `nav_msgs/Odometry` | Pub | Fused ENU odometry, frame `map`, child `gnss_base_link` |
| `/gnss_imu_eskf/path` | `nav_msgs/Path` | Pub | Trail of fused poses in `map` frame |
| TF `map → gnss_base_link` | — | Pub | Dynamic transform broadcast each IMU tick (once origin is set) |

**Filter**: Error-State Kalman Filter (ESKF). The nominal state (position, velocity, quaternion orientation, accel bias, gyro bias) is propagated every IMU sample via strapdown integration — quaternion-based, so there's no gimbal-lock singularity. A 15-dim error state `[dp, dv, dtheta, d_accel_bias, d_gyro_bias]` is corrected on each GNSS position update via a standard EKF measurement update, then injected back into the nominal state (quaternion multiply for the attitude correction, not additive). This is the standard formulation for GNSS/IMU fusion and avoids the approximate Euler-angle Jacobians used by `ekf_imu_gps_node`.

The node waits for the first IMU sample before consuming GNSS (to seed attitude/initialize prediction), and the first GNSS fix is consumed only to latch the ENU origin (like `gnss_pvt_enu_odom`), not applied as a position measurement. Odometry is published on every IMU tick once the origin is set. Position measurement covariance scales with the reported `fix` type (RTK fixed < RTK float < plain 3D fix), same table as `gnss_pvt_enu_odom`.

**ESKF data flow:**

```
                         ┌─────────────────────────────┐
 /imu/data  ───(~100Hz)─▶│  predict()                   │
 (accel, gyro)           │  strapdown integration:      │
                         │   p += v·dt + ½a·dt²         │
                         │   v += a·dt                  │
                         │   q  = q ⊗ dq(gyro·dt)        │──▶ nominal state
                         │  propagate error-state:      │    [p, v, q, ab, gb]
                         │   P = F·P·Fᵀ + Q             │         │
                         └───────────────┬──────────────┘         │
                                         │ every IMU tick          │
                                         ▼                         ▼
                              ┌────────────────────┐     publishState()
                              │   nominal state     │     → /gnss_imu_eskf/odom
                              │   (always current)  │     → /gnss_imu_eskf/path
                              └─────────┬───────────┘     → TF map→gnss_base_link
                                        ▲
                                        │ inject error state
                                        │ (quaternion ⊗ for attitude)
                              ┌─────────┴───────────┐
                              │  injectErrorState()  │
                              │   p  += dp           │
                              │   v  += dv           │
                              │   q   = q ⊗ dq(dθ)    │
                              │   ab += d(ab)         │
                              │   gb += d(gb)         │
                              └─────────▲─────────────┘
                                        │ dx = K·y
                              ┌─────────┴───────────┐
 /gnss/pvt  ───(~1-10Hz)────▶│  updatePosition()    │
 (lat/lon/alt JSON)          │   y = pos_meas - p    │
       │                     │   S = H·P·Hᵀ + R      │
       ▼                     │   K = P·Hᵀ·S⁻¹         │
 latlon_to_enu()             │   P = (I - K·H)·P     │
 (gps_utils, ENU              └───────────────────────┘
  origin latched on
  first accepted fix)
```

- **Prediction** (`predict`, driven by `/imu/data`): advances the 15-dim error-state covariance `P` using the Jacobian `F` and process noise `Q` (from `acc_noise_density`, `gyro_noise_density`, `acc_bias_rw`, `gyro_bias_rw`); nominal state is integrated directly (not part of the linear KF math).
- **Update** (`updatePosition`, driven by `/gnss/pvt`): standard 3-DOF position Kalman correction against the nominal position `p`, producing an error-state correction `dx`.
- **Injection** (`injectErrorState`): folds `dx` back into the nominal state — additive for `p, v, ab, gb`, quaternion-multiplicative for the attitude error `dθ` (via `smallAngleQuat`) — avoiding gimbal lock.

### Services

| Service | Type | Description |
|---|---|---|
| `/rtk/enable` | `std_srvs/SetBool` | Enable / disable RTK odom publishing |

---

## Heading Computation

Antennas are mounted **left–right** across the vehicle (perpendicular to forward direction).

```
Baseline vector:  dx = UTM_right.E − UTM_left.E
                  dy = UTM_right.N − UTM_left.N

baseline_yaw = atan2(dy, dx)          # angle of the baseline
vehicle_yaw  = baseline_yaw + π/2     # forward is perpendicular to baseline
yaw          = atan2(sin(yaw), cos(yaw))  # normalize to [−π, π]
```

If antennas were **front–back** instead, drop the `+ π/2` correction.

```
Antenna config (left-right):

         Forward →
              N
              ^
              |  [Left]──────[Right]
              |         ↑ baseline
    W <───────+───────> E
              |
```

---

## Coordinate Frames

```
earth (WGS84 lat/lon)
    │
    ▼ UTM / ECEF conversion
  map  (ENU, origin = first valid RTK fix)
    │
    └── rtk_pub / base_link  (vehicle position, from rtk_mid / rtk_odom)
```

Map origin is set once on the **first valid RTK fix** (STATUS_FIX or better). All poses are local offsets from that origin.

---

## Coordinate Convention

```
         Math Convention              Navigation Convention
         North (90°)                  North (0°)
               |                            |
 West (180°) --+-- East (0°)   West (270°) --+-- East (90°)
               |                            |
         South (270°)                 South (180°)
```

ROS uses **ENU** (East = +X, North = +Y), which matches the math convention.

---

## GPS Utility Library (`gps_utils`)

Supported conversions:

| From | To |
|---|---|
| Geodetic (lat/lon/alt) | ECEF |
| ECEF | Geodetic |
| Geodetic | ENU |
| ENU | Geodetic |
| ECEF | ENU |
| ENU | ECEF |
| ENU | AER |
| AER | ENU |
| ENU | NED |
| NED | ENU |
| NED | AER |
| AER | NED |
| Geodetic | UTM |
| UTM | Geodetic |

**Typical robotics pipeline:**
```cpp
gps.set_enu_origin(lat0, lon0, alt0);      // set once on first fix
ENU p   = gps.latlon_to_enu(lat, lon, alt);
AER tgt = gps.enu_to_aer(p);
```

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `actual_baseline` | `1.20` m | Physical distance between antennas |
| `baseline_tolerance` | `0.25` m | Max allowed baseline error before WARN |
| `motion_threshold_xy` | `0.3–0.5` m | Min movement before ENU origin is locked |

### gnss_pvt_enu_odom

| Parameter | Default | Description |
|---|---|---|
| `map_frame` | `map` | Parent frame for odometry/TF |
| `base_frame` | `gnss_base_link` | Child frame for odometry/TF |
| `min_fix_type` | `3` | Minimum `fix` value from `/gnss/pvt` accepted before publishing |
| `min_heading_dist` | `0.1` m | Minimum displacement between fixes before heading is recomputed |

### gnss_imu_eskf_node

| Parameter | Default | Description |
|---|---|---|
| `map_frame` | `map` | Parent frame for odometry/TF |
| `base_frame` | `gnss_base_link` | Child frame for odometry/TF |
| `min_fix_type` | `3` | Minimum `fix` value from `/gnss/pvt` accepted before publishing |
| `acc_noise_density` | `0.05` m/s²/√Hz | Accelerometer white-noise density (process noise) |
| `gyro_noise_density` | `0.005` rad/s/√Hz | Gyroscope white-noise density (process noise) |
| `acc_bias_rw` | `0.001` | Accelerometer bias random-walk rate |
| `gyro_bias_rw` | `0.0001` | Gyroscope bias random-walk rate |
| `publish_tf` | `true` | Whether to broadcast `map → base_frame` TF |
| `gnss_lever_arm` | `[0.0, 0.0, 0.0]` | GNSS antenna position in the body frame (IMU origin), meters `[x, y, z]`. Compensated for both static offset and rotation-induced displacement during turns. |
| `default_hacc` | `1.0` m | Fallback horizontal accuracy (1σ) if `/gnss/pvt` omits `hacc` |
| `default_vacc` | `2.0` m | Fallback vertical accuracy (1σ) if `/gnss/pvt` omits `vacc` |
| `motion_pos_deadband` | `0.3` m | Displacement threshold used by fused and raw-GNSS motion classification |
| `motion_pos_deadband_topic` | `/gnss_imu_eskf/motion_pos_deadband` | Runtime `std_msgs/Float32` update topic |

Tune `acc_noise_density` / `gyro_noise_density` / `*_bias_rw` against your IMU's datasheet noise specs before field use. **Measure and set `gnss_lever_arm`** if the GNSS antenna is not co-located with the IMU — otherwise turns will inject position error proportional to the offset.

**Initialization**: the node buffers the first 100 stationary IMU samples, averages them to estimate gyro bias and align roll/pitch to gravity (yaw is left unobserved until GNSS corrects it), and only then starts strapdown integration. The vehicle **must be stationary** during this window. Position/velocity integration does not start until the first accepted GNSS fix latches the ENU origin (`p`/`v` are reset to zero at that point) — this avoids accumulating drift in an unanchored frame.

**RTK quality**: `carr` (0=none, 1=float, 2=fixed) determines RTK state; the receiver-reported `hacc`/`vacc` (1σ, meters) are used directly as measurement covariance when present, falling back to `carr`-based buckets otherwise.

**GNSS timing caveat**: `/gnss/pvt` is an unstamped `std_msgs/String`; the correction is applied against whatever the nominal state is when the message is processed (no timestamp-based rollback/replay). This is adequate when GNSS latency is small/consistent relative to the IMU rate; if the upstream driver starts publishing a timestamp, replace this with a buffered rollback-and-replay update instead.

---

## Startup Sequence (Best Practice)

```
1. Start sensors: IMU → LiDAR → dual RTK
2. Wait for RTK = FIXED and LIO initialized
3. Keep robot STATIONARY
4. Both origins are set at the same moment:
       gps.set_enu_origin(lat0, lon0, alt0)   ← RTK
       LIO map origin = (0, 0, 0)              ← LIO
   → map_lio ≡ map_gps  (no offset needed)
```

If origins are set at different times or after movement, a frame offset `T_map_gps_to_map_lio` must be estimated (what `robot_localization` does internally).

---

## Build

```bash
cd agv_ws
colcon build --packages-select rtk_localization --merge-install
source install/setup.bash
```

GeographicLib is found from the system (`libGeographic`) or downloaded automatically via CMake `FetchContent`.

---

## Test / Debug

```bash
# Play a bag file
cd /media/dj/dj/bags/sf_feb4 && ros2 bag play feb_2_rtk --clock

# Inject a synthetic GPS fix
ros2 topic pub /rtk/left/fix sensor_msgs/msg/NavSatFix "{
  header: {frame_id: 'gps'},
  status: {status: 0, service: 1},
  latitude: 49.9101, longitude: 8.8999, altitude: 120.0
}" -1

# Check heading output
ros2 topic echo /rtk/pub/odom
ros2 topic echo /rtk/odom/fix
```

### gnss_pvt_enu_odom

```bash
ros2 run rtk_localization gnss_pvt_enu_odom

# Inject a synthetic PVT fix
ros2 topic pub /gnss/pvt std_msgs/msg/String \
  '{data: "{\"type\":\"pvt\",\"lat\":2.9562305,\"lon\":101.6519943,\"alt\":59.594,\"fix\":3,\"carr\":1,\"sats\":29,\"hacc\":0.059,\"vacc\":0.077,\"pdop\":1.69}"}' -1

ros2 topic pub /gnss/rtk_status std_msgs/msg/Bool "{data: true}" -1

ros2 topic echo /gnss/odom
```

### gnss_imu_eskf_node

```bash
ros2 run rtk_localization gnss_imu_eskf_node

# Requires /imu/data flowing first (real driver, or a synthetic single-shot for smoke-testing)
ros2 topic pub /imu/data sensor_msgs/msg/Imu "{
  header: {frame_id: 'imu_link'},
  linear_acceleration: {x: 0.0, y: 0.0, z: 9.80665},
  angular_velocity: {x: 0.0, y: 0.0, z: 0.0}
}" -r 50

# Then inject GNSS PVT fixes (same payload as gnss_pvt_enu_odom)
ros2 topic pub /gnss/pvt std_msgs/msg/String \
  '{data: "{\"type\":\"pvt\",\"lat\":2.9562305,\"lon\":101.6519943,\"alt\":59.594,\"fix\":3,\"carr\":1,\"sats\":29,\"hacc\":0.059,\"vacc\":0.077,\"pdop\":1.69}"}' -1

ros2 topic echo /gnss_imu_eskf/odom
```
