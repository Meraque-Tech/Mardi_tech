# base_receiver

Reads the ESP32 newline-delimited JSON stream used by `base_receiver.py` and
publishes GNSS ROS 2 topics:

- `/receiver/fix` (`sensor_msgs/msg/NavSatFix`): latitude, longitude,
  altitude, fix status, and covariance when the receiver supplies both
  horizontal and vertical accuracy.
- `/gnss/pvt` (`std_msgs/msg/String`): the complete enriched PVT JSON,
  including `rtkState`, `sats`, `hacc`, and `corrAgeLabel`.
- `/gnss/rtk_status` (`std_msgs/msg/Bool`): `true` for `RTK_FIXED` or
  `RTK_FLOAT`; `false` for all other states or stale PVT data.
- `/gps/enu_position` (`geometry_msgs/msg/PointStamped`): displacement in
  metres from the first valid fix, with `x=East`, `y=North`, and `z=Up`.
- `/gnss/is_forward` (`std_msgs/msg/Bool`): `true` after recent movement
  toward geographic north.
- `/gnss/is_backward` (`std_msgs/msg/Bool`): `true` after recent movement
  toward geographic south.
- `/gnss/motion_state` (`std_msgs/msg/String`): lat/lon-only trajectory state:
  `UNKNOWN`, `LEARNING`, `STATIONARY`, `FORWARD`, `BACKTRACKING`,
  `TURN_CANDIDATE`, or `HEADLAND_TURNING`.
- `/gnss/is_backtracking` (`std_msgs/msg/Bool`): `true` after sustained motion
  opposite a recently learned straight path. This is not proof of reverse gear.
- `/gnss/is_headland_turning` (`std_msgs/msg/Bool`): `true` during a confirmed
  progressive course change after a straight approach.
- `/gnss/headland_turn_completed` (`std_msgs/msg/Bool`): a one-update `true`
  pulse after the course settles approximately opposite the approach course.

The launch file starts three separate nodes: `base_receiver` publishes the raw
GNSS topics, `gnss_enu` converts `/receiver/fix` into a local ENU position, and
`gnss_motion_classifier` classifies the two-dimensional lat/lon trajectory.
The ENU origin remains fixed until `gnss_enu` restarts.

Build and run from this workspace:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths . --ignore-src -r -y
colcon build --packages-select base_receiver --symlink-install
source install/setup.bash
ros2 launch base_receiver receiver.launch.py
```

To use a fixed serial device:

```bash
ros2 run base_receiver base_receiver --ros-args \
  -p port:=/dev/ttyUSB0 -p baud:=115200
```

The launch file accepts the same deployment settings:

```bash
ros2 launch base_receiver receiver.launch.py \
  port:=/dev/ttyUSB0 \
  baud:=115200 \
  stale_timeout:=3.0 \
  reconnect_interval:=2.0
```

To run only the ENU converter against an existing `/receiver/fix` publisher:

```bash
ros2 run base_receiver gnss_enu
```

Inspect the output with:

```bash
ros2 topic echo /receiver/fix
ros2 topic echo /gnss/pvt
ros2 topic echo /gnss/rtk_status
ros2 topic echo /gps/enu_position
ros2 topic echo /gnss/is_forward
ros2 topic echo /gnss/is_backward
ros2 topic echo /gnss/motion_state
ros2 topic echo /gnss/is_backtracking
ros2 topic echo /gnss/is_headland_turning
ros2 topic echo /gnss/headland_turn_completed
```

The first valid fix publishes approximately `(0, 0, 0)`. Positive `x` is east,
positive `y` is north, and positive `z` is up. These are geographic directions,
not vehicle-relative forward, left, or right.

The first valid fix also establishes a separate movement reference and publishes
both direction flags as `false`. North/south changes accumulate from that
reference. Once the change reaches `movement_threshold_m`, the matching
direction flag becomes `true`, the other flag becomes `false`, and the current
north position becomes the next movement reference. If no threshold-crossing
movement occurs for `stationary_timeout_s`, both flags become `false`. Invalid
or stale fixes clear both flags and reset the movement reference, so the first
valid fix after recovery is not classified as movement.

The default `0.20 m` movement threshold is intended for sufficiently accurate
RTK fixes. Configure a larger threshold or add accuracy filtering when normal
GNSS position noise can exceed `0.20 m`. The flags describe recent geographic
north/south movement, not vehicle-relative forward/reverse unless the vehicle
is aligned north/south.

## Lat/lon-only motion and headland-turn classification

`gnss_motion_classifier` reads only the latitude and longitude fields from
`/receiver/fix`. It projects them into a local two-dimensional metric frame,
applies a rolling median, and waits for displacement of at least
`segment_distance_m` before calculating a new course segment.

After sufficiently straight travel over `straight_min_distance_m`, the node
learns the current row direction. Sustained motion approximately opposite that
direction and within `reverse_cross_track_m` of the learned line becomes
`BACKTRACKING`. The term is deliberate: a single GNSS antenna cannot distinguish
reverse gear from forward travel after the vehicle has turned around.

A course deviation over `turn_entry_deg` first becomes `TURN_CANDIDATE`.
Progressive deviation across `turn_confirmation_segments` changes the state to
`HEADLAND_TURNING` and suppresses backtracking classification. After the course
settles at least `turn_completion_min_deg` from the approach course for
`turn_exit_straight_distance_m`, the node emits the completion pulse and learns
the new row direction.

This requires no boundary or row map. Consequently, it recognizes the
trajectory pattern "straight row, large turn, opposite straight row" rather
than a semantic geographic headland. A similar U-turn elsewhere will also be
classified as a headland turn. The initial defaults target RTK-quality fixes;
increase the distance thresholds when position noise is larger.

The standard configuration exposes the parameters that correspond most
directly to observable vehicle behaviour:

- `filter_window`, `segment_distance_m`, and `straight_min_distance_m` control
  position smoothing and how quickly the row direction is learned.
- `turn_entry_deg`, `turn_completion_min_deg`, and
  `turn_exit_straight_distance_m` control turn detection and completion.
- `reverse_angle_deg`, `reverse_min_distance_m`, and
  `reverse_cross_track_m` control backtracking detection.
- `stationary_timeout_s`, `stale_timeout_s`, and `max_speed_mps` control input
  timeouts and implausible-position-jump rejection.

The following advanced parameters use internal defaults and are intentionally
omitted from `receiver.yaml`:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `straight_course_spread_deg` | `12.0` | Maximum segment-course variation accepted as straight travel. |
| `straight_cross_track_m` | `1.0` | Maximum line-fit deviation accepted while learning a row. |
| `turn_abort_deg` | `15.0` | Cancels a turn candidate when its course returns near the learned row. |
| `turn_confirmation_segments` | `3` | Consecutive deviating segments required to confirm a turn. |
| `turn_timeout_s` | `30.0` | Maximum elapsed time allowed for a turn. |
| `turn_max_distance_m` | `60.0` | Maximum trajectory distance allowed for a turn. |
| `reverse_confirmation_segments` | `3` | Consecutive opposite segments required to confirm backtracking. |

These remain declared ROS parameters and can be added to `receiver.yaml` when
advanced tuning is necessary. Their thresholds are coupled: values must satisfy
`turn_abort_deg < turn_entry_deg < reverse_angle_deg <
turn_completion_min_deg < 180`. In addition, `stale_timeout_s` must be less
than `stationary_timeout_s`.

Run only the classifier against an existing fix publisher with:

```bash
ros2 run base_receiver gnss_motion_classifier --ros-args \
  --params-file install/base_receiver/share/base_receiver/config/receiver.yaml
```

## Docker Compose deployment

The Compose service builds a ROS 2 Humble image containing all three nodes and
starts them through `receiver.launch.py`. It uses host networking for ROS 2
discovery and host IPC for the default DDS shared-memory transport. It uses
privileged device access so the receiver can auto-detect supported USB serial
adapters by their VID/PID, matching the standalone receiver behavior.

The deployment is intended for a Linux host with Docker Engine and Docker
Compose v2:

```bash
docker --version
docker compose version
```

### 1. Identify the receiver

Check that Linux detects the receiver:

```bash
lsusb
ls -l /dev/ttyUSB* /dev/ttyACM*
```

The receiver code recognizes these USB IDs:

```text
10c4:ea60  CP210x
1a86:7523  CH340
0403:6001  FTDI
303a:1001  Espressif USB
```

The serial device must be detected before Compose starts. The container runs as
a non-root user and retains membership in group `20`, normally Ubuntu's
`dialout` group. If the host uses a different device group, update `group_add`
in the Compose file. Check the group after the device appears:

```bash
stat -c '%g' /dev/ttyACM0
```

If the host user also needs direct access, add that user to the device-owning
group, commonly `dialout`, and start a new login session.

### 2. Build and start

```bash
docker compose -f docker-compose.base_receiver.yaml config
docker compose -f docker-compose.base_receiver.yaml build
docker compose -f docker-compose.base_receiver.yaml up
```

For background operation:

```bash
docker compose -f docker-compose.base_receiver.yaml up -d
docker compose -f docker-compose.base_receiver.yaml logs -f
```

Stop the deployment with:

```bash
docker compose -f docker-compose.base_receiver.yaml down
```

Compose sends `SIGINT` so ROS and the serial reader can shut down cleanly. The
service uses `restart: unless-stopped`.

### 3. Verify ROS output

Use a host ROS 2 Humble shell on the default ROS domain:

```bash
ros2 node list
ros2 topic echo --once /receiver/fix
ros2 topic echo --once /gnss/pvt
ros2 topic echo --once /gnss/rtk_status
ros2 topic echo --once /gps/enu_position
ros2 topic echo --once /gnss/is_forward
ros2 topic echo --once /gnss/is_backward
```

Inspect container state with:

```bash
docker compose -f docker-compose.base_receiver.yaml ps
```

Interactive Bash shells automatically load the ROS 2 environment and the
workspace overlay:

```bash
docker exec -it base_receiver bash
ros2 topic list
```

The Dockerfile uses the multi-architecture ROS Humble base image. Build
natively on `amd64` or `arm64`; no GPU runtime is required.

### Troubleshooting

**No receiver is detected**

- Reconnect the receiver and check `lsusb`.
- Check `/dev/ttyUSB*` and `/dev/ttyACM*`.
- Confirm that the USB VID/PID is one of the supported IDs above.

**The node cannot open the serial device**

- Compare `group_add` with the group reported by `stat`, for example:

  ```bash
  stat -c '%g %n' /dev/ttyACM0
  ```

**The container is running but no GNSS topics contain data**

- Check the configured baud rate and ESP32 JSON output.
- Inspect logs with:

  ```bash
  docker compose -f docker-compose.base_receiver.yaml logs -f
  ```

- Confirm that PVT records include valid finite latitude, longitude, and
  altitude fields.

**Host ROS nodes cannot discover the container**

- Confirm that both sides use the default ROS domain.
- Verify that the host firewall allows the selected DDS implementation.

**The receiver was unplugged and reconnected**

The receiver performs USB auto-discovery and reconnects when supported devices
reappear. If the container does not see a newly recreated device, restart it:

```bash
docker compose -f docker-compose.base_receiver.yaml restart
```

If deployment requires unattended hot-plug recovery, use a persistent udev
device name and validate reconnect behavior on the target host. A narrowly
scoped udev-triggered service restart is preferable to giving the container
privileged access to all host devices.
