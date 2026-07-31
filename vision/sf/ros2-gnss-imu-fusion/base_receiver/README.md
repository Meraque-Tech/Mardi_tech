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
- `/gnss/is_forward` (`std_msgs/msg/Bool`): `true` during confirmed forward
  travel.
- `/gnss/is_backward` (`std_msgs/msg/Bool`): `true` during confirmed reverse
  travel.
- `/gnss/movement_state` (`std_msgs/msg/String`): an event for every accepted
  movement segment. Values are `initializing`, `forward`,
  `reverse_suspected`, `reversing`, `forward_suspected`, and `stationary`.

The launch file starts two separate nodes: `base_receiver` publishes the raw
GNSS topics, and `gnss_enu` converts `/receiver/fix` into a local ENU position.
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
ros2 topic echo /gnss/movement_state
```

The first valid fix publishes approximately `(0, 0, 0)`. Positive `x` is east,
positive `y` is north, and positive `z` is up. These are geographic directions,
not vehicle-relative forward, left, or right.

The first valid fix establishes a movement reference and publishes both
direction flags as `false`. East/North displacement accumulates from that
reference. At the default `0.50 m` threshold, the complete two-dimensional
movement vector is accepted and its endpoint becomes the next reference.

Because no gear, vehicle-heading, or route-direction input is available, the
first accepted vector is assumed to be forward. A change of at least
`reversal_angle_deg` (default `150` degrees) creates a suspected reversal. A
second vector within `direction_consistency_deg` (default `30` degrees) of the
candidate is required to publish `reversing`. Returning to forward uses the
same two-segment confirmation. Suspected states publish both Boolean flags as
`false`.

Gradual turns update the confirmed travel vector and remain forward. A single
abrupt direction change can therefore be treated as a reversal candidate, even
if it was physically caused by a very tight turn. GNSS positions alone cannot
determine which way the vehicle body faces.

If no threshold-crossing movement occurs for `stationary_timeout_s`, the
Boolean flags become `false`, but the confirmed travel vector is retained so a
reversal after a stop can still be detected. Invalid or stale fixes reset the
tracker; after recovery, the first accepted vector is again assumed forward.
Use a larger movement threshold when normal GNSS position noise can approach
`0.50 m`.

## Docker Compose deployment

The Compose service builds a ROS 2 Humble image containing both nodes and starts
them through `receiver.launch.py`. It uses host networking for ROS 2 discovery
and host IPC for the default DDS shared-memory transport. It uses privileged
device access so the receiver can auto-detect supported USB serial adapters by
their VID/PID, matching the standalone receiver behavior.

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
