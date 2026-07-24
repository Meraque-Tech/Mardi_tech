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
```

The first valid fix publishes approximately `(0, 0, 0)`. Positive `x` is east,
positive `y` is north, and positive `z` is up. These are geographic directions,
not vehicle-relative forward, left, or right.

## Docker Compose deployment

The Compose service builds a ROS 2 Humble image containing both nodes and starts
them through `receiver.launch.py`. It uses host networking for ROS 2 discovery
and host IPC for the default DDS shared-memory transport. It maps one host
serial device to the stable container path `/dev/gnss`.

The deployment is intended for a Linux host with Docker Engine and Docker
Compose v2:

```bash
docker --version
docker compose version
```

### 1. Identify the receiver

Prefer a persistent USB path when the host provides one:

```bash
ls -l /dev/serial/by-id/
```

If no persistent path exists, identify the current kernel device:

```bash
ls -l /dev/ttyUSB* /dev/ttyACM*
```

The serial device must exist before Compose starts the service. The Compose file
maps `/dev/ttyUSB0` by default. If the receiver uses another path, edit the
source path under `devices` in `docker-compose.base_receiver.yaml`:

```yaml
devices:
  - /dev/serial/by-id/usb-your-receiver-id:/dev/gnss
```

The container runs as a non-root user. Check the numeric group that owns the
device:

```bash
stat -c '%g' /dev/ttyUSB0
```

If the result is not `20`, replace the value under `group_add` in the Compose
file. If the host user also needs direct access, add that user to the
device-owning group, commonly `dialout`, and start a new login session.

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
```

Inspect container state with:

```bash
docker compose -f docker-compose.base_receiver.yaml ps
```

The Dockerfile uses the multi-architecture ROS Humble base image. Build
natively on `amd64` or `arm64`; no GPU runtime is required.

### Troubleshooting

**Compose reports that the device does not exist**

- Confirm that the source device configured under `devices` exists on the host.
- Reconnect the receiver and check `/dev/serial/by-id`, `/dev/ttyUSB*`, and
  `/dev/ttyACM*`.

**The node cannot open `/dev/gnss`**

- Compare `group_add` with the group reported by `stat -c '%g' /dev/ttyUSB0`.
- Confirm the device mapping with:

  ```bash
  docker compose -f docker-compose.base_receiver.yaml \
    exec base_receiver ls -l /dev/gnss
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

Docker device mappings do not behave identically on every kernel/runtime after
a device node is recreated. First try:

```bash
docker compose -f docker-compose.base_receiver.yaml restart
```

If deployment requires unattended hot-plug recovery, use a persistent udev
device name and validate reconnect behavior on the target host. A narrowly
scoped udev-triggered service restart is preferable to giving the container
privileged access to all host devices.
