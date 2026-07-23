# base_receiver

Reads the ESP32 newline-delimited JSON stream used by `base_receiver.py` and
publishes GNSS and movement ROS 2 topics:

- `/receiver/fix` (`sensor_msgs/msg/NavSatFix`): latitude, longitude,
  altitude, fix status, and covariance when the receiver supplies both
  horizontal and vertical accuracy.
- `/gnss/pvt` (`std_msgs/msg/String`): the complete enriched PVT JSON,
  including `rtkState`, `sats`, `hacc`, and `corrAgeLabel`.
- `/gnss/rtk_status` (`std_msgs/msg/Bool`): `true` for `RTK_FIXED` or
  `RTK_FLOAT`; `false` for all other states or stale PVT data.

The launch file also starts the movement calculator. It subscribes to
`/receiver/fix` and `/imu/processed`, then publishes threshold-crossing events:

- `/receiver/moving_forward` (`std_msgs/msg/Bool`)
- `/receiver/moving_backward` (`std_msgs/msg/Bool`)

The movement parameters are configured in `config/movement_calculator.yaml`:

```yaml
movement_calculator:
  ros__parameters:
    distance_threshold: 1.0  # metres; inclusive trigger
    max_yaw_age: 0.5         # seconds
```

The calculator projects each displacement between consecutive GNSS fixes onto
the vehicle's forward axis and accumulates the signed result. It publishes
`true` when the accumulated forward displacement reaches `distance_threshold`,
or when backward displacement reaches its negative, then starts the next event
from zero. It publishes `false` on both topics otherwise. Invalid GNSS fixes and
missing or stale IMU orientation reset the partial measurement so a data gap
cannot produce a movement event.

IMU orientation must follow the ROS ENU convention: yaw zero points east,
positive yaw turns counter-clockwise toward north, and vehicle body `+X` points
forward. The configured distance threshold should be larger than normal GNSS
position noise.

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

Inspect the output with:

```bash
ros2 topic echo /receiver/fix
ros2 topic echo /gnss/pvt
ros2 topic echo /gnss/rtk_status
ros2 topic echo /receiver/moving_forward
ros2 topic echo /receiver/moving_backward
```
