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

The only movement parameter is configured in
`config/movement_calculator.yaml`:

```yaml
movement_calculator:
  ros__parameters:
    distance_threshold: 0.05  # metres; inclusive trigger
```

The calculator publishes `true` when signed forward displacement is greater
than or equal to this value, or when signed backward displacement is less than
or equal to its negative. It publishes `false` on both topics otherwise.

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
