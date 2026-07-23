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
