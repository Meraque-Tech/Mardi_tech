# rtk_localization

Converts the standard GNSS topics published by `base_receiver` into local ENU
odometry suitable for RViz and downstream localization.

## Interfaces

Subscribed topics:

- `/receiver/fix` (`sensor_msgs/msg/NavSatFix`): position, timestamp, validity,
  and covariance.
- `/gnss/rtk_status` (`std_msgs/msg/Bool`): RTK float/fixed availability.

Published topics:

- `/gnss/odom` (`nav_msgs/msg/Odometry`)
- `/gnss/path` (`nav_msgs/msg/Path`)
- `/gnss/is_forward` (`std_msgs/msg/Bool`)
- `/gnss/is_backward` (`std_msgs/msg/Bool`)
- TF `map -> gnss_base_link` when `publish_tf` is enabled.

The first accepted fix establishes the fixed ENU origin. X is East, Y is
North, and Z is Up. In the default two-dimensional mode, published Z is zero.
Yaw is course-over-ground and becomes observable after the receiver moves at
least `min_heading_distance` metres.

The direction flags use the two-dimensional history behind `/gnss/path`. The
first accepted movement segment is treated as forward. Gradual direction
changes, including normal U-turns, remain forward because each segment updates
the established travel direction. An abrupt change of at least
`reversal_angle_deg` must be repeated within `direction_consistency_deg` before
reverse is reported. Both flags are false while initializing, stationary,
awaiting confirmation, or when odometry is stale. The flags are always mutually
exclusive.

This is the strongest inference possible from one GNSS antenna without vehicle
heading or gear data. A tight U-turn that appears as two opposite path segments
can still be indistinguishable from reversing.

## Build and run

From the `ros2-gnss-imu-fusion` workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select base_receiver rtk_localization --symlink-install
source install/setup.bash
```

Start the receiver:

```bash
ros2 launch base_receiver receiver.launch.py
```

Start localization in another sourced terminal:

```bash
ros2 launch rtk_localization gnss_odom.launch.py
```

Start it with the supplied RViz configuration:

```bash
ros2 launch rtk_localization gnss_odom.launch.py use_rviz:=true
```

The legacy executable name is retained, so this also works:

```bash
ros2 run rtk_localization gnss_pvt_enu_odom
```

Despite that compatibility name, the node consumes `/receiver/fix`, not JSON
from `/gnss/pvt`.

## Important parameters

- `require_rtk`: reject otherwise-valid fixes until `/gnss/rtk_status` is true.
- `min_heading_distance`: movement required before course updates.
- `stationary_timeout_s`: time without threshold-crossing movement before both
  direction flags are cleared.
- `direction_stale_timeout_s`: time without valid GNSS odometry before the
  direction state is reset.
- `reversal_angle_deg`: minimum abrupt path-direction change that can begin a
  reversal.
- `direction_consistency_deg`: maximum difference between the two segments
  required to confirm a direction change.
- `two_d_mode`: publish zero altitude and treat Z as unobserved.
- `publish_tf`: broadcast `map -> gnss_base_link`.
- `path_max_poses`: maximum number of poses retained in `/gnss/path`.
