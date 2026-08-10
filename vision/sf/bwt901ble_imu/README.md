# BWT901BLE ROS 2 driver

This package wraps the supplied `device_model.py` BLE SDK and publishes its
combined acceleration, angular-velocity and orientation data as
`sensor_msgs/msg/Imu` on `imu/data_raw` and the sensor magnetic vector as
`sensor_msgs/msg/MagneticField` on `imu/mag`. The launch file also starts
`imu_filter_madgwick` with magnetometer fusion enabled to publish an absolute,
magnetic-north-referenced orientation on `imu/data`.

The SDK values are converted to the units required by ROS:

- acceleration: g to m/s^2
- angular velocity: degrees/s to radians/s
- orientation: normalized quaternion (with Euler-angle fallback until the
  quaternion register has been received)

## Build

The node requires Python's `bleak` package for Bluetooth Low Energy:

```bash
# https://github.com/WITMOTION/WitBluetooth_BWT901BLE5_0/tree/main

python3 -m pip install --user bleak
cd /home/dj/robotics/Meraque/race_nav/agv_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select bwt901ble_imu --symlink-install --merge-install
source install/setup.bash
```

## Run

The configured default BLE MAC address is `F7:75:A3:1D:9C:AB`, so run:

```bash
ros2 launch bwt901ble_imu imu.launch.py
```

To use another sensor, override `device_address`:

```bash
ros2 launch bwt901ble_imu imu.launch.py device_address:=DF:E9:1F:2C:BD:59
```

If `device_address` is explicitly set to an empty string, the node connects to
the first BLE device whose name contains `WT`.

Verify the output:

```bash
ros2 topic hz /imu/data_raw
ros2 topic echo /imu/mag
ros2 topic echo /imu/data
ros2 topic hz /imu/data
```

Madgwick uses the ENU world convention. To correct magnetic north to true
north, supply the local declination in radians; use `yaw_offset_radians` for a
fixed sensor-mounting correction:

```bash
ros2 launch bwt901ble_imu imu.launch.py \
  magnetic_declination_radians:=0.0 yaw_offset_radians:=0.0
```

Calibrate the BWT901 magnetometer in its installed environment before relying
on absolute heading. Nearby motors, steel, high-current wiring, and magnets can
distort the result.

Useful launch arguments are `device_address`, `device_name_filter`, `topic`,
`mag_topic`, `frame_id`, `publish_tf`, `tf_parent_frame`,
`magnetic_declination_radians`, and `yaw_offset_radians`. The node publishes
every combined IMU frame received from the BLE device and automatically scans
again after a connection failure or disconnect.

## Test the orientation with TF

TF output is disabled by default. Enable Madgwick's filtered absolute-heading
transform from `imu_reference` to `imu_link` with:

```bash
ros2 launch bwt901ble_imu imu.launch.py publish_tf:=true
```

Inspect it from a second terminal:

```bash
ros2 run tf2_ros tf2_echo imu_reference imu_link
```

In RViz, select `imu_reference` as the fixed frame and add a TF display. Do not
enable this transform if another node already publishes a transform whose child
frame is `imu_link`.

The published axes are the BWT901's native axes. Mount the sensor according to
its axis marking, or transform `imu_link` into the robot's REP-103 frame before
using the data for localization.

## Docker

The repository-root `docker-compose.yml` starts the combined launch:

```bash
ros2 launch bwt901ble_imu imu_gps_raw.launch.py
```

That launch runs the BLE IMU publisher, complementary filter, `base_receiver`,
`gnss_enu`, and the required `/rtk_localization` node in one container.
Localization consumes `/receiver/fix` and `/gnss/rtk_status` and publishes:

- `/gnss/odom`
- `/gnss/path`
- `/gnss/is_forward`
- `/gnss/is_backward`

The root Compose deployment selects Cyclone DDS and mounts the repository-root
`cyclone_dds_profile.xml` file read-only. Do not run another localization node
or replay recorded direction topics at the same time, because each direction
topic must have exactly one publisher.

Verify the combined deployment with:

```bash
ros2 node list
ros2 topic info --verbose /gnss/is_forward
ros2 topic info --verbose /gnss/is_backward
```

The expected localization node name is `/rtk_localization`.

Build the standalone ROS 2 Humble image using the package directory as the
Docker build context:

```bash
cd /home/dj/robotics/Meraque/race_nav/agv_ws/src/Localization/IMU/bwt901ble_imu
docker build -t bwt901ble_imu:humble .
```

Bleak communicates with the host's BlueZ service through the system D-Bus
socket. ROS 2 DDS communication is simplest with host networking:

```bash
docker run --rm -it \
  --network host \
  --volume /run/dbus/system_bus_socket:/run/dbus/system_bus_socket \
  bwt901ble_imu:humble
```

Enable the filtered absolute-heading TF or override other launch arguments by
appending them to the default command:

```bash
docker run --rm -it \
  --network host \
  --volume /run/dbus/system_bus_socket:/run/dbus/system_bus_socket \
  bwt901ble_imu:humble \
  ros2 launch bwt901ble_imu imu.launch.py publish_tf:=true
```

The phone app must be disconnected before starting the container because the
IMU accepts only one active BLE connection.
