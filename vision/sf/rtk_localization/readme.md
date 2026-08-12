# bag file for lio and rtk sf
```
      cd /media/dj/dj/bags/sf_feb4 && ros2 bag play feb_2_rtk --clock
```
# Coordinate convention
```
                  Math Convention            Navigation Convention
                  North (90°)                North (0°)
                        |                           |
                        |                           |
                        |                           |
      West (180°) ------+------ East (0°)  West (270°)------+------ East (90°)
                        |                           |
                        |                           |
                        |                           |
                  South (270°)                 South (180°)
```
# Coordinate geo
```
      ✔ geodetic → ECEF
      ✔ ECEF → geodetic
      ✔ geodetic → ENU
      ✔ ENU → geodetic
      ✔ ECEF → ENU
      ✔ ENU → ECEF
      ✔ ENU → AER
      ✔ AER → ENU
      ✔ ECEF → NED
      ✔ NED → ECEF
      ✔ NED → AER
      ✔ AER → NED

      for Robotics -->

      1. geodetic → ECEF  ---> ECEF → ENU
      2. ENU → ECEF ---> ECEF → geodetic

      or,
      1. geodetic → ENU 
      2. ENU → geodetic


      How to use (example)
      gps.set_enu_origin(lat0, lon0, alt0);

      ENU p = gps.latlon_to_enu(lat, lon, alt);
      AER target = gps.enu_to_aer(p);


      Option 1 — First valid GPS fix (MOST COMMON)

      Outdoor AGV
      RTK-based localization
      No predefined map origin
      How it works
      Wait for GPS fix (STATUS_FIX or RTK FIX)

      Call set_enu_origin()

      From that moment, ENU = (0,0,0)

      Example (ROS2 callback)
      void gpsCallback(const NavSatFix &msg)
      {
      if (!gps.is_origin_set() &&
            msg.status.status >= NavSatStatus::STATUS_FIX)
      {
      gps.set_enu_origin(
            msg.latitude,
            msg.longitude,
            msg.altitude);

      RCLCPP_INFO(logger, "ENU origin set");
      }
      }
      

      Correct startup sequence (BEST PRACTICE)
      1️⃣ Start sensors

      IMU

      LiDAR

      GNSS (dual RTK)

      2️⃣ Wait until BOTH are ready

      You must wait for:

      RTK = FIXED

      LIO initialized (first pose available, map origin set)

      Robot not moving (important!)

      3️⃣ Set origins TOGETHER (this is the key)

      At that moment:

      gps.set_enu_origin(lat0, lon0, alt0);


      And simultaneously:

      LIO map origin = (0, 0, 0)

      LIO yaw aligned to RTK yaw (ideally)

      Now you have:

      LIO map frame  ≡  GPS ENU frame

      🧠 What this means mathematically

      After that point:

      LIO pose (x, y, z)  ≈  GPS ENU (e, n, u)

      So yes:

      You can treat LIO local coordinates as global GPS coordinates (ENU)

      🔁 Typical frame tree (correct)
      earth (lat/lon)
      │
      ▼
      map  (ENU, GPS origin + LIO origin)
      │
      └── odom / base_link (LIO)


      No extra offset needed if initialized correctly.

      ❌ What goes wrong if you don’t synchronize origins

      If you:

      Start LIO first

      Let it drift or move

      Then later call set_enu_origin()

      You’ll get:

      map_lio ≠ map_gps


      Then you must estimate an offset:

      T_map_gps_to_map_lio


      This is what robot_localization does internally.




```

# test code
```
ros2 topic pub /gps/fix sensor_msgs/msg/NavSatFix "
header:
  frame_id: 'gps'
status:
  status: 0
  service: 1
latitude: 49.910100
longitude: 8.899900
altitude: 120.0
" -1
```
# Bag file play
```
cd ~/robotics/race_nav/agv_ws/src/Localization/VIO/vins/RosBag_converter/bags && ros2 bag play rtk_lio_imu_out_humble --loop --clock

```

## Visual Guide:
```
      N
      ^
      |
      |
W <---|---> E
      |
      |
      S

Antenna configuration:
1. Front-back:  Left=front, Right=back
   Heading = atan2(dE, dN)  [no correction]

2. Left-right: Left=left side, Right=right side
   Heading = atan2(dE, dN) + 90°  [vehicle forward is perpendicular]



LIO Frame:                  RTK/ENU Frame:
      ^ y                       ^ N
      |                         |
      |                         |
<-----|-----> x           <-----|-----> E
      |                         |
      |                         |
      (0,0)                     (E0, N0)
      
Apply: Rotate LIO by yaw_offset to make LIO's x-axis align with East


```
