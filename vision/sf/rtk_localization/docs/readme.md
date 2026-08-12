# Main Purpose
```
    This node fuses:
    Dual RTK-GPS (two antennas for accurate heading)
    LIO-SAM (LiDAR-Inertial Odometry)
    to create a globally-referenced, drift-free localization system.
```

# Dual RTK Heading Calculation
```
    Subscribes to RTK fixes from /rtk/left/fix and /rtk/right/fix

    Computes vehicle heading using the baseline between two RTK antennas

    Converts latitude/longitude to UTM coordinates

    Calculates yaw from the vector between two antennas: yaw = atan2(N_diff, E_diff)

    Monitors baseline length to detect potential RTK errors
```

# RTK Odometry Publishing
```
    Publishes RTK-based odometry at /rtk/odom

    Broadcasts transform from map → base_link frames

    Maintains and publishes a path at /rtk/path

    Visualizes GPS points as green markers at /rtk/points

```

# LIO-SAM Integration
```
    Subscribes to LIO-SAM odometry at /Odometry

    Publishes raw LIO odometry at /lio/odom

    Maintains LIO path at /lio/odom/path
```

# Sensor Fusion & Frame Alignment
```
    Key innovation: Automatically computes yaw offset between RTK and LIO frames

    Waits for both sensors to initialize and move sufficiently (>0.5m)

    Computes alignment using relative motion vectors from initial positions

    Applies the computed yaw transform to align LIO pose with RTK/global frame
```

# Global Positioning
```
    Creates fused odometry at /rtk/lio/odom

    Publishes fused path at /rtk/lio/odom/path

    Converts UTM coordinates back to lat/lon for GPS topics:

    /gnss/global_pose/latlon (RTK-only)

    /lio/global_pose/latlon (Fused RTK+LIO)
```

# Services & Control
```
    Service /rtk/enable to start/stop RTK odometry publishing

    Monitors baseline tolerance (default ±0.25m from actual 1.20m baseline)

    Data Flow Summary
    text
    Dual RTK-GPS → Heading + Position → /rtk/odom
    LIO-SAM → Local Odometry → /lio/odom
    ↓
    Auto-alignment (motion-based)
    ↓
    Fused Global Pose → /rtk/lio/odom
    ↓
    UTM → Lat/Lon conversion
    ↓
    GPS topics + TF broadcasting
    Configuration Parameters
    actual_baseline: Physical distance between RTK antennas (1.20m default)

    baseline_tolerance: Allowed baseline measurement error (0.25m default)

    When would you use this?
    This is ideal for autonomous vehicles that need:

    GPS-denied or GPS-degraded operation (indoor/outdoor transitions)

    Global positioning without drift

    High-precision heading from dual antennas

    Fusion with LiDAR for obstacle-rich environments

    The system essentially gives you the local accuracy of LIO-SAM with the global consistency of RTK-GPS, automatically aligned without manual calibration.

```