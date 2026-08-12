# Race-UGV Messages Package

## Overview

The `race_msgs` package contains a collection of custom message definitions for ROS topics related to the Race-UGV project. These messages facilitate communication between different nodes in the UGV's system.

## Message Types

- `Cluster.msg`: Defines a cluster of points or objects detected in the environment.
- `ClusterArray.msg`: An array of `Cluster.msg`, usually representing multiple detected objects.
- `Custom_Twist.msg`: An extension of the standard geometry `Twist` message with additional fields for UGV control.
- `Encoder.msg`: Contains encoder data from UGV's wheels or motors.
- `ImuFusionStatus.msg`: Status message for the Inertial Measurement Unit (IMU) fusion process.
- `MapToOdomStatus.msg`: Provides information on the mapping from map frame to odometry frame.
- `Mission.msg`: Details about a specific mission or task assigned to the UGV.
- `MissionCommand.msg`: Command message for starting, stopping, or modifying a mission.
- `RtkDual.msg`: Real-Time Kinematic (RTK) GPS data for high-precision location.
- `RtkDualStatus.msg`: Status information for the dual RTK GPS system.
- `RtkStatus.msg`: Status information for the RTK GPS system.
- `Task.msg`: Represents a single actionable task in the UGV's mission.
- `TopicRateStatus.msg`: Information on the rate of messages being published on a topic.
- `TopicRateStatusArray.msg`: An array of `TopicRateStatus.msg`, for monitoring multiple topics.
- `Waypoint.msg`: Defines a single waypoint for navigation purposes.

## Usage

To utilize these custom messages in your ROS package, you need to build the `race_msgs` package in your workspace and ensure it is sourced correctly.

```bash
cd ~/catkin_ws/src
git clone [repository-url]/race_msgs.git
cd ~/catkin_ws
catkin_make
source devel/setup.bash
