#!/bin/bash
set -e

# setup ros2 environment
. "/opt/ros/$ROS_DISTRO/setup.bash"
. "$AGV_WS/install/setup.bash"
exec "$@"
