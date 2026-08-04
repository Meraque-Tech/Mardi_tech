#!/bin/bash
set -e

source "/opt/ros/${ROS_DISTRO}/setup.bash"
source "${BWT901_WS}/install/setup.bash"

exec "$@"
