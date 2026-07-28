# syntax=docker/dockerfile:1

ARG ROS_IMAGE=ros:humble-ros-base-jammy

FROM ${ROS_IMAGE} AS base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-pymap3d \
        python3-serial \
        ros-humble-geometry-msgs \
        ros-humble-launch \
        ros-humble-launch-ros \
        ros-humble-sensor-msgs \
        ros-humble-std-msgs \
    && rm -rf /var/lib/apt/lists/*


FROM base AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-pytest \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws/src/base_receiver
COPY . .

RUN source /opt/ros/humble/setup.bash \
    && PYTHONDONTWRITEBYTECODE=1 \
       python3 -m pytest -q -p no:cacheprovider test \
    && cd /ros2_ws \
    && colcon build --packages-select base_receiver


FROM base AS runtime

COPY --from=builder /ros2_ws/install/ /ros2_ws/install/
COPY docker/entrypoint.sh /usr/local/bin/base-receiver-entrypoint

RUN useradd --uid 10001 --user-group --create-home ros \
    && printf '%s\n' \
        'source /opt/ros/humble/setup.bash' \
        'source /ros2_ws/install/setup.bash' \
        >> /home/ros/.bashrc \
    && chmod 0755 /usr/local/bin/base-receiver-entrypoint

USER ros
WORKDIR /ros2_ws

ENTRYPOINT ["/usr/local/bin/base-receiver-entrypoint"]
CMD ["ros2", "launch", "base_receiver", "receiver.launch.py"]
