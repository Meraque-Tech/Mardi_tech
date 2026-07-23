"""Publish forward/backward movement events from GNSS and IMU data."""

from __future__ import annotations

from math import isfinite
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Bool

from .movement_math import (
    classify_movement,
    enu_displacement_m,
    quaternion_to_yaw_rad,
)


class MovementCalculatorNode(Node):
    """Convert accepted GNSS displacement into forward/backward events."""

    def __init__(self) -> None:
        super().__init__("movement_calculator")

        self.declare_parameter("distance_threshold", 0.05)
        self.distance_threshold_m = float(
            self.get_parameter("distance_threshold").value
        )
        if not isfinite(self.distance_threshold_m) or self.distance_threshold_m <= 0.0:
            raise ValueError("distance_threshold must be finite and greater than zero")

        self.reference_position: Optional[Tuple[float, float]] = None
        self.latest_yaw_rad: Optional[float] = None

        self.forward_pub = self.create_publisher(
            Bool, "/receiver/moving_forward", 10
        )
        self.backward_pub = self.create_publisher(
            Bool, "/receiver/moving_backward", 10
        )
        self.create_subscription(NavSatFix, "/receiver/fix", self.fix_callback, 10)
        self.create_subscription(Imu, "/imu/processed", self.imu_callback, 10)

        self.get_logger().info(
            "Movement calculator started with distance_threshold=%.3f m",
            self.distance_threshold_m,
        )

    def imu_callback(self, msg: Imu) -> None:
        # REP-145 uses orientation_covariance[0] == -1 to indicate that the
        # IMU does not provide orientation. Treat its quaternion as unusable.
        if msg.orientation_covariance and msg.orientation_covariance[0] < 0.0:
            self.latest_yaw_rad = None
            return
        try:
            self.latest_yaw_rad = quaternion_to_yaw_rad(
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            )
        except ValueError:
            self.latest_yaw_rad = None

    def fix_callback(self, msg: NavSatFix) -> None:
        if not self._valid_fix(msg):
            self._publish_state(False, False)
            return

        current_position = (float(msg.latitude), float(msg.longitude))
        if self.reference_position is None:
            self.reference_position = current_position
            self._publish_state(False, False)
            return

        if self.latest_yaw_rad is None:
            self._publish_state(False, False)
            return

        east_m, north_m = enu_displacement_m(
            self.reference_position[0],
            self.reference_position[1],
            current_position[0],
            current_position[1],
        )
        moving_forward, moving_backward = classify_movement(
            east_m,
            north_m,
            self.latest_yaw_rad,
            self.distance_threshold_m,
        )
        self._publish_state(moving_forward, moving_backward)

        # This is an event threshold: after triggering, measure the next event
        # from the fix that caused this event.
        if moving_forward or moving_backward:
            self.reference_position = current_position

    @staticmethod
    def _valid_fix(msg: NavSatFix) -> bool:
        return (
            msg.status.status >= 0
            and isfinite(msg.latitude)
            and isfinite(msg.longitude)
            and -90.0 <= msg.latitude <= 90.0
        )

    def _publish_state(self, forward: bool, backward: bool) -> None:
        forward_msg = Bool()
        forward_msg.data = forward
        self.forward_pub.publish(forward_msg)

        backward_msg = Bool()
        backward_msg.data = backward
        self.backward_pub.publish(backward_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MovementCalculatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
