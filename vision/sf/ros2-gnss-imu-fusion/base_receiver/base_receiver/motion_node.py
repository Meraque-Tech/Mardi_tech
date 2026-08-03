"""ROS 2 wrapper for the lat/lon-only trajectory motion classifier."""

import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, String

from .trajectory import TrajectoryMotionClassifier


class GnssMotionClassifierNode(Node):
    """Publish trajectory-derived motion state from NavSatFix lat/lon."""

    def __init__(self) -> None:
        super().__init__("gnss_motion_classifier")

        defaults = {
            "fix_topic": "/receiver/fix",
            "motion_state_topic": "/gnss/motion_state",
            "is_backtracking_topic": "/gnss/is_backtracking",
            "is_headland_turning_topic": "/gnss/is_headland_turning",
            "headland_turn_completed_topic": "/gnss/headland_turn_completed",
            "filter_window": 5,
            "segment_distance_m": 0.5,
            "straight_min_distance_m": 7.0,
            "straight_course_spread_deg": 12.0,
            "straight_cross_track_m": 1.0,
            "turn_entry_deg": 30.0,
            "turn_abort_deg": 15.0,
            "turn_confirmation_segments": 3,
            "turn_completion_min_deg": 150.0,
            "turn_exit_straight_distance_m": 4.0,
            "turn_timeout_s": 60.0,
            "turn_max_distance_m": 60.0,
            "reverse_min_distance_m": 1.0,
            "reverse_angle_deg": 135.0,
            "reverse_confirmation_segments": 3,
            "reverse_cross_track_m": 1.0,
            "stationary_timeout_s": 5.0,
            "stale_timeout_s": 3.0,
            "max_speed_mps": 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.fix_topic = str(self.get_parameter("fix_topic").value)
        self.motion_state_topic = str(
            self.get_parameter("motion_state_topic").value
        )
        self.is_backtracking_topic = str(
            self.get_parameter("is_backtracking_topic").value
        )
        self.is_headland_turning_topic = str(
            self.get_parameter("is_headland_turning_topic").value
        )
        self.headland_turn_completed_topic = str(
            self.get_parameter("headland_turn_completed_topic").value
        )

        integer_parameters = {
            "filter_window",
            "turn_confirmation_segments",
            "reverse_confirmation_segments",
        }
        classifier_parameters = {}
        for name in defaults:
            if name.endswith("_topic") or name == "fix_topic":
                continue
            value = self.get_parameter(name).value
            classifier_parameters[name] = (
                int(value) if name in integer_parameters else float(value)
            )
        self.classifier = TrajectoryMotionClassifier(**classifier_parameters)

        self.state_pub = self.create_publisher(
            String,
            self.motion_state_topic,
            10,
        )
        self.backtracking_pub = self.create_publisher(
            Bool,
            self.is_backtracking_topic,
            10,
        )
        self.headland_pub = self.create_publisher(
            Bool,
            self.is_headland_turning_topic,
            10,
        )
        self.completed_pub = self.create_publisher(
            Bool,
            self.headland_turn_completed_topic,
            10,
        )
        self.fix_sub = self.create_subscription(
            NavSatFix,
            self.fix_topic,
            self._fix_callback,
            10,
        )
        self.timeout_timer = self.create_timer(0.5, self._check_timeouts)
        self.last_published_state = None
        self._publish(self.classifier.result)

        self.get_logger().info(
            "Classifying lat/lon-only motion from "
            f"{self.fix_topic}; publishing state on {self.motion_state_topic}"
        )

    def _fix_callback(self, msg: NavSatFix) -> None:
        result = self.classifier.update(
            msg.latitude,
            msg.longitude,
            time.monotonic(),
        )
        self._publish(result)

    def _check_timeouts(self) -> None:
        result = self.classifier.tick(time.monotonic())
        self._publish(result)

    def _publish(self, result) -> None:
        state_msg = String()
        state_msg.data = result.state
        self.state_pub.publish(state_msg)

        backtracking_msg = Bool()
        backtracking_msg.data = result.is_backtracking
        self.backtracking_pub.publish(backtracking_msg)

        headland_msg = Bool()
        headland_msg.data = result.is_headland_turning
        self.headland_pub.publish(headland_msg)

        completed_msg = Bool()
        completed_msg.data = result.headland_turn_completed
        self.completed_pub.publish(completed_msg)

        if result.state != self.last_published_state:
            self.get_logger().info(f"GNSS motion state: {result.state}")
            self.last_published_state = result.state
        if result.headland_turn_completed:
            self.get_logger().info("Headland turn completed")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GnssMotionClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
