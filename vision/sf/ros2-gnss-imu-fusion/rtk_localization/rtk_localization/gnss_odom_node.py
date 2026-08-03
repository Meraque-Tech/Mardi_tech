"""Publish local ENU odometry from base_receiver NavSatFix messages."""

import math
import time
from typing import Optional

from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix, NavSatStatus
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from .enu_odometry import (
    EnuCourseEstimator,
    PathDirectionTracker,
    valid_geodetic,
)


UNOBSERVED_VARIANCE = 1.0e6


class GnssFixEnuOdomNode(Node):
    """Convert validated GNSS fixes into odometry in a local ENU frame."""

    def __init__(self) -> None:
        super().__init__("gnss_fix_enu_odom")

        self.declare_parameter("fix_topic", "/receiver/fix")
        self.declare_parameter("rtk_status_topic", "/gnss/rtk_status")
        self.declare_parameter("odom_topic", "/gnss/odom")
        self.declare_parameter("path_topic", "/gnss/path")
        self.declare_parameter("is_forward_topic", "/gnss/is_forward")
        self.declare_parameter("is_backward_topic", "/gnss/is_backward")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "gnss_base_link")
        self.declare_parameter("require_rtk", False)
        self.declare_parameter("min_heading_distance", 0.1)
        self.declare_parameter("stationary_timeout_s", 5.0)
        self.declare_parameter("direction_stale_timeout_s", 3.0)
        self.declare_parameter("reversal_angle_deg", 150.0)
        self.declare_parameter("direction_consistency_deg", 30.0)
        self.declare_parameter("heading_variance", 0.05)
        self.declare_parameter("fallback_horizontal_variance", 1.0)
        self.declare_parameter("fallback_vertical_variance", 4.0)
        self.declare_parameter("two_d_mode", True)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("path_max_poses", 5000)

        self.fix_topic = str(self.get_parameter("fix_topic").value)
        self.rtk_status_topic = str(
            self.get_parameter("rtk_status_topic").value
        )
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.is_forward_topic = str(
            self.get_parameter("is_forward_topic").value
        )
        self.is_backward_topic = str(
            self.get_parameter("is_backward_topic").value
        )
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.require_rtk = bool(self.get_parameter("require_rtk").value)
        self.min_heading_distance = self._positive_parameter(
            "min_heading_distance"
        )
        self.stationary_timeout_s = self._positive_parameter(
            "stationary_timeout_s"
        )
        self.direction_stale_timeout_s = self._positive_parameter(
            "direction_stale_timeout_s"
        )
        self.heading_variance = self._positive_parameter("heading_variance")
        self.fallback_horizontal_variance = self._positive_parameter(
            "fallback_horizontal_variance"
        )
        self.fallback_vertical_variance = self._positive_parameter(
            "fallback_vertical_variance"
        )
        self.two_d_mode = bool(self.get_parameter("two_d_mode").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.path_max_poses = int(self.get_parameter("path_max_poses").value)
        if self.path_max_poses < 1:
            raise ValueError("path_max_poses must be at least 1")

        self.estimator = EnuCourseEstimator(
            self.min_heading_distance
        )
        self.direction_tracker = PathDirectionTracker(
            self.min_heading_distance,
            self.stationary_timeout_s,
            float(self.get_parameter("reversal_angle_deg").value),
            float(self.get_parameter("direction_consistency_deg").value),
        )
        self.rtk_active = False
        self._invalid_fix_warning_active = False
        self._rtk_wait_warning_active = False
        self._last_valid_odom_monotonic: Optional[float] = None

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.path_pub = self.create_publisher(Path, self.path_topic, 10)
        self.is_forward_pub = self.create_publisher(
            Bool,
            self.is_forward_topic,
            10,
        )
        self.is_backward_pub = self.create_publisher(
            Bool,
            self.is_backward_topic,
            10,
        )
        self.tf_broadcaster: Optional[TransformBroadcaster] = None
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        self.fix_sub = self.create_subscription(
            NavSatFix,
            self.fix_topic,
            self._fix_callback,
            10,
        )
        self.rtk_status_sub = self.create_subscription(
            Bool,
            self.rtk_status_topic,
            self._rtk_status_callback,
            10,
        )
        self.direction_timer = self.create_timer(
            min(
                0.5,
                self.stationary_timeout_s,
                self.direction_stale_timeout_s,
            ),
            self._check_direction_timeouts,
        )

        self.path = Path()
        self.path.header.frame_id = self.map_frame
        self._publish_direction(False, False)

        self.get_logger().info(
            f"Waiting for GNSS fixes on {self.fix_topic}; publishing ENU "
            f"odometry on {self.odom_topic}"
        )
        self.get_logger().info(
            f"Frames: {self.map_frame} -> {self.base_frame}; "
            f"require_rtk={self.require_rtk}, publish_tf={self.publish_tf}"
        )
        self.get_logger().info(
            f"Publishing path-based direction on {self.is_forward_topic} "
            f"and {self.is_backward_topic}"
        )

    def _positive_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")
        return value

    def _rtk_status_callback(self, msg: Bool) -> None:
        self.rtk_active = bool(msg.data)
        if self.rtk_active:
            self._rtk_wait_warning_active = False
        elif self.require_rtk:
            self._invalidate_direction()

    def _fix_callback(self, msg: NavSatFix) -> None:
        if not self._valid_fix(msg):
            if not self._invalid_fix_warning_active:
                self.get_logger().warning(
                    "Ignoring invalid GNSS fix; waiting for a valid finite "
                    "NavSatFix"
                )
                self._invalid_fix_warning_active = True
            self._invalidate_direction()
            return
        self._invalid_fix_warning_active = False

        if self.require_rtk and not self.rtk_active:
            if not self._rtk_wait_warning_active:
                self.get_logger().warning(
                    "Ignoring GNSS fixes until RTK float/fixed status is active"
                )
                self._rtk_wait_warning_active = True
            self._invalidate_direction()
            return

        establishing_origin = self.estimator.origin is None
        try:
            estimate = self.estimator.update(
                msg.latitude,
                msg.longitude,
                msg.altitude,
            )
        except (TypeError, ValueError, OverflowError) as error:
            self.get_logger().error(f"GNSS coordinate conversion failed: {error}")
            self._invalidate_direction()
            return

        if establishing_origin:
            latitude, longitude, altitude = self.estimator.origin
            self.get_logger().info(
                "ENU origin initialized: "
                f"lat={latitude:.9f}, lon={longitude:.9f}, "
                f"alt={altitude:.3f} m"
            )

        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            stamp = self.get_clock().now().to_msg()

        z_position = 0.0 if self.two_d_mode else estimate.up
        half_yaw = estimate.yaw * 0.5
        orientation_z = math.sin(half_yaw)
        orientation_w = math.cos(half_yaw)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.map_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = estimate.east
        odom.pose.pose.position.y = estimate.north
        odom.pose.pose.position.z = z_position
        odom.pose.pose.orientation.z = orientation_z
        odom.pose.pose.orientation.w = orientation_w
        self._set_pose_covariance(odom, msg, estimate.heading_observed)
        self._set_unobserved_twist_covariance(odom)
        self.odom_pub.publish(odom)

        pose = PoseStamped()
        pose.header = odom.header
        pose.pose = odom.pose.pose
        self.path.header = odom.header
        self.path.poses.append(pose)
        if len(self.path.poses) > self.path_max_poses:
            del self.path.poses[: len(self.path.poses) - self.path_max_poses]
        self.path_pub.publish(self.path)

        now_monotonic = time.monotonic()
        self._last_valid_odom_monotonic = now_monotonic
        is_forward, is_backward = self.direction_tracker.update(
            estimate.east,
            estimate.north,
            now_monotonic,
        )
        self._publish_direction(is_forward, is_backward)

        if self.tf_broadcaster is not None:
            transform = TransformStamped()
            transform.header = odom.header
            transform.child_frame_id = self.base_frame
            transform.transform.translation.x = estimate.east
            transform.transform.translation.y = estimate.north
            transform.transform.translation.z = z_position
            transform.transform.rotation = odom.pose.pose.orientation
            self.tf_broadcaster.sendTransform(transform)

    def _publish_direction(
        self,
        is_forward: bool,
        is_backward: bool,
    ) -> None:
        forward = Bool()
        forward.data = bool(is_forward)
        backward = Bool()
        backward.data = bool(is_backward)
        self.is_forward_pub.publish(forward)
        self.is_backward_pub.publish(backward)

    def _invalidate_direction(self) -> None:
        self._last_valid_odom_monotonic = None
        self.direction_tracker.reset()
        self._publish_direction(False, False)

    def _check_direction_timeouts(self) -> None:
        if self._last_valid_odom_monotonic is None:
            return

        now_monotonic = time.monotonic()
        if (
            now_monotonic - self._last_valid_odom_monotonic
            > self.direction_stale_timeout_s
        ):
            self.get_logger().warning(
                "GNSS odometry is stale; clearing direction flags"
            )
            self._invalidate_direction()
            return

        if self.direction_tracker.clear_if_stationary(now_monotonic):
            self._publish_direction(False, False)

    @staticmethod
    def _valid_fix(msg: NavSatFix) -> bool:
        return (
            msg.status.status >= NavSatStatus.STATUS_FIX
            and valid_geodetic(msg.latitude, msg.longitude, msg.altitude)
        )

    def _set_pose_covariance(
        self,
        odom: Odometry,
        fix: NavSatFix,
        heading_observed: bool,
    ) -> None:
        covariance_known = fix.position_covariance_type in (
            NavSatFix.COVARIANCE_TYPE_APPROXIMATED,
            NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN,
            NavSatFix.COVARIANCE_TYPE_KNOWN,
        )
        if covariance_known:
            for row in range(3):
                for column in range(3):
                    odom.pose.covariance[row * 6 + column] = (
                        fix.position_covariance[row * 3 + column]
                    )
        else:
            odom.pose.covariance[0] = self.fallback_horizontal_variance
            odom.pose.covariance[7] = self.fallback_horizontal_variance
            odom.pose.covariance[14] = self.fallback_vertical_variance

        if self.two_d_mode:
            odom.pose.covariance[14] = UNOBSERVED_VARIANCE
        odom.pose.covariance[21] = UNOBSERVED_VARIANCE
        odom.pose.covariance[28] = UNOBSERVED_VARIANCE
        odom.pose.covariance[35] = (
            self.heading_variance
            if heading_observed
            else UNOBSERVED_VARIANCE
        )

    @staticmethod
    def _set_unobserved_twist_covariance(odom: Odometry) -> None:
        for index in range(6):
            odom.twist.covariance[index * 6 + index] = UNOBSERVED_VARIANCE


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GnssFixEnuOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
