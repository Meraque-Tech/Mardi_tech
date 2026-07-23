"""Publish forward/backward movement events from GNSS and IMU data."""

from __future__ import annotations

from math import atan2, cos, isfinite, radians, sin, sqrt
from typing import Optional, Tuple

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, NavSatFix
    from std_msgs.msg import Bool
except ModuleNotFoundError:
    # Allows the pure helpers to be tested without ROS.
    rclpy = None
    Node = object
    Imu = NavSatFix = Bool = object
    qos_profile_sensor_data = None


# WGS-84 ellipsoid parameters.
WGS84_SEMI_MAJOR_M = 6378137.0
WGS84_FIRST_ECCENTRICITY_SQUARED = 6.6943799901413165e-3
PI = 3.141592653589793


def _wgs84_radii(latitude_rad: float) -> Tuple[float, float]:
    """Return meridian and prime-vertical radii at a latitude."""
    sin_lat = sin(latitude_rad)
    denominator = sqrt(1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat**2)
    prime_vertical = WGS84_SEMI_MAJOR_M / denominator
    meridian = (
        WGS84_SEMI_MAJOR_M
        * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED)
        / denominator**3
    )
    return meridian, prime_vertical


def enu_displacement_m(
    reference_latitude_deg: float,
    reference_longitude_deg: float,
    latitude_deg: float,
    longitude_deg: float,
) -> Tuple[float, float]:
    """Return local east/north displacement in metres."""
    values = (
        reference_latitude_deg,
        reference_longitude_deg,
        latitude_deg,
        longitude_deg,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("latitude and longitude must be finite")
    if not -90.0 <= reference_latitude_deg <= 90.0:
        raise ValueError("reference latitude is outside [-90, 90] degrees")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude is outside [-90, 90] degrees")
    if not -180.0 <= reference_longitude_deg <= 180.0:
        raise ValueError("reference longitude is outside [-180, 180] degrees")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("longitude is outside [-180, 180] degrees")

    reference_latitude_rad = radians(reference_latitude_deg)
    latitude_rad = radians(latitude_deg)
    mean_latitude_rad = (reference_latitude_rad + latitude_rad) / 2.0

    # Select the shortest longitude arc, including across the date line.
    delta_longitude_rad = radians(longitude_deg - reference_longitude_deg)
    while delta_longitude_rad > PI:
        delta_longitude_rad -= 2.0 * PI
    while delta_longitude_rad < -PI:
        delta_longitude_rad += 2.0 * PI

    meridian, prime_vertical = _wgs84_radii(mean_latitude_rad)
    north_m = meridian * (latitude_rad - reference_latitude_rad)
    east_m = prime_vertical * cos(mean_latitude_rad) * delta_longitude_rad
    return east_m, north_m


def quaternion_to_yaw_rad(x: float, y: float, z: float, w: float) -> float:
    """Convert an orientation quaternion to ROS ENU yaw in radians."""
    values = (x, y, z, w)
    if not all(isfinite(value) for value in values):
        raise ValueError("quaternion must contain finite values")
    norm_squared = sum(value * value for value in values)
    if norm_squared <= 1.0e-12:
        raise ValueError("quaternion has zero magnitude")

    scale = sqrt(norm_squared)
    x /= scale
    y /= scale
    z /= scale
    w /= scale

    return atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def signed_forward_distance_m(
    east_m: float,
    north_m: float,
    yaw_rad: float,
) -> float:
    """Project ENU displacement onto the vehicle's forward axis."""
    return east_m * cos(yaw_rad) + north_m * sin(yaw_rad)


def classify_movement(
    east_m: float,
    north_m: float,
    yaw_rad: float,
    distance_threshold_m: float,
) -> Tuple[bool, bool]:
    """Return ``(moving_forward, moving_backward)`` using inclusive bounds."""
    signed_distance = signed_forward_distance_m(east_m, north_m, yaw_rad)
    return classify_signed_distance(signed_distance, distance_threshold_m)


def classify_signed_distance(
    signed_distance_m: float,
    distance_threshold_m: float,
) -> Tuple[bool, bool]:
    """Classify an accumulated signed distance using inclusive bounds."""
    if not isfinite(distance_threshold_m) or distance_threshold_m <= 0.0:
        raise ValueError("distance threshold must be finite and positive")
    if not isfinite(signed_distance_m):
        raise ValueError("signed distance must be finite")

    return (
        signed_distance_m >= distance_threshold_m,
        signed_distance_m <= -distance_threshold_m,
    )


def accumulate_movement(
    accumulated_distance_m: float,
    east_m: float,
    north_m: float,
    yaw_rad: float,
    distance_threshold_m: float,
) -> Tuple[float, bool, bool]:
    """Accumulate one segment and return distance and threshold events."""
    if not isfinite(accumulated_distance_m):
        raise ValueError("accumulated distance must be finite")

    accumulated_distance_m += signed_forward_distance_m(
        east_m, north_m, yaw_rad
    )
    moving_forward, moving_backward = classify_signed_distance(
        accumulated_distance_m,
        distance_threshold_m,
    )
    if moving_forward or moving_backward:
        accumulated_distance_m = 0.0
    return accumulated_distance_m, moving_forward, moving_backward


class MovementCalculatorNode(Node):
    """Convert accepted GNSS displacement into forward/backward events."""

    def __init__(self) -> None:
        super().__init__("movement_calculator")

        self.declare_parameter("distance_threshold", 1.0)
        self.distance_threshold_m = float(
            self.get_parameter("distance_threshold").value
        )
        if (
            not isfinite(self.distance_threshold_m)
            or self.distance_threshold_m <= 0.0
        ):
            raise ValueError(
                "distance_threshold must be finite and greater than zero"
            )

        self.declare_parameter("max_yaw_age", 0.5)
        self.max_yaw_age_s = float(self.get_parameter("max_yaw_age").value)
        if not isfinite(self.max_yaw_age_s) or self.max_yaw_age_s <= 0.0:
            raise ValueError(
                "max_yaw_age must be finite and greater than zero"
            )

        self.previous_position: Optional[Tuple[float, float]] = None
        self.accumulated_distance_m = 0.0
        self.latest_yaw_rad: Optional[float] = None
        self.latest_yaw_time_ns: Optional[int] = None

        self.forward_pub = self.create_publisher(
            Bool, "/receiver/moving_forward", 10
        )
        self.backward_pub = self.create_publisher(
            Bool, "/receiver/moving_backward", 10
        )
        self.fix_sub = self.create_subscription(
            NavSatFix, "/receiver/fix", self.fix_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu,
            "/imu/processed",
            self.imu_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Movement calculator started with "
            f"distance_threshold={self.distance_threshold_m:.3f} m, "
            f"max_yaw_age={self.max_yaw_age_s:.3f} s"
        )

    def imu_callback(self, msg: Imu) -> None:
        # REP-145 uses orientation_covariance[0] == -1 to indicate that the
        # IMU does not provide orientation. Treat its quaternion as unusable.
        if (
            len(msg.orientation_covariance) > 0
            and msg.orientation_covariance[0] < 0.0
        ):
            self._clear_yaw()
            return
        try:
            self.latest_yaw_rad = quaternion_to_yaw_rad(
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            )
            self.latest_yaw_time_ns = self.get_clock().now().nanoseconds
        except ValueError:
            self._clear_yaw()

    def fix_callback(self, msg: NavSatFix) -> None:
        if not self._valid_fix(msg):
            self._reset_movement()
            self._publish_state(False, False)
            return

        current_position = (float(msg.latitude), float(msg.longitude))
        if not self._yaw_is_fresh():
            self._reset_movement()
            self._publish_state(False, False)
            return

        if self.previous_position is None:
            self.previous_position = current_position
            self._publish_state(False, False)
            return

        east_m, north_m = enu_displacement_m(
            self.previous_position[0],
            self.previous_position[1],
            current_position[0],
            current_position[1],
        )
        self.previous_position = current_position
        (
            self.accumulated_distance_m,
            moving_forward,
            moving_backward,
        ) = accumulate_movement(
            self.accumulated_distance_m,
            east_m,
            north_m,
            self.latest_yaw_rad,
            self.distance_threshold_m,
        )
        self._publish_state(moving_forward, moving_backward)

    @staticmethod
    def _valid_fix(msg: NavSatFix) -> bool:
        return (
            msg.status.status >= 0
            and isfinite(msg.latitude)
            and isfinite(msg.longitude)
            and -90.0 <= msg.latitude <= 90.0
            and -180.0 <= msg.longitude <= 180.0
        )

    def _yaw_is_fresh(self) -> bool:
        if self.latest_yaw_rad is None or self.latest_yaw_time_ns is None:
            return False
        age_ns = self.get_clock().now().nanoseconds - self.latest_yaw_time_ns
        return 0 <= age_ns <= self.max_yaw_age_s * 1.0e9

    def _clear_yaw(self) -> None:
        self.latest_yaw_rad = None
        self.latest_yaw_time_ns = None
        self._reset_movement()

    def _reset_movement(self) -> None:
        self.previous_position = None
        self.accumulated_distance_m = 0.0

    def _publish_state(self, forward: bool, backward: bool) -> None:
        forward_msg = Bool()
        forward_msg.data = forward
        self.forward_pub.publish(forward_msg)

        backward_msg = Bool()
        backward_msg.data = backward
        self.backward_pub.publish(backward_msg)


def main(args=None) -> None:
    if rclpy is None:
        raise RuntimeError(
            "ROS 2 Python dependencies are required to run this node"
        )
    rclpy.init(args=args)
    node = MovementCalculatorNode()
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
