"""Convert GNSS fixes to local ENU displacement from the first valid fix."""

import math
import time

from geometry_msgs.msg import PointStamped
import pymap3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool


def is_valid_fix(msg) -> bool:
    """Return whether a NavSatFix is suitable for ECEF/ENU conversion."""
    return (
        msg.status.status >= 0
        and math.isfinite(msg.latitude)
        and -90.0 <= msg.latitude <= 90.0
        and math.isfinite(msg.longitude)
        and -180.0 <= msg.longitude <= 180.0
        and math.isfinite(msg.altitude)
    )


def _zero_if_negligible(value: float, tolerance: float = 1e-9) -> float:
    """Remove floating-point residue at the ENU origin."""
    return 0.0 if abs(value) < tolerance else float(value)


def direction_from_north(north: float, deadband: float):
    """Classify north displacement into mutually exclusive direction flags."""
    north = float(north)
    deadband = float(deadband)
    if not math.isfinite(north):
        return False, False
    if not math.isfinite(deadband) or deadband < 0.0:
        raise ValueError("north deadband must be finite and non-negative")
    if north > deadband:
        return True, False
    if north < -deadband:
        return False, True
    return False, False


class EnuProjector:
    """Project WGS84 positions into ENU using the first point as origin."""

    def __init__(self) -> None:
        self.origin_geodetic = None
        self.origin_ecef = None

    @property
    def initialized(self) -> bool:
        return self.origin_geodetic is not None

    def to_enu(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ):
        """Return East, North, Up in metres relative to the fixed origin."""
        if not self.initialized:
            origin_geodetic = (
                float(latitude),
                float(longitude),
                float(altitude),
            )
            origin_ecef = tuple(
                float(value)
                for value in pymap3d.geodetic2ecef(*origin_geodetic)
            )
            self.origin_geodetic = origin_geodetic
            self.origin_ecef = origin_ecef

        current_ecef = pymap3d.geodetic2ecef(
            float(latitude),
            float(longitude),
            float(altitude),
        )
        east, north, up = pymap3d.ecef2enu(
            *current_ecef,
            *self.origin_geodetic,
        )
        return tuple(_zero_if_negligible(value) for value in (east, north, up))


def fix_to_enu(projector: EnuProjector, msg):
    """Return ENU coordinates for a valid fix, otherwise return None."""
    if not is_valid_fix(msg):
        return None
    return projector.to_enu(msg.latitude, msg.longitude, msg.altitude)


class GnssEnuNode(Node):
    """Publish local ENU displacement from the first valid GNSS fix."""

    def __init__(self) -> None:
        super().__init__("gnss_enu")

        self.declare_parameter("fix_topic", "/receiver/fix")
        self.declare_parameter("output_topic", "/gps/enu_position")
        self.declare_parameter("frame_id", "enu")
        self.declare_parameter("is_forward_topic", "/gnss/is_forward")
        self.declare_parameter("is_backward_topic", "/gnss/is_backward")
        self.declare_parameter("north_deadband_m", 0.10)
        self.declare_parameter("direction_stale_timeout", 3.0)

        self.fix_topic = str(self.get_parameter("fix_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.is_forward_topic = str(
            self.get_parameter("is_forward_topic").value
        )
        self.is_backward_topic = str(
            self.get_parameter("is_backward_topic").value
        )
        self.north_deadband_m = float(
            self.get_parameter("north_deadband_m").value
        )
        self.direction_stale_timeout = float(
            self.get_parameter("direction_stale_timeout").value
        )
        if (
            not math.isfinite(self.north_deadband_m)
            or self.north_deadband_m < 0.0
        ):
            raise ValueError("north_deadband_m must be finite and non-negative")
        if (
            not math.isfinite(self.direction_stale_timeout)
            or self.direction_stale_timeout <= 0.0
        ):
            raise ValueError(
                "direction_stale_timeout must be finite and greater than zero"
            )

        self.projector = EnuProjector()
        self.invalid_fix_warning_active = False
        self.last_valid_enu_monotonic = None

        self.position_pub = self.create_publisher(
            PointStamped,
            self.output_topic,
            10,
        )
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
        self.fix_sub = self.create_subscription(
            NavSatFix,
            self.fix_topic,
            self._fix_callback,
            10,
        )
        self.direction_stale_timer = self.create_timer(
            min(0.5, self.direction_stale_timeout),
            self._clear_stale_direction,
        )
        self._publish_direction(False, False)

        self.get_logger().info(
            f"Waiting for the first valid GNSS fix on {self.fix_topic}; "
            f"publishing ENU positions on {self.output_topic}"
        )
        self.get_logger().info(
            f"Publishing north-position flags on {self.is_forward_topic} "
            f"and {self.is_backward_topic} with a "
            f"{self.north_deadband_m:.3f} m deadband"
        )

    def _fix_callback(self, msg: NavSatFix) -> None:
        establishing_origin = not self.projector.initialized

        try:
            enu = fix_to_enu(self.projector, msg)
        except (TypeError, ValueError, OverflowError) as error:
            self.get_logger().error(
                f"GNSS coordinate conversion failed: {error}"
            )
            self._invalidate_direction()
            return

        if enu is None:
            if not self.invalid_fix_warning_active:
                self.get_logger().warning(
                    "Ignoring invalid GNSS fix; waiting for finite latitude, "
                    "longitude, altitude, and a valid fix status"
                )
                self.invalid_fix_warning_active = True
            self._invalidate_direction()
            return

        self.invalid_fix_warning_active = False
        east, north, up = enu
        self.last_valid_enu_monotonic = time.monotonic()

        if establishing_origin:
            origin_lat, origin_lon, origin_alt = self.projector.origin_geodetic
            origin_x, origin_y, origin_z = self.projector.origin_ecef
            self.get_logger().info(
                "ENU origin initialized from first valid fix: "
                f"lat={origin_lat:.9f}, lon={origin_lon:.9f}, "
                f"alt={origin_alt:.3f} m; "
                f"ECEF=({origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f}) m"
            )

        position = PointStamped()
        position.header.stamp = msg.header.stamp
        position.header.frame_id = self.frame_id
        position.point.x = east
        position.point.y = north
        position.point.z = up
        self.position_pub.publish(position)
        is_forward, is_backward = direction_from_north(
            north,
            self.north_deadband_m,
        )
        self._publish_direction(is_forward, is_backward)

    def _publish_direction(
        self,
        is_forward: bool,
        is_backward: bool,
    ) -> None:
        forward_msg = Bool()
        forward_msg.data = bool(is_forward)
        backward_msg = Bool()
        backward_msg.data = bool(is_backward)
        self.is_forward_pub.publish(forward_msg)
        self.is_backward_pub.publish(backward_msg)

    def _invalidate_direction(self) -> None:
        self.last_valid_enu_monotonic = None
        self._publish_direction(False, False)

    def _clear_stale_direction(self) -> None:
        if self.last_valid_enu_monotonic is None:
            return
        if (
            time.monotonic() - self.last_valid_enu_monotonic
            <= self.direction_stale_timeout
        ):
            return
        self.get_logger().warning(
            "ENU direction data is stale; clearing forward/backward flags"
        )
        self._invalidate_direction()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GnssEnuNode()

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
