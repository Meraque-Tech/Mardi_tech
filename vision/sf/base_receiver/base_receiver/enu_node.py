"""Convert GNSS fixes to local ENU displacement from the first valid fix."""

import math

from geometry_msgs.msg import PointStamped
import pymap3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix


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

        self.fix_topic = str(self.get_parameter("fix_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.projector = EnuProjector()
        self.invalid_fix_warning_active = False

        self.position_pub = self.create_publisher(
            PointStamped,
            self.output_topic,
            10,
        )
        self.fix_sub = self.create_subscription(
            NavSatFix,
            self.fix_topic,
            self._fix_callback,
            10,
        )

        self.get_logger().info(
            f"Waiting for the first valid GNSS fix on {self.fix_topic}; "
            f"publishing ENU positions on {self.output_topic}"
        )

    def _fix_callback(self, msg: NavSatFix) -> None:
        establishing_origin = not self.projector.initialized

        try:
            enu = fix_to_enu(self.projector, msg)
        except (TypeError, ValueError, OverflowError) as error:
            self.get_logger().error(
                f"GNSS coordinate conversion failed: {error}"
            )
            return

        if enu is None:
            if not self.invalid_fix_warning_active:
                self.get_logger().warning(
                    "Ignoring invalid GNSS fix; waiting for finite latitude, "
                    "longitude, altitude, and a valid fix status"
                )
                self.invalid_fix_warning_active = True
            return

        self.invalid_fix_warning_active = False
        east, north, up = enu

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
