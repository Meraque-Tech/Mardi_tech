"""Convert GNSS fixes to local ENU displacement from the first valid fix."""

from dataclasses import dataclass
import math
import time

from geometry_msgs.msg import PointStamped
import pymap3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, String


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


@dataclass(frozen=True)
class DirectionUpdate:
    """Result of processing one ENU position."""

    state: str
    is_forward: bool
    is_backward: bool
    segment_accepted: bool
    state_changed: bool


class MovementDirectionTracker:
    """Infer travel direction from consecutive two-dimensional vectors."""

    INITIALIZING = "initializing"
    FORWARD = "forward"
    REVERSE_SUSPECTED = "reverse_suspected"
    REVERSING = "reversing"
    FORWARD_SUSPECTED = "forward_suspected"
    STATIONARY = "stationary"

    def __init__(
        self,
        movement_threshold_m: float,
        stationary_timeout_s: float,
        reversal_angle_deg: float = 150.0,
        direction_consistency_deg: float = 30.0,
    ) -> None:
        self.movement_threshold_m = float(movement_threshold_m)
        self.stationary_timeout_s = float(stationary_timeout_s)
        self.reversal_angle_deg = float(reversal_angle_deg)
        self.direction_consistency_deg = float(direction_consistency_deg)
        if (
            not math.isfinite(self.movement_threshold_m)
            or self.movement_threshold_m <= 0.0
        ):
            raise ValueError(
                "movement_threshold_m must be finite and greater than zero"
            )
        if (
            not math.isfinite(self.stationary_timeout_s)
            or self.stationary_timeout_s <= 0.0
        ):
            raise ValueError(
                "stationary_timeout_s must be finite and greater than zero"
            )
        if (
            not math.isfinite(self.reversal_angle_deg)
            or not 90.0 < self.reversal_angle_deg < 180.0
        ):
            raise ValueError(
                "reversal_angle_deg must be finite and between 90 and 180"
            )
        if (
            not math.isfinite(self.direction_consistency_deg)
            or not 0.0 <= self.direction_consistency_deg < 90.0
        ):
            raise ValueError(
                "direction_consistency_deg must be finite and in [0, 90)"
            )

        self.reference_position = None
        self.travel_vector = None
        self.candidate_vector = None
        self.state = self.INITIALIZING
        self.motion_active = False
        self.last_motion_monotonic = None

    @property
    def direction(self):
        """Return mutually exclusive forward/backward flags."""
        return (
            self.motion_active and self.state == self.FORWARD,
            self.motion_active and self.state == self.REVERSING,
        )

    @property
    def reported_state(self) -> str:
        """Return the externally visible movement state."""
        if self.state == self.INITIALIZING:
            return self.INITIALIZING
        if not self.motion_active:
            return self.STATIONARY
        return self.state

    @staticmethod
    def _angle_degrees(first, second) -> float:
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        cosine = (
            first[0] * second[0] + first[1] * second[1]
        ) / (first_length * second_length)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    def _snapshot(
        self,
        segment_accepted: bool,
        state_changed: bool = False,
    ) -> DirectionUpdate:
        is_forward, is_backward = self.direction
        return DirectionUpdate(
            self.reported_state,
            is_forward,
            is_backward,
            segment_accepted,
            state_changed,
        )

    def update(
        self,
        east: float,
        north: float,
        now_monotonic: float,
    ) -> DirectionUpdate:
        """Update direction from a valid East/North position and time."""
        east = float(east)
        north = float(north)
        now_monotonic = float(now_monotonic)
        if not math.isfinite(east) or not math.isfinite(north):
            raise ValueError("East/North positions must be finite")
        if not math.isfinite(now_monotonic):
            raise ValueError("monotonic time must be finite")

        position = (east, north)
        if self.reference_position is None:
            self.reference_position = position
            return self._snapshot(False)

        vector = (
            position[0] - self.reference_position[0],
            position[1] - self.reference_position[1],
        )
        if (
            math.hypot(*vector) + 1e-9
            < self.movement_threshold_m
        ):
            became_stationary = self.clear_if_stationary(now_monotonic)
            return self._snapshot(False, became_stationary)

        self.reference_position = position
        self.last_motion_monotonic = now_monotonic
        self.motion_active = True

        previous_state = self.state
        if self.state == self.INITIALIZING:
            # The agreed operating assumption: the first accepted movement is
            # forward because no vehicle-heading or route direction is given.
            self.state = self.FORWARD
            self.travel_vector = vector
        elif self.state == self.FORWARD:
            if (
                self._angle_degrees(vector, self.travel_vector)
                >= self.reversal_angle_deg
            ):
                self.state = self.REVERSE_SUSPECTED
                self.candidate_vector = vector
            else:
                self.travel_vector = vector
        elif self.state == self.REVERSE_SUSPECTED:
            candidate_angle = self._angle_degrees(
                vector,
                self.candidate_vector,
            )
            original_angle = self._angle_degrees(
                vector,
                self.travel_vector,
            )
            if (
                candidate_angle <= self.direction_consistency_deg
                and original_angle >= self.reversal_angle_deg
            ):
                self.state = self.REVERSING
                self.travel_vector = vector
                self.candidate_vector = None
            elif original_angle < self.reversal_angle_deg:
                self.state = self.FORWARD
                self.travel_vector = vector
                self.candidate_vector = None
            else:
                self.candidate_vector = vector
        elif self.state == self.REVERSING:
            if (
                self._angle_degrees(vector, self.travel_vector)
                >= self.reversal_angle_deg
            ):
                self.state = self.FORWARD_SUSPECTED
                self.candidate_vector = vector
            else:
                self.travel_vector = vector
        elif self.state == self.FORWARD_SUSPECTED:
            candidate_angle = self._angle_degrees(
                vector,
                self.candidate_vector,
            )
            reverse_angle = self._angle_degrees(
                vector,
                self.travel_vector,
            )
            if (
                candidate_angle <= self.direction_consistency_deg
                and reverse_angle >= self.reversal_angle_deg
            ):
                self.state = self.FORWARD
                self.travel_vector = vector
                self.candidate_vector = None
            elif reverse_angle < self.reversal_angle_deg:
                self.state = self.REVERSING
                self.travel_vector = vector
                self.candidate_vector = None
            else:
                self.candidate_vector = vector

        return self._snapshot(True, self.state != previous_state)

    def clear_if_stationary(self, now_monotonic: float) -> bool:
        """Deactivate flags after a stop while preserving travel direction."""
        if self.last_motion_monotonic is None:
            return False
        if (
            float(now_monotonic) - self.last_motion_monotonic
            <= self.stationary_timeout_s
        ):
            return False

        changed = self.motion_active
        self.last_motion_monotonic = None
        self.motion_active = False
        return changed

    def reset(self) -> None:
        """Clear all state and assume the next accepted movement is forward."""
        self.reference_position = None
        self.travel_vector = None
        self.candidate_vector = None
        self.state = self.INITIALIZING
        self.motion_active = False
        self.last_motion_monotonic = None


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
        self.declare_parameter(
            "movement_state_topic",
            "/gnss/movement_state",
        )
        self.declare_parameter("movement_threshold_m", 0.50)
        self.declare_parameter("reversal_angle_deg", 150.0)
        self.declare_parameter("direction_consistency_deg", 30.0)
        self.declare_parameter("stationary_timeout_s", 5.0)
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
        self.movement_state_topic = str(
            self.get_parameter("movement_state_topic").value
        )
        self.movement_threshold_m = float(
            self.get_parameter("movement_threshold_m").value
        )
        self.reversal_angle_deg = float(
            self.get_parameter("reversal_angle_deg").value
        )
        self.direction_consistency_deg = float(
            self.get_parameter("direction_consistency_deg").value
        )
        self.stationary_timeout_s = float(
            self.get_parameter("stationary_timeout_s").value
        )
        self.direction_stale_timeout = float(
            self.get_parameter("direction_stale_timeout").value
        )
        if (
            not math.isfinite(self.direction_stale_timeout)
            or self.direction_stale_timeout <= 0.0
        ):
            raise ValueError(
                "direction_stale_timeout must be finite and greater than zero"
            )

        self.projector = EnuProjector()
        self.direction_tracker = MovementDirectionTracker(
            self.movement_threshold_m,
            self.stationary_timeout_s,
            self.reversal_angle_deg,
            self.direction_consistency_deg,
        )
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
        self.movement_state_pub = self.create_publisher(
            String,
            self.movement_state_topic,
            10,
        )
        self.fix_sub = self.create_subscription(
            NavSatFix,
            self.fix_topic,
            self._fix_callback,
            10,
        )
        self.direction_timer = self.create_timer(
            min(
                0.5,
                self.stationary_timeout_s,
                self.direction_stale_timeout,
            ),
            self._check_direction_timeouts,
        )
        self._publish_direction(False, False)
        self._publish_movement_state(
            self.direction_tracker.reported_state
        )

        self.get_logger().info(
            f"Waiting for the first valid GNSS fix on {self.fix_topic}; "
            f"publishing ENU positions on {self.output_topic}"
        )
        self.get_logger().info(
            f"Publishing inferred movement on {self.movement_state_topic}, "
            f"{self.is_forward_topic}, and {self.is_backward_topic}; "
            f"threshold={self.movement_threshold_m:.3f} m, "
            f"reversal={self.reversal_angle_deg:.1f} degrees, "
            f"confirmation tolerance="
            f"{self.direction_consistency_deg:.1f} degrees"
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
        now_monotonic = time.monotonic()
        self.last_valid_enu_monotonic = now_monotonic

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
        direction_update = self.direction_tracker.update(
            east,
            north,
            now_monotonic,
        )
        self._publish_direction(
            direction_update.is_forward,
            direction_update.is_backward,
        )
        if (
            direction_update.segment_accepted
            or direction_update.state_changed
        ):
            self._publish_movement_state(direction_update.state)

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

    def _publish_movement_state(self, state: str) -> None:
        state_msg = String()
        state_msg.data = state
        self.movement_state_pub.publish(state_msg)

    def _invalidate_direction(self) -> None:
        self.last_valid_enu_monotonic = None
        self.direction_tracker.reset()
        self._publish_direction(False, False)
        self._publish_movement_state(
            self.direction_tracker.reported_state
        )

    def _check_direction_timeouts(self) -> None:
        if self.last_valid_enu_monotonic is None:
            return

        now_monotonic = time.monotonic()
        if (
            now_monotonic - self.last_valid_enu_monotonic
            > self.direction_stale_timeout
        ):
            self.get_logger().warning(
                "ENU direction data is stale; clearing forward/backward flags"
            )
            self._invalidate_direction()
            return

        if self.direction_tracker.clear_if_stationary(now_monotonic):
            self.get_logger().info(
                "No threshold-crossing movement detected; marking stationary"
            )
            self._publish_direction(False, False)
            self._publish_movement_state(
                self.direction_tracker.reported_state
            )


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
