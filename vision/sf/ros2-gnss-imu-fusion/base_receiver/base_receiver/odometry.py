"""State-aware GNSS odometry helpers.

The estimator treats ``BACKTRACKING`` as physical reverse motion.  That is a
deployment assumption: a single GNSS antenna cannot prove the selected gear.
"""

from collections import deque
from dataclasses import dataclass
import math
import statistics
from typing import Deque, Optional, Tuple

from .trajectory import (
    BACKTRACKING,
    FORWARD,
    HEADLAND_TURNING,
    LEARNING,
    STATIONARY,
    TURN_CANDIDATE,
    UNKNOWN,
    wrap_angle,
)


MOVING_STATES = {
    FORWARD,
    BACKTRACKING,
    TURN_CANDIDATE,
    HEADLAND_TURNING,
}


@dataclass(frozen=True)
class OdometryEstimate:
    """A ROS-independent planar odometry estimate."""

    timestamp: float
    x: float
    y: float
    z: float
    yaw: float
    linear_x: float
    linear_y: float
    linear_z: float
    angular_z: float
    state: str


class StateAwareGnssOdometry:
    """Estimate ``base_link`` pose and twist from GNSS and motion state.

    GNSS course is the direction of travel.  During BACKTRACKING the vehicle
    body is assumed to point 180 degrees away from that course, which produces
    a negative body-frame longitudinal velocity without flipping base_link.
    """

    def __init__(
        self,
        *,
        filter_window: int = 3,
        yaw_smoothing_alpha: float = 0.5,
        max_speed_mps: float = 10.0,
        two_d_mode: bool = True,
        publish_during_learning: bool = False,
        gps_offset_x_m: float = 0.0,
        gps_offset_y_m: float = 0.0,
        gps_offset_z_m: float = 0.0,
    ) -> None:
        if isinstance(filter_window, bool) or int(filter_window) < 1:
            raise ValueError("filter_window must be a positive integer")
        if not 0.0 < float(yaw_smoothing_alpha) <= 1.0:
            raise ValueError("yaw_smoothing_alpha must be in (0, 1]")
        if not math.isfinite(float(max_speed_mps)) or max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be finite and positive")

        offsets = (gps_offset_x_m, gps_offset_y_m, gps_offset_z_m)
        if not all(math.isfinite(float(value)) for value in offsets):
            raise ValueError("GPS offsets must be finite")

        self.filter_window = int(filter_window)
        self.yaw_smoothing_alpha = float(yaw_smoothing_alpha)
        self.max_speed_mps = float(max_speed_mps)
        self.two_d_mode = bool(two_d_mode)
        self.publish_during_learning = bool(publish_during_learning)
        self.gps_offset_x_m = float(gps_offset_x_m)
        self.gps_offset_y_m = float(gps_offset_y_m)
        self.gps_offset_z_m = float(gps_offset_z_m)

        self.positions: Deque[Tuple[float, float, float]] = deque(
            maxlen=self.filter_window
        )
        self.last_filtered_position: Optional[
            Tuple[float, float, float]
        ] = None
        self.last_timestamp: Optional[float] = None
        self.last_yaw: Optional[float] = None
        self.last_estimate: Optional[OdometryEstimate] = None

    def reset_tracking(self) -> None:
        """Forget velocity and orientation while leaving the ENU origin alone."""
        self.positions.clear()
        self.last_filtered_position = None
        self.last_timestamp = None
        self.last_yaw = None
        self.last_estimate = None

    def update(
        self,
        position_enu: Tuple[float, float, float],
        timestamp: float,
        course: Optional[float],
        state: str,
    ) -> Optional[OdometryEstimate]:
        """Consume one ENU point and return an odometry estimate if usable."""
        position = tuple(float(value) for value in position_enu)
        timestamp = float(timestamp)
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise ValueError("position_enu must contain three finite values")
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if course is not None:
            course = float(course)
            if not math.isfinite(course):
                course = None

        if state == UNKNOWN:
            self.reset_tracking()
            return None

        self.positions.append(position)
        filtered = tuple(
            statistics.median(point[index] for point in self.positions)
            for index in range(3)
        )

        previous_position = self.last_filtered_position
        previous_timestamp = self.last_timestamp
        previous_yaw = self.last_yaw
        unconfirmed_opposite_motion = (
            state == FORWARD
            and course is not None
            and self.last_yaw is not None
            and abs(wrap_angle(course - self.last_yaw)) > math.pi / 2.0
        )

        candidate_yaw = self._body_yaw(course, state)
        yaw = self._smooth_yaw(candidate_yaw, state)

        self.last_filtered_position = filtered
        self.last_timestamp = timestamp
        if yaw is not None:
            self.last_yaw = yaw

        if state == LEARNING and not self.publish_during_learning:
            return None
        if unconfirmed_opposite_motion:
            return None
        if yaw is None:
            return None

        linear_x = 0.0
        linear_y = 0.0
        linear_z = 0.0
        angular_z = 0.0

        if previous_position is not None and previous_timestamp is not None:
            dt = timestamp - previous_timestamp
            if dt > 0.0:
                delta_east = filtered[0] - previous_position[0]
                delta_north = filtered[1] - previous_position[1]
                delta_up = filtered[2] - previous_position[2]
                world_speed = math.hypot(delta_east, delta_north) / dt
                if world_speed > self.max_speed_mps:
                    return None

                if state != STATIONARY:
                    velocity_east = delta_east / dt
                    velocity_north = delta_north / dt
                    cosine = math.cos(yaw)
                    sine = math.sin(yaw)
                    linear_x = cosine * velocity_east + sine * velocity_north
                    linear_y = -sine * velocity_east + cosine * velocity_north
                    if not self.two_d_mode:
                        linear_z = delta_up / dt

                    if previous_yaw is not None:
                        angular_z = wrap_angle(yaw - previous_yaw) / dt

        if state == STATIONARY and self.last_estimate is not None:
            base_x = self.last_estimate.x
            base_y = self.last_estimate.y
            base_z = self.last_estimate.z
        else:
            base_x, base_y, base_z = self._base_position(filtered, yaw)
        estimate = OdometryEstimate(
            timestamp=timestamp,
            x=base_x,
            y=base_y,
            z=base_z,
            yaw=yaw,
            linear_x=linear_x,
            linear_y=linear_y,
            linear_z=linear_z,
            angular_z=angular_z,
            state=state,
        )
        self.last_estimate = estimate
        return estimate

    def _body_yaw(self, course: Optional[float], state: str) -> Optional[float]:
        if state == STATIONARY:
            return self.last_yaw
        if course is None:
            return self.last_yaw
        if state == BACKTRACKING:
            return wrap_angle(course + math.pi)
        if state in MOVING_STATES or state == LEARNING:
            return wrap_angle(course)
        return self.last_yaw

    def _smooth_yaw(
        self,
        candidate_yaw: Optional[float],
        state: str,
    ) -> Optional[float]:
        if candidate_yaw is None or self.last_yaw is None:
            return candidate_yaw

        difference = wrap_angle(candidate_yaw - self.last_yaw)
        # Before reverse is confirmed, the trajectory classifier remains in
        # FORWARD.  Hold the body orientation instead of allowing a temporary
        # 180-degree base_link flip during those confirmation segments.
        if state == FORWARD and abs(difference) > math.pi / 2.0:
            return self.last_yaw
        return wrap_angle(
            self.last_yaw + self.yaw_smoothing_alpha * difference
        )

    def _base_position(
        self,
        gps_position: Tuple[float, float, float],
        yaw: float,
    ) -> Tuple[float, float, float]:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        offset_east = (
            cosine * self.gps_offset_x_m - sine * self.gps_offset_y_m
        )
        offset_north = (
            sine * self.gps_offset_x_m + cosine * self.gps_offset_y_m
        )
        base_z = (
            0.0
            if self.two_d_mode
            else gps_position[2] - self.gps_offset_z_m
        )
        return (
            gps_position[0] - offset_east,
            gps_position[1] - offset_north,
            base_z,
        )
