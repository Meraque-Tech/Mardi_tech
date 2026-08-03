"""ROS-independent geodetic-to-ENU and course estimation."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import pymap3d


@dataclass(frozen=True)
class EnuEstimate:
    """One local ENU position and the latest observable course."""

    east: float
    north: float
    up: float
    yaw: float
    heading_observed: bool


class PathDirectionTracker:
    """Infer forward/reverse travel from successive ENU path segments.

    The first accepted segment establishes the forward direction. A large,
    abrupt direction change must be repeated by a second segment before it is
    reported as a reversal. Gradual turns update the forward direction one
    segment at a time, so ordinary U-turns remain classified as forward.
    """

    INITIALIZING = "initializing"
    FORWARD = "forward"
    REVERSE_SUSPECTED = "reverse_suspected"
    BACKWARD = "backward"
    FORWARD_SUSPECTED = "forward_suspected"

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
                "reversal_angle_deg must be between 90 and 180 degrees"
            )
        if (
            not math.isfinite(self.direction_consistency_deg)
            or not 0.0 <= self.direction_consistency_deg < 90.0
        ):
            raise ValueError(
                "direction_consistency_deg must be in [0, 90) degrees"
            )

        self.reference_position: Optional[Tuple[float, float]] = None
        self.travel_vector: Optional[Tuple[float, float]] = None
        self.candidate_vector: Optional[Tuple[float, float]] = None
        self.last_motion_monotonic: Optional[float] = None
        self.state = self.INITIALIZING
        self.motion_active = False

    @property
    def direction(self) -> Tuple[bool, bool]:
        """Return mutually exclusive forward/backward flags."""
        return (
            self.motion_active and self.state == self.FORWARD,
            self.motion_active and self.state == self.BACKWARD,
        )

    def update(
        self,
        east: float,
        north: float,
        now_monotonic: float,
    ) -> Tuple[bool, bool]:
        """Update the classification from one valid ENU position."""
        east = float(east)
        north = float(north)
        now_monotonic = float(now_monotonic)
        if not all(math.isfinite(value) for value in (east, north)):
            raise ValueError("east and north must be finite")
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")

        position = (east, north)
        if self.reference_position is None:
            self.reference_position = position
            return self.direction

        vector = (
            position[0] - self.reference_position[0],
            position[1] - self.reference_position[1],
        )
        if math.hypot(*vector) + 1.0e-9 < self.movement_threshold_m:
            self.clear_if_stationary(now_monotonic)
            return self.direction

        self.reference_position = position
        self.last_motion_monotonic = now_monotonic
        self.motion_active = True

        if self.state == self.INITIALIZING:
            self.state = self.FORWARD
            self.travel_vector = vector
        elif self.state == self.FORWARD:
            if self._is_reversal(vector, self.travel_vector):
                self.state = self.REVERSE_SUSPECTED
                self.candidate_vector = vector
            else:
                self.travel_vector = vector
        elif self.state == self.REVERSE_SUSPECTED:
            if self._confirms_candidate(vector):
                self.state = self.BACKWARD
                self.travel_vector = vector
                self.candidate_vector = None
            elif not self._is_reversal(vector, self.travel_vector):
                self.state = self.FORWARD
                self.travel_vector = vector
                self.candidate_vector = None
            else:
                self.candidate_vector = vector
        elif self.state == self.BACKWARD:
            if self._is_reversal(vector, self.travel_vector):
                self.state = self.FORWARD_SUSPECTED
                self.candidate_vector = vector
            else:
                self.travel_vector = vector
        elif self.state == self.FORWARD_SUSPECTED:
            if self._confirms_candidate(vector):
                self.state = self.FORWARD
                self.travel_vector = vector
                self.candidate_vector = None
            elif not self._is_reversal(vector, self.travel_vector):
                self.state = self.BACKWARD
                self.travel_vector = vector
                self.candidate_vector = None
            else:
                self.candidate_vector = vector

        return self.direction

    def clear_if_stationary(self, now_monotonic: float) -> bool:
        """Clear active flags after a stop while retaining travel direction."""
        now_monotonic = float(now_monotonic)
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")
        if self.last_motion_monotonic is None:
            return False
        if (
            now_monotonic - self.last_motion_monotonic
            <= self.stationary_timeout_s
        ):
            return False

        changed = self.motion_active
        self.motion_active = False
        self.last_motion_monotonic = None
        return changed

    def reset(self) -> None:
        """Clear all learned path direction and movement state."""
        self.reference_position = None
        self.travel_vector = None
        self.candidate_vector = None
        self.last_motion_monotonic = None
        self.state = self.INITIALIZING
        self.motion_active = False

    def _confirms_candidate(self, vector: Tuple[float, float]) -> bool:
        return (
            self._angle_degrees(vector, self.candidate_vector)
            <= self.direction_consistency_deg
            and self._is_reversal(vector, self.travel_vector)
        )

    def _is_reversal(
        self,
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> bool:
        return self._angle_degrees(first, second) >= self.reversal_angle_deg

    @staticmethod
    def _angle_degrees(
        first: Tuple[float, float],
        second: Tuple[float, float],
    ) -> float:
        first_length = math.hypot(*first)
        second_length = math.hypot(*second)
        cosine = (
            first[0] * second[0] + first[1] * second[1]
        ) / (first_length * second_length)
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def valid_geodetic(latitude: float, longitude: float, altitude: float) -> bool:
    """Return whether a geodetic position is finite and in range."""
    return (
        math.isfinite(latitude)
        and -90.0 <= latitude <= 90.0
        and math.isfinite(longitude)
        and -180.0 <= longitude <= 180.0
        and math.isfinite(altitude)
    )


class EnuCourseEstimator:
    """Project fixes into ENU and estimate course after sufficient movement."""

    def __init__(self, min_heading_distance: float = 0.1) -> None:
        min_heading_distance = float(min_heading_distance)
        if (
            not math.isfinite(min_heading_distance)
            or min_heading_distance <= 0.0
        ):
            raise ValueError(
                "min_heading_distance must be finite and greater than zero"
            )

        self.min_heading_distance = min_heading_distance
        self.origin: Optional[Tuple[float, float, float]] = None
        self.heading_reference: Optional[Tuple[float, float]] = None
        self.yaw = 0.0
        self.heading_observed = False

    def update(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> EnuEstimate:
        """Return ENU coordinates relative to the first accepted position."""
        latitude = float(latitude)
        longitude = float(longitude)
        altitude = float(altitude)
        if not valid_geodetic(latitude, longitude, altitude):
            raise ValueError("GNSS position must contain valid finite coordinates")

        if self.origin is None:
            self.origin = (latitude, longitude, altitude)
            east = north = up = 0.0
        else:
            east, north, up = pymap3d.geodetic2enu(
                latitude,
                longitude,
                altitude,
                *self.origin,
            )
            east = self._zero_negligible(east)
            north = self._zero_negligible(north)
            up = self._zero_negligible(up)

        if self.heading_reference is None:
            self.heading_reference = (east, north)
        else:
            delta_east = east - self.heading_reference[0]
            delta_north = north - self.heading_reference[1]
            if math.hypot(delta_east, delta_north) >= self.min_heading_distance:
                self.yaw = math.atan2(delta_north, delta_east)
                self.heading_observed = True
                self.heading_reference = (east, north)

        return EnuEstimate(
            east=east,
            north=north,
            up=up,
            yaw=self.yaw,
            heading_observed=self.heading_observed,
        )

    @staticmethod
    def _zero_negligible(value: float, tolerance: float = 1e-9) -> float:
        return 0.0 if abs(value) < tolerance else float(value)
