"""Lat/lon-only trajectory classification for agricultural vehicles."""

from collections import deque
from dataclasses import dataclass
import math
import statistics
from typing import Deque, Iterable, Optional, Sequence, Tuple


EARTH_RADIUS_M = 6378137.0

UNKNOWN = "UNKNOWN"
LEARNING = "LEARNING"
STATIONARY = "STATIONARY"
FORWARD = "FORWARD"
BACKTRACKING = "BACKTRACKING"
TURN_CANDIDATE = "TURN_CANDIDATE"
HEADLAND_TURNING = "HEADLAND_TURNING"


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return math.atan2(math.sin(angle), math.cos(angle))


def circular_mean(angles: Iterable[Tuple[float, float]]) -> float:
    """Return the distance-weighted circular mean of (angle, weight) pairs."""
    pairs = list(angles)
    if not pairs:
        raise ValueError("at least one angle is required")
    x = sum(math.cos(angle) * weight for angle, weight in pairs)
    y = sum(math.sin(angle) * weight for angle, weight in pairs)
    if math.isclose(x, 0.0, abs_tol=1e-12) and math.isclose(
        y, 0.0, abs_tol=1e-12
    ):
        return pairs[-1][0]
    return math.atan2(y, x)


@dataclass(frozen=True)
class Segment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    distance: float
    course: float
    timestamp: float


@dataclass(frozen=True)
class MotionResult:
    state: str
    headland_turn_completed: bool = False

    @property
    def is_backtracking(self) -> bool:
        return self.state == BACKTRACKING

    @property
    def is_headland_turning(self) -> bool:
        return self.state == HEADLAND_TURNING


class TrajectoryMotionClassifier:
    """Classify straight motion, path backtracking, and headland U-turns.

    The classifier intentionally uses latitude, longitude, and local monotonic
    time only. BACKTRACKING is a trajectory observation, not proof that the
    vehicle gearbox is in reverse.
    """

    def __init__(
        self,
        *,
        filter_window: int = 5,
        segment_distance_m: float = 0.5,
        straight_min_distance_m: float = 7.0,
        straight_course_spread_deg: float = 12.0,
        straight_cross_track_m: float = 1.0,
        turn_entry_deg: float = 30.0,
        turn_abort_deg: float = 15.0,
        turn_confirmation_segments: int = 3,
        turn_completion_min_deg: float = 150.0,
        turn_exit_straight_distance_m: float = 4.0,
        turn_timeout_s: float = 30.0,
        turn_max_distance_m: float = 60.0,
        reverse_min_distance_m: float = 1.0,
        reverse_angle_deg: float = 135.0,
        reverse_confirmation_segments: int = 3,
        reverse_cross_track_m: float = 1.0,
        stationary_timeout_s: float = 5.0,
        stale_timeout_s: float = 3.0,
        max_speed_mps: float = 10.0,
    ) -> None:
        if isinstance(filter_window, bool) or int(filter_window) < 1:
            raise ValueError("filter_window must be a positive integer")
        if isinstance(turn_confirmation_segments, bool) or int(
            turn_confirmation_segments
        ) < 1:
            raise ValueError(
                "turn_confirmation_segments must be a positive integer"
            )
        if isinstance(reverse_confirmation_segments, bool) or int(
            reverse_confirmation_segments
        ) < 1:
            raise ValueError(
                "reverse_confirmation_segments must be a positive integer"
            )

        positive_values = {
            "segment_distance_m": segment_distance_m,
            "straight_min_distance_m": straight_min_distance_m,
            "straight_course_spread_deg": straight_course_spread_deg,
            "straight_cross_track_m": straight_cross_track_m,
            "turn_entry_deg": turn_entry_deg,
            "turn_abort_deg": turn_abort_deg,
            "turn_completion_min_deg": turn_completion_min_deg,
            "turn_exit_straight_distance_m": turn_exit_straight_distance_m,
            "turn_timeout_s": turn_timeout_s,
            "turn_max_distance_m": turn_max_distance_m,
            "reverse_min_distance_m": reverse_min_distance_m,
            "reverse_angle_deg": reverse_angle_deg,
            "reverse_cross_track_m": reverse_cross_track_m,
            "stationary_timeout_s": stationary_timeout_s,
            "stale_timeout_s": stale_timeout_s,
            "max_speed_mps": max_speed_mps,
        }
        for name, value in positive_values.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"{name} must be finite and greater than zero"
                )

        valid_angle_order = (
            0.0
            < turn_abort_deg
            < turn_entry_deg
            < reverse_angle_deg
            < 180.0
        )
        if not valid_angle_order:
            raise ValueError(
                "angle thresholds must satisfy turn_abort < turn_entry < "
                "reverse_angle < 180 degrees"
            )
        if not reverse_angle_deg < turn_completion_min_deg < 180.0:
            raise ValueError(
                "turn_completion_min_deg must be between reverse_angle_deg "
                "and 180 degrees"
            )
        if stationary_timeout_s <= stale_timeout_s:
            raise ValueError(
                "stationary_timeout_s must be greater than stale_timeout_s"
            )

        self.filter_window = int(filter_window)
        self.segment_distance_m = float(segment_distance_m)
        self.straight_min_distance_m = float(straight_min_distance_m)
        self.straight_course_spread = math.radians(straight_course_spread_deg)
        self.straight_cross_track_m = float(straight_cross_track_m)
        self.turn_entry = math.radians(turn_entry_deg)
        self.turn_abort = math.radians(turn_abort_deg)
        self.turn_confirmation_segments = int(turn_confirmation_segments)
        self.turn_completion_min = math.radians(turn_completion_min_deg)
        self.turn_exit_straight_distance_m = float(
            turn_exit_straight_distance_m
        )
        self.turn_timeout_s = float(turn_timeout_s)
        self.turn_max_distance_m = float(turn_max_distance_m)
        self.reverse_min_distance_m = float(reverse_min_distance_m)
        self.reverse_angle = math.radians(reverse_angle_deg)
        self.reverse_confirmation_segments = int(
            reverse_confirmation_segments
        )
        self.reverse_cross_track_m = float(reverse_cross_track_m)
        self.stationary_timeout_s = float(stationary_timeout_s)
        self.stale_timeout_s = float(stale_timeout_s)
        self.max_speed_mps = float(max_speed_mps)

        self._clear_all()

    def _clear_all(self) -> None:
        self.state = UNKNOWN
        self.origin_latitude: Optional[float] = None
        self.origin_longitude: Optional[float] = None
        self.raw_positions: Deque[Tuple[float, float]] = deque(
            maxlen=self.filter_window
        )
        self.segments: Deque[Segment] = deque(maxlen=500)
        self.last_accepted: Optional[Tuple[float, float, float]] = None
        self.last_fix_time: Optional[float] = None
        self.last_motion_time: Optional[float] = None
        self.resume_state = UNKNOWN
        self.forward_course: Optional[float] = None
        self.reverse_anchor: Optional[Tuple[float, float]] = None
        self.reverse_distance = 0.0
        self.reverse_count = 0
        self.forward_return_count = 0
        self.turn_start_time: Optional[float] = None
        self.turn_distance = 0.0
        self.turn_confirmation_count = 0
        self.turn_last_course: Optional[float] = None
        self.turn_rotation = 0.0
        self.turn_side = 0.0
        self.turn_exit_segments: Deque[Segment] = deque(maxlen=100)

    @property
    def result(self) -> MotionResult:
        return MotionResult(self.state, False)

    @property
    def forward_unit(self) -> Optional[Tuple[float, float]]:
        if self.forward_course is None:
            return None
        return math.cos(self.forward_course), math.sin(self.forward_course)

    def invalidate(self) -> MotionResult:
        """Clear all learned state after invalid or stale input."""
        self._clear_all()
        return self.result

    def tick(self, now_monotonic: float) -> MotionResult:
        """Apply stale and stationary timeouts without receiving a new fix."""
        now = float(now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")
        if self.last_fix_time is None:
            return self.result
        if now - self.last_fix_time > self.stale_timeout_s:
            return self.invalidate()
        self._check_stationary(now)
        return self.result

    def update(
        self,
        latitude: float,
        longitude: float,
        now_monotonic: float,
    ) -> MotionResult:
        """Consume one latitude/longitude fix and return the current state."""
        try:
            latitude = float(latitude)
            longitude = float(longitude)
            now = float(now_monotonic)
        except (TypeError, ValueError):
            return self.invalidate()
        if (
            not math.isfinite(latitude)
            or not -90.0 <= latitude <= 90.0
            or not math.isfinite(longitude)
            or not -180.0 <= longitude <= 180.0
            or not math.isfinite(now)
        ):
            return self.invalidate()

        if self.origin_latitude is None:
            self._initialize_origin(latitude, longitude, now, LEARNING)
            return self.result

        self.last_fix_time = now
        position = self._to_local(latitude, longitude)
        self.raw_positions.append(position)
        filtered = (
            statistics.median(point[0] for point in self.raw_positions),
            statistics.median(point[1] for point in self.raw_positions),
        )
        if self.last_accepted is None:
            self.last_accepted = (filtered[0], filtered[1], now)
            self.last_motion_time = now
            self.state = LEARNING
            return self.result

        previous = self.last_accepted
        dx = filtered[0] - previous[0]
        dy = filtered[1] - previous[1]
        distance = math.hypot(dx, dy)
        if distance < self.segment_distance_m:
            self._check_stationary(now)
            return self.result

        dt = now - previous[2]
        if dt <= 0.0:
            return self.result
        if distance / dt > self.max_speed_mps:
            self._clear_all()
            self._initialize_origin(latitude, longitude, now, UNKNOWN)
            return self.result

        if self.state == STATIONARY:
            self.state = (
                self.resume_state
                if self.resume_state != UNKNOWN
                else LEARNING
            )

        segment = Segment(
            start=(previous[0], previous[1]),
            end=filtered,
            distance=distance,
            course=math.atan2(dy, dx),
            timestamp=now,
        )
        self.last_accepted = (filtered[0], filtered[1], now)
        self.last_motion_time = now
        self.segments.append(segment)

        completed = False
        if self.forward_course is None:
            self.state = LEARNING
            self._try_learn_forward()
        elif self.state == TURN_CANDIDATE:
            self._process_turn_candidate(segment)
        elif self.state == HEADLAND_TURNING:
            completed = self._process_headland_turn(segment)
        elif self.state == BACKTRACKING:
            self._process_backtracking(segment)
        else:
            self._process_forward_reference(segment)

        return MotionResult(self.state, completed)

    def _initialize_origin(
        self,
        latitude: float,
        longitude: float,
        now: float,
        state: str,
    ) -> None:
        self.origin_latitude = latitude
        self.origin_longitude = longitude
        self.raw_positions.append((0.0, 0.0))
        self.last_accepted = (0.0, 0.0, now)
        self.last_fix_time = now
        self.last_motion_time = now
        self.state = state

    def _to_local(
        self,
        latitude: float,
        longitude: float,
    ) -> Tuple[float, float]:
        delta_lon = longitude - float(self.origin_longitude)
        delta_lon = (delta_lon + 180.0) % 360.0 - 180.0
        mean_latitude = math.radians(
            0.5 * (latitude + float(self.origin_latitude))
        )
        east = (
            EARTH_RADIUS_M * math.cos(mean_latitude) * math.radians(delta_lon)
        )
        north = EARTH_RADIUS_M * math.radians(
            latitude - float(self.origin_latitude)
        )
        return east, north

    def _check_stationary(self, now: float) -> None:
        if self.last_motion_time is None:
            return
        if now - self.last_motion_time <= self.stationary_timeout_s:
            return
        if self.state != STATIONARY:
            self.resume_state = self.state
            self.state = STATIONARY

    def _recent_segments(self, required_distance: float) -> Sequence[Segment]:
        selected = []
        distance = 0.0
        for segment in reversed(self.segments):
            selected.append(segment)
            distance += segment.distance
            if distance >= required_distance:
                break
        selected.reverse()
        return selected if distance >= required_distance else []

    def _straight_course(
        self,
        segments: Sequence[Segment],
    ) -> Optional[float]:
        if not segments:
            return None
        mean_course = circular_mean(
            (segment.course, segment.distance) for segment in segments
        )
        if any(
            abs(wrap_angle(segment.course - mean_course))
            > self.straight_course_spread
            for segment in segments
        ):
            return None

        unit = math.cos(mean_course), math.sin(mean_course)
        origin = segments[0].start
        for segment in segments:
            dx = segment.end[0] - origin[0]
            dy = segment.end[1] - origin[1]
            cross_track = abs(dx * unit[1] - dy * unit[0])
            if cross_track > self.straight_cross_track_m:
                return None
        return mean_course

    def _try_learn_forward(self) -> bool:
        segments = self._recent_segments(self.straight_min_distance_m)
        course = self._straight_course(segments)
        if course is None:
            return False
        self.forward_course = course
        self.state = FORWARD
        self.resume_state = FORWARD
        self.reverse_anchor = segments[-1].end
        self._reset_reverse_candidate()
        self._reset_turn()
        return True

    def _refresh_forward_course(self) -> None:
        segments = self._recent_segments(self.straight_min_distance_m)
        course = self._straight_course(segments)
        if course is None:
            return
        difference = abs(wrap_angle(course - float(self.forward_course)))
        if difference < self.turn_entry:
            self.forward_course = course

    def _process_forward_reference(self, segment: Segment) -> None:
        difference = abs(
            wrap_angle(segment.course - float(self.forward_course))
        )
        if self.turn_entry <= difference < self.reverse_angle:
            self._begin_turn(segment)
            return
        if difference >= self.reverse_angle:
            self._process_reverse_candidate(segment)
            return

        self.state = FORWARD
        self.resume_state = FORWARD
        self.reverse_anchor = segment.end
        self._reset_reverse_candidate()
        self._refresh_forward_course()

    def _process_reverse_candidate(self, segment: Segment) -> bool:
        unit = self.forward_unit
        if unit is None:
            return False
        if self.reverse_anchor is None:
            self.reverse_anchor = segment.start
        anchor_dx = segment.end[0] - self.reverse_anchor[0]
        anchor_dy = segment.end[1] - self.reverse_anchor[1]
        cross_track = abs(anchor_dx * unit[1] - anchor_dy * unit[0])
        longitudinal = (
            (segment.end[0] - segment.start[0]) * unit[0]
            + (segment.end[1] - segment.start[1]) * unit[1]
        )
        if longitudinal >= 0.0 or cross_track > self.reverse_cross_track_m:
            self._reset_reverse_candidate()
            return False

        self.reverse_distance += -longitudinal
        self.reverse_count += 1
        if (
            self.reverse_distance >= self.reverse_min_distance_m
            and self.reverse_count >= self.reverse_confirmation_segments
        ):
            self.state = BACKTRACKING
            self.resume_state = BACKTRACKING
        return True

    def _process_backtracking(self, segment: Segment) -> None:
        difference = abs(
            wrap_angle(segment.course - float(self.forward_course))
        )
        if self.turn_entry <= difference < self.reverse_angle:
            self._begin_turn(segment)
            return
        if difference >= self.reverse_angle:
            self.forward_return_count = 0
            if not self._process_reverse_candidate(segment):
                self._forget_direction()
            return
        if difference <= self.straight_course_spread:
            self.forward_return_count += 1
            if self.forward_return_count >= self.reverse_confirmation_segments:
                self.state = FORWARD
                self.resume_state = FORWARD
                self.reverse_anchor = segment.end
                self._reset_reverse_candidate()

    def _begin_turn(self, segment: Segment) -> None:
        self.state = TURN_CANDIDATE
        self.resume_state = TURN_CANDIDATE
        self.turn_start_time = segment.timestamp
        self.turn_distance = segment.distance
        self.turn_confirmation_count = 1
        self.turn_last_course = segment.course
        signed_difference = wrap_angle(
            segment.course - float(self.forward_course)
        )
        self.turn_rotation = 0.0
        self.turn_side = 1.0 if signed_difference >= 0.0 else -1.0
        self.turn_exit_segments.clear()
        self._reset_reverse_candidate()

    def _process_turn_candidate(self, segment: Segment) -> None:
        difference = abs(
            wrap_angle(segment.course - float(self.forward_course))
        )
        self._advance_turn(segment)
        if difference < self.turn_abort:
            self.state = FORWARD
            self.resume_state = FORWARD
            self.reverse_anchor = segment.end
            self._reset_turn()
            return
        signed_difference = wrap_angle(
            segment.course - float(self.forward_course)
        )
        if (
            difference >= self.turn_entry
            and signed_difference * self.turn_side > 0.0
        ):
            self.turn_confirmation_count += 1
        if self._turn_expired(segment.timestamp):
            self._forget_direction()
            return
        progressive_rotation = (
            self.turn_rotation * self.turn_side
            >= self.straight_course_spread
        )
        if (
            self.turn_confirmation_count >= self.turn_confirmation_segments
            and progressive_rotation
        ):
            self.state = HEADLAND_TURNING
            self.resume_state = HEADLAND_TURNING
            self._collect_turn_exit(segment, difference)

    def _process_headland_turn(self, segment: Segment) -> bool:
        difference = abs(
            wrap_angle(segment.course - float(self.forward_course))
        )
        self._advance_turn(segment)
        if self._turn_expired(segment.timestamp):
            self._forget_direction()
            return False
        self._collect_turn_exit(segment, difference)
        exit_segments = list(self.turn_exit_segments)
        exit_distance = sum(item.distance for item in exit_segments)
        exit_course = self._straight_course(exit_segments)
        if (
            exit_distance < self.turn_exit_straight_distance_m
            or exit_course is None
            or abs(wrap_angle(exit_course - float(self.forward_course)))
            < self.turn_completion_min
        ):
            return False

        self.forward_course = exit_course
        self.state = FORWARD
        self.resume_state = FORWARD
        self.segments = deque(exit_segments, maxlen=500)
        self.reverse_anchor = exit_segments[-1].end
        self._reset_reverse_candidate()
        self._reset_turn()
        return True

    def _advance_turn(self, segment: Segment) -> None:
        self.turn_distance += segment.distance
        if self.turn_last_course is not None:
            self.turn_rotation += wrap_angle(
                segment.course - self.turn_last_course
            )
        self.turn_last_course = segment.course

    def _collect_turn_exit(
        self,
        segment: Segment,
        difference: float,
    ) -> None:
        if difference < self.turn_completion_min:
            self.turn_exit_segments.clear()
            return
        self.turn_exit_segments.append(segment)
        if self._straight_course(list(self.turn_exit_segments)) is None:
            self.turn_exit_segments.clear()
            self.turn_exit_segments.append(segment)

    def _turn_expired(self, now: float) -> bool:
        return (
            self.turn_start_time is not None
            and now - self.turn_start_time > self.turn_timeout_s
        ) or self.turn_distance > self.turn_max_distance_m

    def _reset_reverse_candidate(self) -> None:
        self.reverse_distance = 0.0
        self.reverse_count = 0
        self.forward_return_count = 0

    def _reset_turn(self) -> None:
        self.turn_start_time = None
        self.turn_distance = 0.0
        self.turn_confirmation_count = 0
        self.turn_last_course = None
        self.turn_rotation = 0.0
        self.turn_side = 0.0
        self.turn_exit_segments.clear()

    def _forget_direction(self) -> None:
        self.forward_course = None
        self.reverse_anchor = None
        self.state = LEARNING
        self.resume_state = LEARNING
        self.segments.clear()
        self._reset_reverse_candidate()
        self._reset_turn()
