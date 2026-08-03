import math

import pytest

from base_receiver.trajectory import (
    BACKTRACKING,
    FORWARD,
    HEADLAND_TURNING,
    LEARNING,
    STATIONARY,
    UNKNOWN,
    EARTH_RADIUS_M,
    TrajectoryMotionClassifier,
)


LATITUDE = 3.139
LONGITUDE = 101.6869


def classifier(**overrides):
    parameters = {
        "filter_window": 1,
        "segment_distance_m": 0.4,
        "straight_min_distance_m": 4.0,
        "straight_course_spread_deg": 12.0,
        "straight_cross_track_m": 0.6,
        "turn_entry_deg": 30.0,
        "turn_abort_deg": 15.0,
        "turn_confirmation_segments": 3,
        "turn_completion_min_deg": 150.0,
        "turn_exit_straight_distance_m": 2.0,
        "turn_timeout_s": 30.0,
        "turn_max_distance_m": 50.0,
        "reverse_min_distance_m": 1.0,
        "reverse_angle_deg": 135.0,
        "reverse_confirmation_segments": 3,
        "reverse_cross_track_m": 0.8,
        "stationary_timeout_s": 5.0,
        "stale_timeout_s": 3.0,
        "max_speed_mps": 5.0,
    }
    parameters.update(overrides)
    return TrajectoryMotionClassifier(**parameters)


def to_latlon(east, north):
    latitude = LATITUDE + math.degrees(north / EARTH_RADIUS_M)
    longitude = LONGITUDE + math.degrees(
        east / (EARTH_RADIUS_M * math.cos(math.radians(LATITUDE)))
    )
    return latitude, longitude


def feed_points(tracker, points, start_time=0.0, step_s=1.0):
    results = []
    now = start_time
    for east, north in points:
        latitude, longitude = to_latlon(east, north)
        results.append(tracker.update(latitude, longitude, now))
        now += step_s
    return results, now


def eastbound_row(length=6.0, step=0.5):
    count = int(round(length / step))
    return [(index * step, 0.0) for index in range(count + 1)]


def test_straight_history_learns_forward_direction():
    tracker = classifier()

    results, _ = feed_points(tracker, eastbound_row())

    assert results[0].state == LEARNING
    assert results[-1].state == FORWARD
    assert tracker.forward_course == pytest.approx(0.0, abs=1e-6)


def test_straight_path_retrace_is_classified_as_backtracking():
    tracker = classifier()
    _, now = feed_points(tracker, eastbound_row())

    reverse_points = [(5.5, 0.0), (5.0, 0.0), (4.5, 0.0)]
    results, _ = feed_points(tracker, reverse_points, start_time=now)

    assert results[-1].state == BACKTRACKING
    assert results[-1].is_backtracking
    assert not results[-1].is_headland_turning


def test_backtracking_works_for_a_northbound_row():
    tracker = classifier()
    northbound = [(0.0, 0.5 * index) for index in range(13)]
    _, now = feed_points(tracker, northbound)

    reverse_points = [(0.0, 5.5), (0.0, 5.0), (0.0, 4.5)]
    results, _ = feed_points(tracker, reverse_points, start_time=now)

    assert results[-1].state == BACKTRACKING


def test_progressive_u_turn_suppresses_backtracking_and_completes():
    tracker = classifier()
    _, now = feed_points(tracker, eastbound_row())

    radius = 2.0
    arc = []
    for degrees in range(-75, 91, 15):
        angle = math.radians(degrees)
        arc.append(
            (
                6.0 + radius * math.cos(angle),
                2.0 + radius * math.sin(angle),
            )
        )
    exit_row = [(7.5 - 0.5 * index, 4.0) for index in range(1, 8)]
    results, _ = feed_points(tracker, arc + exit_row, start_time=now)

    assert any(result.state == HEADLAND_TURNING for result in results)
    assert any(result.headland_turn_completed for result in results)
    assert all(not result.is_backtracking for result in results)
    assert results[-1].state == FORWARD
    assert abs(abs(tracker.forward_course) - math.pi) < math.radians(12.0)


def test_constant_course_after_single_corner_is_not_a_headland_turn():
    tracker = classifier()
    _, now = feed_points(tracker, eastbound_row())
    diagonal = [(6.0 + 0.5 * index, 0.5 * index) for index in range(1, 8)]

    results, _ = feed_points(tracker, diagonal, start_time=now)

    assert all(result.state != HEADLAND_TURNING for result in results)
    assert all(not result.headland_turn_completed for result in results)


def test_small_stationary_jitter_does_not_create_motion():
    tracker = classifier()
    points = [
        (0.00, 0.00),
        (0.05, -0.03),
        (-0.04, 0.02),
        (0.03, 0.01),
    ]

    results, _ = feed_points(tracker, points, step_s=2.0)

    assert results[-1].state == STATIONARY
    assert not results[-1].is_backtracking


def test_stale_input_clears_learned_direction():
    tracker = classifier()
    _, now = feed_points(tracker, eastbound_row())
    assert tracker.state == FORWARD

    result = tracker.tick(now + 3.1)

    assert result.state == UNKNOWN
    assert tracker.forward_course is None


def test_unrealistic_position_jump_resets_classifier():
    tracker = classifier(max_speed_mps=2.0)
    _, now = feed_points(tracker, eastbound_row(), step_s=1.0)

    result, _ = feed_points(tracker, [(100.0, 0.0)], start_time=now)

    assert result[-1].state == UNKNOWN
    assert tracker.forward_course is None


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("segment_distance_m", 0.0),
        ("straight_min_distance_m", float("nan")),
        ("filter_window", 0),
        ("turn_confirmation_segments", 0),
        ("reverse_confirmation_segments", -1),
        ("turn_entry_deg", 140.0),
        ("turn_completion_min_deg", 120.0),
        ("stationary_timeout_s", 3.0),
    ],
)
def test_invalid_configuration_is_rejected(parameter, value):
    with pytest.raises(ValueError):
        classifier(**{parameter: value})
