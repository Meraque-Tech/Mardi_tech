import math

import pytest

from rtk_localization.enu_odometry import (
    EnuCourseEstimator,
    PathDirectionTracker,
    valid_geodetic,
)


ORIGIN_LAT = 2.9562305
ORIGIN_LON = 101.6519943
ORIGIN_ALT = 59.594


def test_first_fix_establishes_zero_origin():
    estimator = EnuCourseEstimator()

    estimate = estimator.update(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    assert estimate.east == 0.0
    assert estimate.north == 0.0
    assert estimate.up == 0.0
    assert not estimate.heading_observed


def test_eastward_motion_increases_east_and_sets_zero_yaw():
    estimator = EnuCourseEstimator(min_heading_distance=0.1)
    estimator.update(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    estimate = estimator.update(ORIGIN_LAT, ORIGIN_LON + 0.00001, ORIGIN_ALT)

    assert estimate.east > 1.0
    assert abs(estimate.north) < 0.01
    assert estimate.yaw == pytest.approx(0.0, abs=0.01)
    assert estimate.heading_observed


def test_northward_motion_increases_north_and_sets_north_yaw():
    estimator = EnuCourseEstimator(min_heading_distance=0.1)
    estimator.update(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    estimate = estimator.update(ORIGIN_LAT + 0.00001, ORIGIN_LON, ORIGIN_ALT)

    assert estimate.north > 1.0
    assert abs(estimate.east) < 0.01
    assert estimate.yaw == pytest.approx(math.pi / 2.0, abs=0.01)
    assert estimate.heading_observed


def test_subthreshold_motion_does_not_update_heading():
    estimator = EnuCourseEstimator(min_heading_distance=2.0)
    estimator.update(ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

    estimate = estimator.update(ORIGIN_LAT, ORIGIN_LON + 0.000001, ORIGIN_ALT)

    assert not estimate.heading_observed
    assert estimate.yaw == 0.0


@pytest.mark.parametrize(
    ("latitude", "longitude", "altitude"),
    [
        (float("nan"), ORIGIN_LON, ORIGIN_ALT),
        (91.0, ORIGIN_LON, ORIGIN_ALT),
        (ORIGIN_LAT, 181.0, ORIGIN_ALT),
        (ORIGIN_LAT, ORIGIN_LON, float("inf")),
    ],
)
def test_invalid_geodetic_positions_are_rejected(
    latitude,
    longitude,
    altitude,
):
    assert not valid_geodetic(latitude, longitude, altitude)
    with pytest.raises(ValueError):
        EnuCourseEstimator().update(latitude, longitude, altitude)


@pytest.mark.parametrize("threshold", [0.0, -0.1, float("nan"), float("inf")])
def test_invalid_heading_threshold_is_rejected(threshold):
    with pytest.raises(ValueError):
        EnuCourseEstimator(threshold)


def test_first_path_segment_establishes_forward_direction():
    tracker = PathDirectionTracker(0.5, 5.0)

    assert tracker.update(0.0, 0.0, 0.0) == (False, False)
    assert tracker.update(1.0, 0.0, 1.0) == (True, False)


def test_two_consistent_opposite_segments_confirm_reverse():
    tracker = PathDirectionTracker(0.5, 5.0)
    tracker.update(0.0, 0.0, 0.0)
    tracker.update(1.0, 0.0, 1.0)

    assert tracker.update(0.0, 0.0, 2.0) == (False, False)
    assert tracker.state == tracker.REVERSE_SUSPECTED
    assert tracker.update(-1.0, 0.0, 3.0) == (False, True)


def test_gradual_u_turn_remains_forward():
    tracker = PathDirectionTracker(0.5, 5.0)
    positions = (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.7, 0.7),
        (1.7, 1.7),
        (1.0, 2.4),
        (0.0, 2.4),
    )

    results = [
        tracker.update(east, north, float(index))
        for index, (east, north) in enumerate(positions)
    ]

    assert results[0] == (False, False)
    assert all(result == (True, False) for result in results[1:])


def test_subthreshold_path_noise_does_not_activate_direction():
    tracker = PathDirectionTracker(0.5, 5.0)

    assert tracker.update(0.0, 0.0, 0.0) == (False, False)
    assert tracker.update(0.2, 0.1, 1.0) == (False, False)
    assert tracker.update(-0.1, 0.2, 2.0) == (False, False)


def test_stationary_timeout_clears_both_flags_without_forgetting_heading():
    tracker = PathDirectionTracker(0.5, 5.0)
    tracker.update(0.0, 0.0, 0.0)
    tracker.update(1.0, 0.0, 1.0)

    assert tracker.clear_if_stationary(6.0) is False
    assert tracker.clear_if_stationary(6.01) is True
    assert tracker.direction == (False, False)

    assert tracker.update(2.0, 0.0, 7.0) == (True, False)


def test_returning_forward_also_requires_confirmation():
    tracker = PathDirectionTracker(0.5, 5.0)
    tracker.update(0.0, 0.0, 0.0)
    tracker.update(1.0, 0.0, 1.0)
    tracker.update(0.0, 0.0, 2.0)
    tracker.update(-1.0, 0.0, 3.0)

    assert tracker.update(0.0, 0.0, 4.0) == (False, False)
    assert tracker.state == tracker.FORWARD_SUSPECTED
    assert tracker.update(1.0, 0.0, 5.0) == (True, False)


@pytest.mark.parametrize(
    ("threshold", "timeout", "reversal_angle", "consistency_angle"),
    [
        (0.0, 5.0, 150.0, 30.0),
        (0.5, 0.0, 150.0, 30.0),
        (0.5, 5.0, 90.0, 30.0),
        (0.5, 5.0, 180.0, 30.0),
        (0.5, 5.0, 150.0, -1.0),
        (0.5, 5.0, 150.0, 90.0),
    ],
)
def test_path_direction_tracker_rejects_invalid_configuration(
    threshold,
    timeout,
    reversal_angle,
    consistency_angle,
):
    with pytest.raises(ValueError):
        PathDirectionTracker(
            threshold,
            timeout,
            reversal_angle,
            consistency_angle,
        )
