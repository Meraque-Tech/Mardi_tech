import math

import pytest

from rtk_localization.enu_odometry import (
    EnuCourseEstimator,
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
