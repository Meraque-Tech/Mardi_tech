import math

import pytest

from base_receiver.odometry import StateAwareGnssOdometry
from base_receiver.trajectory import (
    BACKTRACKING,
    FORWARD,
    HEADLAND_TURNING,
    LEARNING,
    STATIONARY,
    UNKNOWN,
)


def estimator(**overrides):
    parameters = {
        "filter_window": 1,
        "yaw_smoothing_alpha": 1.0,
        "max_speed_mps": 20.0,
    }
    parameters.update(overrides)
    return StateAwareGnssOdometry(**parameters)


def test_learning_waits_for_publishable_motion_state():
    tracker = estimator()

    result = tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)

    assert result is None
    assert tracker.last_yaw == pytest.approx(0.0)


def test_forward_motion_has_positive_body_velocity():
    tracker = estimator()
    tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)

    result = tracker.update((2.0, 0.0, 0.0), 1.0, 0.0, FORWARD)

    assert result is not None
    assert result.yaw == pytest.approx(0.0)
    assert result.linear_x == pytest.approx(2.0)
    assert result.linear_y == pytest.approx(0.0, abs=1e-12)


def test_backtracking_keeps_body_yaw_and_makes_velocity_negative():
    tracker = estimator()
    tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)
    tracker.update((2.0, 0.0, 0.0), 1.0, 0.0, FORWARD)

    result = tracker.update((1.0, 0.0, 0.0), 2.0, math.pi, BACKTRACKING)

    assert result is not None
    assert result.yaw == pytest.approx(0.0, abs=1e-12)
    assert result.linear_x == pytest.approx(-1.0)
    assert result.linear_y == pytest.approx(0.0, abs=1e-12)


def test_unconfirmed_reverse_course_does_not_flip_base_link():
    tracker = estimator()
    tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)

    result = tracker.update((-1.0, 0.0, 0.0), 1.0, math.pi, FORWARD)

    assert result is None
    assert tracker.last_yaw == pytest.approx(0.0)


def test_stationary_holds_yaw_and_zeroes_twist():
    tracker = estimator()
    tracker.update((0.0, 0.0, 0.0), 0.0, math.pi / 4.0, LEARNING)
    tracker.update((1.0, 1.0, 0.0), 1.0, math.pi / 4.0, FORWARD)

    result = tracker.update((1.1, 1.1, 0.0), 2.0, None, STATIONARY)

    assert result is not None
    assert result.yaw == pytest.approx(math.pi / 4.0)
    assert result.x == pytest.approx(1.0)
    assert result.y == pytest.approx(1.0)
    assert result.linear_x == 0.0
    assert result.linear_y == 0.0
    assert result.angular_z == 0.0


def test_turn_uses_smoothed_course_and_angular_velocity():
    tracker = estimator(yaw_smoothing_alpha=0.5)
    tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)

    result = tracker.update(
        (1.0, 1.0, 0.0),
        1.0,
        math.pi / 2.0,
        HEADLAND_TURNING,
    )

    assert result is not None
    assert result.yaw == pytest.approx(math.pi / 4.0)
    assert result.angular_z == pytest.approx(math.pi / 4.0)


def test_gps_lever_arm_is_removed_from_base_position():
    tracker = estimator(gps_offset_x_m=2.0, gps_offset_y_m=1.0)
    tracker.update((10.0, 5.0, 0.0), 0.0, math.pi / 2.0, LEARNING)

    result = tracker.update(
        (10.0, 6.0, 0.0),
        1.0,
        math.pi / 2.0,
        FORWARD,
    )

    assert result is not None
    assert result.x == pytest.approx(11.0)
    assert result.y == pytest.approx(4.0)


def test_unknown_clears_tracking_and_requires_orientation_again():
    tracker = estimator()
    tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)
    assert tracker.update((1.0, 0.0, 0.0), 1.0, 0.0, FORWARD)

    assert tracker.update((1.0, 0.0, 0.0), 2.0, None, UNKNOWN) is None
    assert tracker.last_yaw is None
    assert tracker.last_estimate is None


def test_implausible_speed_is_not_published():
    tracker = estimator(max_speed_mps=2.0)
    tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, LEARNING)

    assert tracker.update((10.0, 0.0, 0.0), 1.0, 0.0, FORWARD) is None


@pytest.mark.parametrize("state", [FORWARD, BACKTRACKING, HEADLAND_TURNING])
def test_estimate_preserves_motion_state(state):
    tracker = estimator(publish_during_learning=True)

    result = tracker.update((0.0, 0.0, 0.0), 0.0, 0.0, state)

    assert result is not None
    assert result.state == state
