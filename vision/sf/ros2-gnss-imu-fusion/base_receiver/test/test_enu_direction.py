import pytest

from base_receiver.enu_node import MovementDirectionTracker


def tracker(threshold=0.2, timeout=5.0):
    return MovementDirectionTracker(threshold, timeout)


def test_first_fix_establishes_reference_without_direction():
    direction = tracker()

    assert direction.update(10.0, 0.0) == (False, False)
    assert direction.reference_north == 10.0


def test_noise_below_threshold_does_not_trigger_movement():
    direction = tracker()
    direction.update(10.0, 0.0)

    assert direction.update(10.19, 1.0) == (False, False)
    assert direction.update(9.81, 2.0) == (False, False)


def test_small_changes_accumulate_from_the_reference():
    direction = tracker()
    direction.update(10.0, 0.0)

    assert direction.update(10.08, 1.0) == (False, False)
    assert direction.update(10.15, 2.0) == (False, False)
    assert direction.update(10.20, 3.0) == (True, False)
    assert direction.reference_north == 10.20


def test_accumulated_south_movement_activates_backward():
    direction = tracker()
    direction.update(10.0, 0.0)

    assert direction.update(9.90, 1.0) == (False, False)
    assert direction.update(9.80, 2.0) == (False, True)


def test_subthreshold_change_temporarily_retains_current_direction():
    direction = tracker()
    direction.update(10.0, 0.0)
    direction.update(10.2, 1.0)

    assert direction.update(10.25, 5.0) == (True, False)


def test_continued_movement_refreshes_stationary_timer():
    direction = tracker()
    direction.update(10.0, 0.0)
    direction.update(10.2, 1.0)
    direction.update(10.4, 5.0)

    assert direction.clear_if_stationary(9.9) is False
    assert direction.direction == (True, False)


def test_stationary_timeout_clears_both_flags():
    direction = tracker()
    direction.update(10.0, 0.0)
    direction.update(10.2, 1.0)

    assert direction.clear_if_stationary(6.0) is False
    assert direction.clear_if_stationary(6.01) is True
    assert direction.direction == (False, False)
    assert direction.clear_if_stationary(7.0) is False


def test_direction_reversal_switches_flags():
    direction = tracker()
    direction.update(10.0, 0.0)

    assert direction.update(10.2, 1.0) == (True, False)
    assert direction.update(10.0, 2.0) == (False, True)


def test_reset_clears_state_and_next_fix_only_sets_reference():
    direction = tracker()
    direction.update(10.0, 0.0)
    direction.update(10.2, 1.0)

    direction.reset()

    assert direction.direction == (False, False)
    assert direction.reference_north is None
    assert direction.last_motion_monotonic is None
    assert direction.update(25.0, 10.0) == (False, False)
    assert direction.reference_north == 25.0


def test_direction_flags_are_always_mutually_exclusive():
    direction = tracker()

    for now, north in enumerate((0.0, 0.2, 0.1, 0.0, -0.2, 0.0), start=1):
        is_forward, is_backward = direction.update(north, float(now))
        assert not (is_forward and is_backward)


@pytest.mark.parametrize(
    ("threshold", "timeout"),
    [
        (0.0, 5.0),
        (-0.1, 5.0),
        (float("nan"), 5.0),
        (float("inf"), 5.0),
        (0.2, 0.0),
        (0.2, -1.0),
        (0.2, float("nan")),
        (0.2, float("inf")),
    ],
)
def test_tracker_rejects_invalid_configuration(threshold, timeout):
    with pytest.raises(ValueError):
        MovementDirectionTracker(threshold, timeout)
