import pytest

from base_receiver.enu_node import MovementDirectionTracker


def tracker(
    threshold=0.5,
    timeout=5.0,
    reversal_angle=150.0,
    consistency_angle=30.0,
):
    return MovementDirectionTracker(
        threshold,
        timeout,
        reversal_angle,
        consistency_angle,
    )


def update(direction, east, north, now):
    return direction.update(east, north, float(now))


def test_first_fix_establishes_reference_without_direction():
    direction = tracker()

    result = update(direction, 10.0, 20.0, 0)

    assert result.state == direction.INITIALIZING
    assert result.is_forward is False
    assert result.is_backward is False
    assert result.segment_accepted is False
    assert direction.reference_position == (10.0, 20.0)


def test_subthreshold_two_dimensional_changes_accumulate():
    direction = tracker()
    update(direction, 10.0, 20.0, 0)

    assert update(direction, 10.2, 20.1, 1).segment_accepted is False
    assert update(direction, 10.3, 20.2, 2).segment_accepted is False

    result = update(direction, 10.4, 20.3, 3)

    assert result.segment_accepted is True
    assert result.state == direction.FORWARD
    assert result.is_forward is True
    assert direction.reference_position == (10.4, 20.3)


def test_first_accepted_movement_is_assumed_forward_in_any_direction():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)

    result = update(direction, -0.5, 0.0, 1)

    assert result.state == direction.FORWARD
    assert result.is_forward is True
    assert result.is_backward is False


def test_diagonal_distance_uses_east_and_north():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)

    result = update(direction, 0.3, 0.4, 1)

    assert result.segment_accepted is True
    assert result.state == direction.FORWARD


def test_gradual_turn_remains_forward_and_updates_travel_vector():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)

    result = update(direction, 0.5, 0.5, 2)

    assert result.state == direction.FORWARD
    assert result.is_forward is True
    assert direction.travel_vector == pytest.approx((0.5, 0.0))


def test_two_consistent_opposite_segments_confirm_reversing():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)

    suspected = update(direction, 0.0, 0.0, 2)
    confirmed = update(direction, 0.0, -0.5, 3)

    assert suspected.state == direction.REVERSE_SUSPECTED
    assert suspected.is_forward is False
    assert suspected.is_backward is False
    assert confirmed.state == direction.REVERSING
    assert confirmed.is_forward is False
    assert confirmed.is_backward is True


def test_inconsistent_reverse_candidate_returns_to_forward():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)
    update(direction, 0.0, 0.0, 2)

    result = update(direction, 0.0, 0.5, 3)

    assert result.state == direction.FORWARD
    assert result.is_forward is True
    assert result.is_backward is False


def test_reversing_state_is_latched_during_continued_reverse_motion():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)
    update(direction, 0.0, 0.0, 2)
    update(direction, 0.0, -0.5, 3)

    result = update(direction, 0.0, -1.0, 4)

    assert result.state == direction.REVERSING
    assert result.is_backward is True


def test_two_opposite_segments_confirm_return_to_forward():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)
    update(direction, 0.0, 0.0, 2)
    update(direction, 0.0, -0.5, 3)

    suspected = update(direction, 0.0, 0.0, 4)
    confirmed = update(direction, 0.0, 0.5, 5)

    assert suspected.state == direction.FORWARD_SUSPECTED
    assert suspected.is_forward is False
    assert suspected.is_backward is False
    assert confirmed.state == direction.FORWARD
    assert confirmed.is_forward is True
    assert confirmed.is_backward is False


def test_stationary_timeout_clears_flags_but_preserves_direction():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)

    assert direction.clear_if_stationary(6.0) is False
    assert direction.clear_if_stationary(6.01) is True
    assert direction.reported_state == direction.STATIONARY
    assert direction.direction == (False, False)
    assert direction.state == direction.FORWARD
    assert direction.travel_vector == (0.0, 0.5)


def test_subthreshold_update_reports_stationary_transition_event():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)

    result = update(direction, 0.0, 0.6, 6.01)

    assert result.state == direction.STATIONARY
    assert result.segment_accepted is False
    assert result.state_changed is True
    assert result.is_forward is False
    assert result.is_backward is False


def test_reverse_after_stop_is_still_detected_against_preserved_vector():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)
    direction.clear_if_stationary(6.01)

    result = update(direction, 0.0, 0.0, 7)

    assert result.state == direction.REVERSE_SUSPECTED
    assert result.is_forward is False
    assert result.is_backward is False


def test_reset_requires_a_new_reference_and_forward_assumption():
    direction = tracker()
    update(direction, 0.0, 0.0, 0)
    update(direction, 0.0, 0.5, 1)

    direction.reset()

    assert direction.reported_state == direction.INITIALIZING
    assert direction.reference_position is None
    assert direction.travel_vector is None
    assert update(direction, 20.0, 30.0, 2).segment_accepted is False


@pytest.mark.parametrize(
    ("threshold", "timeout", "reversal_angle", "consistency_angle"),
    [
        (0.0, 5.0, 150.0, 30.0),
        (-0.1, 5.0, 150.0, 30.0),
        (float("nan"), 5.0, 150.0, 30.0),
        (float("inf"), 5.0, 150.0, 30.0),
        (0.5, 0.0, 150.0, 30.0),
        (0.5, -1.0, 150.0, 30.0),
        (0.5, float("nan"), 150.0, 30.0),
        (0.5, float("inf"), 150.0, 30.0),
        (0.5, 5.0, 90.0, 30.0),
        (0.5, 5.0, 180.0, 30.0),
        (0.5, 5.0, float("nan"), 30.0),
        (0.5, 5.0, 150.0, -1.0),
        (0.5, 5.0, 150.0, 90.0),
        (0.5, 5.0, 150.0, float("inf")),
    ],
)
def test_tracker_rejects_invalid_configuration(
    threshold,
    timeout,
    reversal_angle,
    consistency_angle,
):
    with pytest.raises(ValueError):
        MovementDirectionTracker(
            threshold,
            timeout,
            reversal_angle,
            consistency_angle,
        )


@pytest.mark.parametrize(
    ("east", "north", "now"),
    [
        (float("nan"), 0.0, 0.0),
        (0.0, float("inf"), 0.0),
        (0.0, 0.0, float("nan")),
    ],
)
def test_tracker_rejects_invalid_updates(east, north, now):
    with pytest.raises(ValueError):
        tracker().update(east, north, now)
