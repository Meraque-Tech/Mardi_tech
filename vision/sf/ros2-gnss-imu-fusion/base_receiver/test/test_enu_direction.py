import pytest

from base_receiver.enu_node import direction_from_north


@pytest.mark.parametrize(
    ("north", "expected"),
    [
        (0.11, (True, False)),
        (5.0, (True, False)),
        (-0.11, (False, True)),
        (-2.0, (False, True)),
        (0.0, (False, False)),
        (0.10, (False, False)),
        (-0.10, (False, False)),
        (0.05, (False, False)),
        (-0.05, (False, False)),
        (float("nan"), (False, False)),
        (float("inf"), (False, False)),
    ],
)
def test_direction_from_north(north, expected):
    assert direction_from_north(north, 0.10) == expected


@pytest.mark.parametrize("deadband", [-0.1, float("nan"), float("inf")])
def test_direction_from_north_rejects_invalid_deadband(deadband):
    with pytest.raises(ValueError):
        direction_from_north(1.0, deadband)


def test_direction_flags_are_mutually_exclusive():
    for north in (-10.0, -0.1, 0.0, 0.1, 10.0):
        is_forward, is_backward = direction_from_north(north, 0.0)

        assert not (is_forward and is_backward)
