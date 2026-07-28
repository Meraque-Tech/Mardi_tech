import json

import pytest
from sensor_msgs.msg import NavSatStatus

from base_receiver.receiver_node import (
    _reject_non_finite_json_constant,
    enrich_pvt,
    has_position_fix,
    navsat_status_for_fix,
    normalize_pvt,
)


def valid_pvt(**overrides):
    pvt = {
        "type": "pvt",
        "fix": 3,
        "carr": 2,
        "diff": True,
        "corrAge": 1,
        "lat": 3.139,
        "lon": 101.6869,
        "alt": 42.5,
        "hacc": 0.02,
        "vacc": 0.04,
    }
    pvt.update(overrides)
    return pvt


@pytest.mark.parametrize("fix", [2, 3, 4])
def test_position_fix_types_are_accepted(fix):
    assert has_position_fix(fix)


@pytest.mark.parametrize("fix", [0, 1, 5])
def test_non_position_fix_types_are_rejected(fix):
    assert not has_position_fix(fix)


def test_time_only_cannot_be_classified_as_rtk_position():
    pvt = enrich_pvt(valid_pvt(fix=5, carr=2))

    assert pvt["fixLabel"] == "TIME_ONLY"
    assert pvt["rtkState"] == "NO_FIX"
    assert not has_position_fix(pvt["fix"])
    assert navsat_status_for_fix(pvt["fix"]) == NavSatStatus.STATUS_NO_FIX


def test_valid_pvt_is_normalized():
    pvt = normalize_pvt(
        valid_pvt(diff=1, lat="3.139", lon="101.6869", alt="42.5")
    )

    assert pvt["diff"] is True
    assert pvt["lat"] == pytest.approx(3.139)
    assert pvt["lon"] == pytest.approx(101.6869)
    assert pvt["alt"] == pytest.approx(42.5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fix", "3"),
        ("fix", 6),
        ("carr", 3),
        ("diff", "true"),
        ("corrAge", -1),
        ("lat", None),
        ("lat", float("nan")),
        ("lat", float("inf")),
        ("lat", 90.0001),
        ("lon", -180.0001),
        ("alt", float("-inf")),
        ("hacc", -0.1),
        ("vacc", float("nan")),
    ],
)
def test_invalid_pvt_fields_are_rejected(field, value):
    with pytest.raises((KeyError, ValueError)):
        normalize_pvt(valid_pvt(**{field: value}))


@pytest.mark.parametrize(
    "missing_field",
    ["fix", "carr", "diff", "corrAge", "lat", "lon", "alt"],
)
def test_missing_required_pvt_fields_are_rejected(missing_field):
    pvt = valid_pvt()
    del pvt[missing_field]

    with pytest.raises(KeyError):
        normalize_pvt(pvt)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_json_constants_are_rejected(constant):
    with pytest.raises(ValueError):
        json.loads(
            '{"type":"pvt","lat":' + constant + "}",
            parse_constant=_reject_non_finite_json_constant,
        )
