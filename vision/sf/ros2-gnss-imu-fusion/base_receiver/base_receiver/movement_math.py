"""Pure geometry helpers for GPS movement classification.

This module intentionally has no ROS dependencies so the calculations can be
tested on a machine without a running ROS graph or physical sensors.
"""

from __future__ import annotations

from math import atan2, cos, isfinite, radians, sin, sqrt
from typing import Tuple


# WGS-84 ellipsoid parameters.
WGS84_SEMI_MAJOR_M = 6378137.0
WGS84_FIRST_ECCENTRICITY_SQUARED = 6.6943799901413165e-3


def _wgs84_radii(latitude_rad: float) -> Tuple[float, float]:
    """Return meridian and prime-vertical radii at a latitude."""
    sin_lat = sin(latitude_rad)
    denominator = sqrt(1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED * sin_lat**2)
    prime_vertical = WGS84_SEMI_MAJOR_M / denominator
    meridian = (
        WGS84_SEMI_MAJOR_M
        * (1.0 - WGS84_FIRST_ECCENTRICITY_SQUARED)
        / denominator**3
    )
    return meridian, prime_vertical


def enu_displacement_m(
    reference_latitude_deg: float,
    reference_longitude_deg: float,
    latitude_deg: float,
    longitude_deg: float,
) -> Tuple[float, float]:
    """Return local east/north displacement in metres.

    This local tangent-plane approximation uses WGS-84 curvature radii and is
    intended for the short intervals between consecutive vehicle fixes.
    """
    values = (
        reference_latitude_deg,
        reference_longitude_deg,
        latitude_deg,
        longitude_deg,
    )
    if not all(isfinite(value) for value in values):
        raise ValueError("latitude and longitude must be finite")
    if not -90.0 <= reference_latitude_deg <= 90.0:
        raise ValueError("reference latitude is outside [-90, 90] degrees")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude is outside [-90, 90] degrees")
    if not -180.0 <= reference_longitude_deg <= 180.0:
        raise ValueError("reference longitude is outside [-180, 180] degrees")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("longitude is outside [-180, 180] degrees")

    reference_latitude_rad = radians(reference_latitude_deg)
    latitude_rad = radians(latitude_deg)
    mean_latitude_rad = (reference_latitude_rad + latitude_rad) / 2.0

    # Select the shortest longitude arc, including across the date line.
    delta_longitude_rad = radians(longitude_deg - reference_longitude_deg)
    while delta_longitude_rad > 3.141592653589793:
        delta_longitude_rad -= 2.0 * 3.141592653589793
    while delta_longitude_rad < -3.141592653589793:
        delta_longitude_rad += 2.0 * 3.141592653589793

    meridian, prime_vertical = _wgs84_radii(mean_latitude_rad)
    north_m = meridian * (latitude_rad - reference_latitude_rad)
    east_m = prime_vertical * cos(mean_latitude_rad) * delta_longitude_rad
    return east_m, north_m


def quaternion_to_yaw_rad(x: float, y: float, z: float, w: float) -> float:
    """Convert an orientation quaternion to ROS ENU yaw in radians."""
    values = (x, y, z, w)
    if not all(isfinite(value) for value in values):
        raise ValueError("quaternion must contain finite values")
    norm_squared = sum(value * value for value in values)
    if norm_squared <= 1.0e-12:
        raise ValueError("quaternion has zero magnitude")

    scale = sqrt(norm_squared)
    x /= scale
    y /= scale
    z /= scale
    w /= scale

    return atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def signed_forward_distance_m(east_m: float, north_m: float, yaw_rad: float) -> float:
    """Project ENU displacement onto the vehicle's forward axis."""
    return east_m * cos(yaw_rad) + north_m * sin(yaw_rad)


def classify_movement(
    east_m: float,
    north_m: float,
    yaw_rad: float,
    distance_threshold_m: float,
) -> Tuple[bool, bool]:
    """Return ``(moving_forward, moving_backward)`` using inclusive bounds."""
    if not isfinite(distance_threshold_m) or distance_threshold_m <= 0.0:
        raise ValueError("distance threshold must be finite and positive")

    signed_distance = signed_forward_distance_m(east_m, north_m, yaw_rad)
    return (
        signed_distance >= distance_threshold_m,
        signed_distance <= -distance_threshold_m,
    )
