"""ROS-independent geodetic-to-ENU and course estimation."""

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import pymap3d


@dataclass(frozen=True)
class EnuEstimate:
    """One local ENU position and the latest observable course."""

    east: float
    north: float
    up: float
    yaw: float
    heading_observed: bool


def valid_geodetic(latitude: float, longitude: float, altitude: float) -> bool:
    """Return whether a geodetic position is finite and in range."""
    return (
        math.isfinite(latitude)
        and -90.0 <= latitude <= 90.0
        and math.isfinite(longitude)
        and -180.0 <= longitude <= 180.0
        and math.isfinite(altitude)
    )


class EnuCourseEstimator:
    """Project fixes into ENU and estimate course after sufficient movement."""

    def __init__(self, min_heading_distance: float = 0.1) -> None:
        min_heading_distance = float(min_heading_distance)
        if (
            not math.isfinite(min_heading_distance)
            or min_heading_distance <= 0.0
        ):
            raise ValueError(
                "min_heading_distance must be finite and greater than zero"
            )

        self.min_heading_distance = min_heading_distance
        self.origin: Optional[Tuple[float, float, float]] = None
        self.heading_reference: Optional[Tuple[float, float]] = None
        self.yaw = 0.0
        self.heading_observed = False

    def update(
        self,
        latitude: float,
        longitude: float,
        altitude: float,
    ) -> EnuEstimate:
        """Return ENU coordinates relative to the first accepted position."""
        latitude = float(latitude)
        longitude = float(longitude)
        altitude = float(altitude)
        if not valid_geodetic(latitude, longitude, altitude):
            raise ValueError("GNSS position must contain valid finite coordinates")

        if self.origin is None:
            self.origin = (latitude, longitude, altitude)
            east = north = up = 0.0
        else:
            east, north, up = pymap3d.geodetic2enu(
                latitude,
                longitude,
                altitude,
                *self.origin,
            )
            east = self._zero_negligible(east)
            north = self._zero_negligible(north)
            up = self._zero_negligible(up)

        if self.heading_reference is None:
            self.heading_reference = (east, north)
        else:
            delta_east = east - self.heading_reference[0]
            delta_north = north - self.heading_reference[1]
            if math.hypot(delta_east, delta_north) >= self.min_heading_distance:
                self.yaw = math.atan2(delta_north, delta_east)
                self.heading_observed = True
                self.heading_reference = (east, north)

        return EnuEstimate(
            east=east,
            north=north,
            up=up,
            yaw=self.yaw,
            heading_observed=self.heading_observed,
        )

    @staticmethod
    def _zero_negligible(value: float, tolerance: float = 1e-9) -> float:
        return 0.0 if abs(value) < tolerance else float(value)
