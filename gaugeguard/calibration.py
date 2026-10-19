from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GaugeProfile:
    name: str
    unit: str
    minimum_value: float
    maximum_value: float
    minimum_angle_degrees: float
    maximum_angle_degrees: float


def angle_to_value(angle_degrees: float, profile: GaugeProfile) -> float:
    """Map an angle to a linear gauge scale.

    Angle convention: degrees increase clockwise from the image's positive x axis
    (rightward). The configured sweep from minimum to maximum must be less than
    360 degrees. Raises ValueError when the angle lies outside that sweep.
    """
    values = (
        profile.minimum_value,
        profile.maximum_value,
        profile.minimum_angle_degrees,
        profile.maximum_angle_degrees,
        angle_degrees,
    )
    if not all(map(_is_finite, values)):
        raise ValueError("Gauge calibration values must be finite numbers.")
    if profile.maximum_value <= profile.minimum_value:
        raise ValueError("Maximum gauge value must exceed minimum gauge value.")

    sweep = (profile.maximum_angle_degrees - profile.minimum_angle_degrees) % 360.0
    offset = (angle_degrees - profile.minimum_angle_degrees) % 360.0
    if sweep == 0 or offset > sweep:
        raise ValueError("Needle angle lies outside the configured gauge scale.")
    fraction = offset / sweep
    return profile.minimum_value + fraction * (profile.maximum_value - profile.minimum_value)


def _is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")
