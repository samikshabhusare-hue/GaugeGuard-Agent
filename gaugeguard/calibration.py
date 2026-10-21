from __future__ import annotations

from dataclasses import dataclass


# === Exceptions ===


class CalibrationError(ValueError):
    """Raised when a gauge profile is invalid (programmer error)."""


class UnknownProfileError(CalibrationError):
    """Raised when a requested profile name is not registered."""


# === Utilities ===


def _is_finite(value: float) -> bool:
    """Check if a value is finite (not NaN, not infinity)."""
    return value == value and abs(value) != float("inf")


# === GaugeProfile ===


@dataclass(frozen=True)
class GaugeProfile:
    """Calibration profile for one gauge model.

    Angle convention: degrees in [0, 360), measured per the team-agreed convention.
    Currently only clockwise direction is supported.
    """
    name: str
    unit: str
    minimum_value: float
    maximum_value: float
    minimum_angle_degrees: float
    maximum_angle_degrees: float
    direction: str = "cw"
    bounds_margin_deg: float = 5.0
    operating_min: float | None = None
    operating_max: float | None = None
    agreement_tolerance_fraction: float = 0.02
    eval_tolerance_fraction: float = 0.05


# === Profile Validation ===


def validate_profile(profile: GaugeProfile) -> None:
    """Validate a GaugeProfile for consistency.

    Raises:
        CalibrationError: If any field violates constraints.
    """
    # Check all numeric fields are finite
    numeric_fields = [
        ("minimum_value", profile.minimum_value),
        ("maximum_value", profile.maximum_value),
        ("minimum_angle_degrees", profile.minimum_angle_degrees),
        ("maximum_angle_degrees", profile.maximum_angle_degrees),
        ("bounds_margin_deg", profile.bounds_margin_deg),
        ("agreement_tolerance_fraction", profile.agreement_tolerance_fraction),
        ("eval_tolerance_fraction", profile.eval_tolerance_fraction),
    ]
    if profile.operating_min is not None:
        numeric_fields.append(("operating_min", profile.operating_min))
    if profile.operating_max is not None:
        numeric_fields.append(("operating_max", profile.operating_max))

    for field_name, value in numeric_fields:
        if not _is_finite(value):
            raise CalibrationError(f"Field '{field_name}' must be a finite number, got {value}")

    # Value range checks
    if profile.minimum_value >= profile.maximum_value:
        raise CalibrationError(
            f"maximum_value ({profile.maximum_value}) must be greater than "
            f"minimum_value ({profile.minimum_value})"
        )

    # Direction check (only "cw" supported currently)
    if profile.direction != "cw":
        raise CalibrationError(
            f"Only clockwise ('cw') direction is currently supported, got '{profile.direction}'"
        )

    # Angle sweep checks
    sweep = (profile.maximum_angle_degrees - profile.minimum_angle_degrees) % 360.0
    if sweep == 0.0:
        raise CalibrationError(
            f"Angle sweep is zero (minimum_angle_degrees={profile.minimum_angle_degrees}, "
            f"maximum_angle_degrees={profile.maximum_angle_degrees})"
        )
    if sweep < 30.0 or sweep > 359.0:
        raise CalibrationError(
            f"Angle sweep must be in range [30, 359] degrees, got {sweep:.1f}° "
            f"(minimum_angle_degrees={profile.minimum_angle_degrees}, "
            f"maximum_angle_degrees={profile.maximum_angle_degrees})"
        )

    # Bounds margin checks
    if profile.bounds_margin_deg < 0.0:
        raise CalibrationError(f"bounds_margin_deg must be non-negative, got {profile.bounds_margin_deg}")
    if profile.bounds_margin_deg >= 45.0:
        raise CalibrationError(
            f"bounds_margin_deg must be less than 45 degrees, got {profile.bounds_margin_deg}"
        )

    # Operating limits checks
    if profile.operating_min is not None:
        if profile.operating_min < profile.minimum_value:
            raise CalibrationError(
                f"operating_min ({profile.operating_min}) must be >= minimum_value ({profile.minimum_value})"
            )
        if profile.operating_min > profile.maximum_value:
            raise CalibrationError(
                f"operating_min ({profile.operating_min}) must be <= maximum_value ({profile.maximum_value})"
            )

    if profile.operating_max is not None:
        if profile.operating_max < profile.minimum_value:
            raise CalibrationError(
                f"operating_max ({profile.operating_max}) must be >= minimum_value ({profile.minimum_value})"
            )
        if profile.operating_max > profile.maximum_value:
            raise CalibrationError(
                f"operating_max ({profile.operating_max}) must be <= maximum_value ({profile.maximum_value})"
            )

    if profile.operating_min is not None and profile.operating_max is not None:
        if profile.operating_min >= profile.operating_max:
            raise CalibrationError(
                f"operating_max ({profile.operating_max}) must be greater than "
                f"operating_min ({profile.operating_min})"
            )

    # Tolerance fraction checks
    if not (0.0 < profile.agreement_tolerance_fraction < 1.0):
        raise CalibrationError(
            f"agreement_tolerance_fraction must be in range (0, 1), got {profile.agreement_tolerance_fraction}"
        )
    if not (0.0 < profile.eval_tolerance_fraction < 1.0):
        raise CalibrationError(
            f"eval_tolerance_fraction must be in range (0, 1), got {profile.eval_tolerance_fraction}"
        )


# === Profile Registry ===


# Example profile for testing - NOT a real gauge model.
# Tolerance values are placeholders pending actual gauge characterization.
_EXAMPLE_PROFILE = GaugeProfile(
    name="EXAMPLE_270",
    unit="units",
    minimum_value=0.0,
    maximum_value=100.0,
    minimum_angle_degrees=225.0,  # 7:30 position
    maximum_angle_degrees=135.0,  # 4:30 position (270° sweep clockwise)
    direction="cw",
    bounds_margin_deg=5.0,
    operating_min=10.0,
    operating_max=90.0,
    agreement_tolerance_fraction=0.02,  # placeholder: 2% of scale
    eval_tolerance_fraction=0.05,  # placeholder: 5% of scale
)

PROFILES: dict[str, GaugeProfile] = {
    "EXAMPLE_270": _EXAMPLE_PROFILE,
}

# Validate all registered profiles at module load
for _profile in PROFILES.values():
    validate_profile(_profile)


def get_profile(name: str) -> GaugeProfile:
    """Retrieve a registered gauge profile by name.

    Args:
        name: Profile identifier.

    Returns:
        The requested GaugeProfile.

    Raises:
        UnknownProfileError: Profile name not registered.
    """
    if name not in PROFILES:
        raise UnknownProfileError(
            f"Profile '{name}' not found. Available profiles: {', '.join(PROFILES.keys())}"
        )
    return PROFILES[name]


# === Angle Utilities ===


def normalize_angle_deg(angle: float) -> float:
    """Normalize angle to [0, 360) range.

    Args:
        angle: Angle in degrees (any finite value).

    Returns:
        Normalized angle in [0, 360).

    Raises:
        TypeError: If angle is not a number.
        ValueError: If angle is NaN or infinity.

    Examples:
        >>> normalize_angle_deg(360.0)
        0.0
        >>> normalize_angle_deg(-90.0)
        270.0
        >>> normalize_angle_deg(725.0)
        5.0
    """
    # Type check
    if not isinstance(angle, (int, float)):
        raise TypeError(f"Angle must be a number, got {type(angle).__name__}")

    # Finite check
    if not _is_finite(angle):
        raise ValueError(f"Angle must be finite (not NaN or infinity), got {angle}")

    # Normalize to [0, 360)
    return float(angle % 360.0)


# === Legacy angle_to_value (to be extended in future tasks) ===


def angle_to_value(angle_degrees: float, profile: GaugeProfile) -> float:
    """Map an angle to a linear gauge scale.

    Angle convention: degrees increase clockwise from the image's positive x axis
    (rightward). The configured sweep from minimum to maximum must be less than
    360 degrees. Raises ValueError when the angle lies outside that sweep.

    NOTE: This is the legacy function. It will be replaced by a version returning
    CalibrationResult in a future task.
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
