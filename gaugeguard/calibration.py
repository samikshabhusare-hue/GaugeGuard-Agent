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


# === Calibration Result ===


@dataclass(frozen=True)
class CalibrationResult:
    """Result from converting a needle angle to a gauge reading.

    Assumption: Linear scale mapping between angle and value.
    """
    value: float | None  # None if angle invalid or out of bounds
    unit: str
    in_bounds: bool  # angle within profile's min/max sweep
    at_scale_limit: bool  # near endpoints (within bounds_margin_deg)
    within_operating_limits: bool  # in safe range if defined
    value_low: float | None  # lower bound if uncertainty provided (future)
    value_high: float | None  # upper bound if uncertainty provided (future)
    flags: tuple[str, ...]  # e.g., "invalid_angle", "angle_outside_scale", "at_scale_min"
    reason: str  # human-readable explanation

    def to_dict(self) -> dict:
        """Return JSON-serializable dict (only built-in types)."""
        return {
            "value": self.value,
            "unit": self.unit,
            "in_bounds": self.in_bounds,
            "at_scale_limit": self.at_scale_limit,
            "within_operating_limits": self.within_operating_limits,
            "value_low": self.value_low,
            "value_high": self.value_high,
            "flags": list(self.flags),
            "reason": self.reason,
        }


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


# === Angle to Value Conversion ===


def angle_to_value(
    angle_deg: float | None,
    profile: GaugeProfile,
    angle_uncertainty_deg: float | None = None
) -> CalibrationResult:
    """Convert needle angle to gauge reading with bounds checking.

    Angle convention: 0° = 12 o'clock, clockwise positive.
    Scale starts at profile.minimum_angle_degrees (minimum_value) and sweeps
    clockwise to profile.maximum_angle_degrees (maximum_value).

    Assumption: Linear scale mapping (flag "linear_scale_assumed" added to all results).

    Args:
        angle_deg: Measured needle angle in degrees. None/NaN/inf treated as invalid.
        profile: Gauge calibration profile.
        angle_uncertainty_deg: Optional ±uncertainty in degrees (reserved for future use).

    Returns:
        CalibrationResult with value, bounds flags, and reason.

        Angle handling:
        - Within scale sweep [0, sweep]: Linear interpolation, in_bounds=True
        - Within margin past max (sweep, sweep+margin]: Clamped to max_value,
          at_scale_limit=True, flag "at_scale_max"
        - Within margin before min [360-margin, 360): Clamped to min_value,
          at_scale_limit=True, flag "at_scale_min"
        - Dead zone (outside scale and margins): value=None, in_bounds=False,
          flag "angle_outside_scale"
        - Invalid angle (None/NaN/inf/non-number): value=None, in_bounds=False,
          flag "invalid_angle"

    Raises:
        CalibrationError: Invalid profile configuration (programmer error).

    Notes:
        - Never returns a reading for an angle outside the scale and margins.
        - Bad measured data returns flagged result, never raises exception.
        - Operating limits and uncertainty ranges not yet implemented.
    """
    # Validate profile (programmer error if invalid)
    validate_profile(profile)

    # Base flags always include linear scale assumption
    flags_list = ["linear_scale_assumed"]

    # Handle invalid angle input (None, NaN, inf, non-number)
    if angle_deg is None:
        return CalibrationResult(
            value=None,
            unit=profile.unit,
            in_bounds=False,
            at_scale_limit=False,
            within_operating_limits=False,
            value_low=None,
            value_high=None,
            flags=tuple(flags_list + ["invalid_angle"]),
            reason="Angle is None; cannot compute reading.",
        )

    if not isinstance(angle_deg, (int, float)):
        return CalibrationResult(
            value=None,
            unit=profile.unit,
            in_bounds=False,
            at_scale_limit=False,
            within_operating_limits=False,
            value_low=None,
            value_high=None,
            flags=tuple(flags_list + ["invalid_angle"]),
            reason=f"Angle must be a number, got {type(angle_deg).__name__}.",
        )

    if not _is_finite(angle_deg):
        return CalibrationResult(
            value=None,
            unit=profile.unit,
            in_bounds=False,
            at_scale_limit=False,
            within_operating_limits=False,
            value_low=None,
            value_high=None,
            flags=tuple(flags_list + ["invalid_angle"]),
            reason=f"Angle must be finite (not NaN or infinity), got {angle_deg}.",
        )

    # Normalize angle to [0, 360)
    angle_norm = normalize_angle_deg(angle_deg)

    # Compute sweep and offset
    sweep = (profile.maximum_angle_degrees - profile.minimum_angle_degrees) % 360.0
    offset = (angle_norm - profile.minimum_angle_degrees) % 360.0
    margin = profile.bounds_margin_deg

    # Case 1: Within scale sweep [0, sweep]
    if offset <= sweep:
        fraction = offset / sweep
        value = profile.minimum_value + fraction * (profile.maximum_value - profile.minimum_value)

        # Check if at scale limits (within margin of endpoints)
        at_limit = False
        if offset <= margin:
            at_limit = True
            flags_list.append("at_scale_min")
        elif offset >= sweep - margin:
            at_limit = True
            flags_list.append("at_scale_max")

        return CalibrationResult(
            value=value,
            unit=profile.unit,
            in_bounds=True,
            at_scale_limit=at_limit,
            within_operating_limits=False,  # TODO: implement in next prompt
            value_low=None,  # TODO: implement uncertainty
            value_high=None,  # TODO: implement uncertainty
            flags=tuple(flags_list),
            reason=f"Angle {angle_norm:.1f}° mapped to value {value:.2f} {profile.unit}.",
        )

    # Case 2: Within margin past maximum (sweep, sweep + margin]
    if sweep < offset <= sweep + margin:
        return CalibrationResult(
            value=profile.maximum_value,
            unit=profile.unit,
            in_bounds=True,
            at_scale_limit=True,
            within_operating_limits=False,  # TODO: implement in next prompt
            value_low=None,
            value_high=None,
            flags=tuple(flags_list + ["at_scale_max"]),
            reason=(
                f"Angle {angle_norm:.1f}° is {offset - sweep:.1f}° past maximum scale angle; "
                f"clamped to maximum value {profile.maximum_value:.2f} {profile.unit}."
            ),
        )

    # Case 3: Within margin before minimum [360 - margin, 360)
    if offset >= 360.0 - margin:
        return CalibrationResult(
            value=profile.minimum_value,
            unit=profile.unit,
            in_bounds=True,
            at_scale_limit=True,
            within_operating_limits=False,  # TODO: implement in next prompt
            value_low=None,
            value_high=None,
            flags=tuple(flags_list + ["at_scale_min"]),
            reason=(
                f"Angle {angle_norm:.1f}° is {360.0 - offset:.1f}° before minimum scale angle; "
                f"clamped to minimum value {profile.minimum_value:.2f} {profile.unit}."
            ),
        )

    # Case 4: Dead zone (outside scale and margins)
    return CalibrationResult(
        value=None,
        unit=profile.unit,
        in_bounds=False,
        at_scale_limit=False,
        within_operating_limits=False,
        value_low=None,
        value_high=None,
        flags=tuple(flags_list + ["angle_outside_scale"]),
        reason=(
            f"Angle {angle_norm:.1f}° is outside the gauge scale "
            f"[{profile.minimum_angle_degrees:.1f}° to {profile.maximum_angle_degrees:.1f}° "
            f"with {margin:.1f}° margin]. Dead zone; no reading available."
        ),
    )
