"""Tests for gaugeguard.calibration module."""

from __future__ import annotations

import math

import pytest

from gaugeguard.calibration import (
    CalibrationError,
    GaugeProfile,
    UnknownProfileError,
    get_profile,
    normalize_angle_deg,
    validate_profile,
)


# === normalize_angle_deg tests ===


def test_normalize_angle_360_returns_0():
    """Angle 360 should normalize to 0."""
    assert normalize_angle_deg(360.0) == 0.0


def test_normalize_angle_negative_90_returns_270():
    """Angle -90 should normalize to 270."""
    assert normalize_angle_deg(-90.0) == 270.0


def test_normalize_angle_725_returns_5():
    """Angle 725 should normalize to 5."""
    assert normalize_angle_deg(725.0) == 5.0


def test_normalize_angle_zero_returns_zero():
    """Angle 0 should remain 0."""
    assert normalize_angle_deg(0.0) == 0.0


def test_normalize_angle_positive_in_range():
    """Angles already in [0, 360) should remain unchanged."""
    assert normalize_angle_deg(45.0) == 45.0
    assert normalize_angle_deg(180.0) == 180.0
    assert normalize_angle_deg(359.9) == 359.9


def test_normalize_angle_large_negative():
    """Large negative angles should wrap correctly."""
    assert normalize_angle_deg(-720.0) == 0.0
    assert normalize_angle_deg(-450.0) == 270.0


def test_normalize_angle_nan_raises_valueerror():
    """NaN should raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        normalize_angle_deg(float("nan"))


def test_normalize_angle_infinity_raises_valueerror():
    """Infinity should raise ValueError."""
    with pytest.raises(ValueError, match="finite"):
        normalize_angle_deg(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        normalize_angle_deg(float("-inf"))


def test_normalize_angle_non_number_raises_typeerror():
    """Non-numeric input should raise TypeError."""
    with pytest.raises(TypeError, match="must be a number"):
        normalize_angle_deg("45")  # type: ignore
    with pytest.raises(TypeError, match="must be a number"):
        normalize_angle_deg(None)  # type: ignore


# === validate_profile tests ===


def test_validate_profile_valid_passes():
    """A valid profile should pass validation."""
    profile = GaugeProfile(
        name="test",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=30.0,
        maximum_angle_degrees=330.0,
        direction="cw",
        bounds_margin_deg=5.0,
        operating_min=10.0,
        operating_max=90.0,
        agreement_tolerance_fraction=0.02,
        eval_tolerance_fraction=0.05,
    )
    validate_profile(profile)  # Should not raise


def test_validate_profile_example_270_passes():
    """The EXAMPLE_270 profile should pass validation."""
    profile = get_profile("EXAMPLE_270")
    validate_profile(profile)  # Should not raise


def test_validate_profile_non_finite_minimum_value():
    """Non-finite minimum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=float("nan"),
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
    )
    with pytest.raises(CalibrationError, match="minimum_value.*finite"):
        validate_profile(profile)


def test_validate_profile_non_finite_maximum_value():
    """Non-finite maximum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=float("inf"),
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
    )
    with pytest.raises(CalibrationError, match="maximum_value.*finite"):
        validate_profile(profile)


def test_validate_profile_min_value_equals_max_value():
    """minimum_value >= maximum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=100.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
    )
    with pytest.raises(CalibrationError, match="maximum_value.*must be greater than.*minimum_value"):
        validate_profile(profile)


def test_validate_profile_min_value_greater_than_max_value():
    """minimum_value > maximum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=150.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
    )
    with pytest.raises(CalibrationError, match="maximum_value.*must be greater than.*minimum_value"):
        validate_profile(profile)


def test_validate_profile_direction_not_cw():
    """Direction other than 'cw' should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        direction="ccw",
    )
    with pytest.raises(CalibrationError, match="Only clockwise.*supported"):
        validate_profile(profile)


def test_validate_profile_angle_sweep_zero():
    """Zero angle sweep should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=45.0,
        maximum_angle_degrees=45.0,
    )
    with pytest.raises(CalibrationError, match="Angle sweep is zero"):
        validate_profile(profile)


def test_validate_profile_angle_sweep_too_small():
    """Angle sweep < 30 degrees should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=25.0,
    )
    with pytest.raises(CalibrationError, match="Angle sweep must be in range.*30"):
        validate_profile(profile)


def test_validate_profile_angle_sweep_too_large():
    """Angle sweep > 359 degrees should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=10.0,
        maximum_angle_degrees=370.0,  # Results in 360° sweep after mod
    )
    with pytest.raises(CalibrationError, match="Angle sweep is zero"):
        validate_profile(profile)


def test_validate_profile_bounds_margin_negative():
    """Negative bounds_margin_deg should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        bounds_margin_deg=-1.0,
    )
    with pytest.raises(CalibrationError, match="bounds_margin_deg must be non-negative"):
        validate_profile(profile)


def test_validate_profile_bounds_margin_too_large():
    """bounds_margin_deg >= 45 should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        bounds_margin_deg=45.0,
    )
    with pytest.raises(CalibrationError, match="bounds_margin_deg must be less than 45"):
        validate_profile(profile)


def test_validate_profile_operating_min_below_minimum_value():
    """operating_min < minimum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        operating_min=-10.0,
    )
    with pytest.raises(CalibrationError, match="operating_min.*must be >= minimum_value"):
        validate_profile(profile)


def test_validate_profile_operating_min_above_maximum_value():
    """operating_min > maximum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        operating_min=110.0,
    )
    with pytest.raises(CalibrationError, match="operating_min.*must be <= maximum_value"):
        validate_profile(profile)


def test_validate_profile_operating_max_below_minimum_value():
    """operating_max < minimum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        operating_max=-10.0,
    )
    with pytest.raises(CalibrationError, match="operating_max.*must be >= minimum_value"):
        validate_profile(profile)


def test_validate_profile_operating_max_above_maximum_value():
    """operating_max > maximum_value should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        operating_max=110.0,
    )
    with pytest.raises(CalibrationError, match="operating_max.*must be <= maximum_value"):
        validate_profile(profile)


def test_validate_profile_operating_min_equals_operating_max():
    """operating_min >= operating_max should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        operating_min=50.0,
        operating_max=50.0,
    )
    with pytest.raises(CalibrationError, match="operating_max.*must be greater than.*operating_min"):
        validate_profile(profile)


def test_validate_profile_operating_min_greater_than_operating_max():
    """operating_min > operating_max should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        operating_min=70.0,
        operating_max=50.0,
    )
    with pytest.raises(CalibrationError, match="operating_max.*must be greater than.*operating_min"):
        validate_profile(profile)


def test_validate_profile_agreement_tolerance_zero():
    """agreement_tolerance_fraction = 0 should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        agreement_tolerance_fraction=0.0,
    )
    with pytest.raises(CalibrationError, match="agreement_tolerance_fraction must be in range"):
        validate_profile(profile)


def test_validate_profile_agreement_tolerance_one():
    """agreement_tolerance_fraction = 1.0 should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        agreement_tolerance_fraction=1.0,
    )
    with pytest.raises(CalibrationError, match="agreement_tolerance_fraction must be in range"):
        validate_profile(profile)


def test_validate_profile_agreement_tolerance_negative():
    """Negative agreement_tolerance_fraction should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        agreement_tolerance_fraction=-0.01,
    )
    with pytest.raises(CalibrationError, match="agreement_tolerance_fraction must be in range"):
        validate_profile(profile)


def test_validate_profile_eval_tolerance_zero():
    """eval_tolerance_fraction = 0 should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        eval_tolerance_fraction=0.0,
    )
    with pytest.raises(CalibrationError, match="eval_tolerance_fraction must be in range"):
        validate_profile(profile)


def test_validate_profile_eval_tolerance_one():
    """eval_tolerance_fraction = 1.0 should raise CalibrationError."""
    profile = GaugeProfile(
        name="bad",
        unit="PSI",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
        eval_tolerance_fraction=1.0,
    )
    with pytest.raises(CalibrationError, match="eval_tolerance_fraction must be in range"):
        validate_profile(profile)


# === get_profile tests ===


def test_get_profile_example_270_exists():
    """EXAMPLE_270 profile should be retrievable."""
    profile = get_profile("EXAMPLE_270")
    assert profile.name == "EXAMPLE_270"
    assert profile.unit == "units"
    assert profile.minimum_value == 0.0
    assert profile.maximum_value == 100.0
    assert profile.minimum_angle_degrees == 225.0
    assert profile.maximum_angle_degrees == 135.0
    assert profile.direction == "cw"
    assert profile.bounds_margin_deg == 5.0
    assert profile.operating_min == 10.0
    assert profile.operating_max == 90.0
    assert profile.agreement_tolerance_fraction == 0.02
    assert profile.eval_tolerance_fraction == 0.05


def test_get_profile_unknown_name_raises():
    """Unknown profile name should raise UnknownProfileError."""
    with pytest.raises(UnknownProfileError, match="Profile 'NONEXISTENT' not found"):
        get_profile("NONEXISTENT")


def test_get_profile_unknown_name_lists_available():
    """UnknownProfileError message should list available profiles."""
    with pytest.raises(UnknownProfileError, match="Available profiles:.*EXAMPLE_270"):
        get_profile("UNKNOWN")


# === GaugeProfile dataclass tests ===


def test_gaugeprofile_is_frozen():
    """GaugeProfile should be immutable (frozen)."""
    profile = get_profile("EXAMPLE_270")
    with pytest.raises(Exception):  # FrozenInstanceError in Python 3.10+
        profile.name = "MODIFIED"  # type: ignore


def test_gaugeprofile_default_values():
    """GaugeProfile should have correct default values for optional fields."""
    profile = GaugeProfile(
        name="minimal",
        unit="bar",
        minimum_value=0.0,
        maximum_value=10.0,
        minimum_angle_degrees=0.0,
        maximum_angle_degrees=180.0,
    )
    assert profile.direction == "cw"
    assert profile.bounds_margin_deg == 5.0
    assert profile.operating_min is None
    assert profile.operating_max is None
    assert profile.agreement_tolerance_fraction == 0.02
    assert profile.eval_tolerance_fraction == 0.05
