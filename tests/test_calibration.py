"""Tests for gaugeguard.calibration module."""

from __future__ import annotations

import math

import pytest

from gaugeguard.calibration import (
    CalibrationError,
    CalibrationResult,
    GaugeProfile,
    UnknownProfileError,
    angle_to_value,
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


# === angle_to_value tests with EXAMPLE_270 ===
# EXAMPLE_270: min_angle=225°, max_angle=135°, sweep=270°, margin=5°
# Values: 0 to 100 units
# Scale: 225° (0) -> 270° -> 0° -> 90° -> 135° (100)


def test_angle_to_value_at_minimum():
    """Angle 225° (minimum) should map to value 0."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(225.0, profile)

    # Expected: offset = (225 - 225) % 360 = 0
    # fraction = 0 / 270 = 0
    # value = 0 + 0 * 100 = 0
    assert result.value == 0.0
    assert result.unit == "units"
    assert result.in_bounds is True
    assert result.at_scale_limit is True  # within 5° of minimum
    assert "at_scale_min" in result.flags
    assert "linear_scale_assumed" in result.flags


def test_angle_to_value_at_270():
    """Angle 270° should map to value 16.667."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(270.0, profile)

    # Expected: offset = (270 - 225) % 360 = 45
    # fraction = 45 / 270 = 0.16667
    # value = 0 + 0.16667 * 100 = 16.667
    assert result.value == pytest.approx(16.667, abs=0.001)
    assert result.in_bounds is True
    assert result.at_scale_limit is False


def test_angle_to_value_at_0():
    """Angle 0° should map to value 50."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(0.0, profile)

    # Expected: offset = (0 - 225) % 360 = 135
    # fraction = 135 / 270 = 0.5
    # value = 0 + 0.5 * 100 = 50
    assert result.value == pytest.approx(50.0, abs=0.001)
    assert result.in_bounds is True
    assert result.at_scale_limit is False


def test_angle_to_value_at_90():
    """Angle 90° should map to value 83.333."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(90.0, profile)

    # Expected: offset = (90 - 225) % 360 = 225
    # fraction = 225 / 270 = 0.83333
    # value = 0 + 0.83333 * 100 = 83.333
    assert result.value == pytest.approx(83.333, abs=0.001)
    assert result.in_bounds is True
    assert result.at_scale_limit is False


def test_angle_to_value_at_maximum():
    """Angle 135° (maximum) should map to value 100."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(135.0, profile)

    # Expected: offset = (135 - 225) % 360 = 270
    # fraction = 270 / 270 = 1.0
    # value = 0 + 1.0 * 100 = 100
    assert result.value == 100.0
    assert result.in_bounds is True
    assert result.at_scale_limit is True  # within 5° of maximum
    assert "at_scale_max" in result.flags


def test_angle_to_value_just_past_maximum_within_margin():
    """Angle 140° (5° past max) should clamp to 100."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(140.0, profile)

    # Expected: offset = (140 - 225) % 360 = 275
    # 275 is in (270, 270+5] = (270, 275]
    # Clamped to maximum value 100
    assert result.value == 100.0
    assert result.in_bounds is True
    assert result.at_scale_limit is True
    assert "at_scale_max" in result.flags


def test_angle_to_value_dead_zone_145():
    """Angle 145° (10° past max) should be in dead zone."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(145.0, profile)

    # Expected: offset = (145 - 225) % 360 = 280
    # 280 > 270 + 5 = 275, so dead zone
    assert result.value is None
    assert result.in_bounds is False
    assert result.at_scale_limit is False
    assert "angle_outside_scale" in result.flags


def test_angle_to_value_dead_zone_180():
    """Angle 180° should be in dead zone."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(180.0, profile)

    # Expected: offset = (180 - 225) % 360 = 315
    # 315 is not in [0, 270] or (270, 275] or [355, 360)
    # Dead zone
    assert result.value is None
    assert result.in_bounds is False
    assert "angle_outside_scale" in result.flags


def test_angle_to_value_just_before_minimum_within_margin():
    """Angle 220° (5° before min) should clamp to 0."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(220.0, profile)

    # Expected: offset = (220 - 225) % 360 = 355
    # 355 is in [360 - 5, 360) = [355, 360)
    # Clamped to minimum value 0
    assert result.value == 0.0
    assert result.in_bounds is True
    assert result.at_scale_limit is True
    assert "at_scale_min" in result.flags


def test_angle_to_value_normalized_585():
    """Angle 585° (normalizes to 225°) should map to value 0."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(585.0, profile)

    # Expected: 585 % 360 = 225
    # offset = (225 - 225) % 360 = 0
    # value = 0
    assert result.value == 0.0
    assert result.in_bounds is True
    assert result.at_scale_limit is True


def test_angle_to_value_none():
    """None angle should return invalid_angle flag."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(None, profile)

    assert result.value is None
    assert result.in_bounds is False
    assert "invalid_angle" in result.flags


def test_angle_to_value_nan():
    """NaN angle should return invalid_angle flag."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(float("nan"), profile)

    assert result.value is None
    assert result.in_bounds is False
    assert "invalid_angle" in result.flags


def test_angle_to_value_inf():
    """Infinity angle should return invalid_angle flag."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value(float("inf"), profile)

    assert result.value is None
    assert result.in_bounds is False
    assert "invalid_angle" in result.flags


def test_angle_to_value_non_number():
    """Non-numeric angle should return invalid_angle flag."""
    profile = get_profile("EXAMPLE_270")
    result = angle_to_value("abc", profile)  # type: ignore

    assert result.value is None
    assert result.in_bounds is False
    assert "invalid_angle" in result.flags


# === angle_to_value tests with non-wrapping profile ===
# Profile where min_angle < max_angle numerically (no 360° wrap)


def test_angle_to_value_no_wrap_profile():
    """Test with a profile where min_angle < max_angle (no wrap)."""
    # Profile: 45° to 225°, sweep = 180°, values 0 to 50, margin 10°
    profile = GaugeProfile(
        name="no_wrap_180",
        unit="bar",
        minimum_value=0.0,
        maximum_value=50.0,
        minimum_angle_degrees=45.0,
        maximum_angle_degrees=225.0,
        bounds_margin_deg=10.0,
    )

    # At minimum: 45°
    result = angle_to_value(45.0, profile)
    # offset = (45 - 45) % 360 = 0
    # fraction = 0 / 180 = 0
    # value = 0
    assert result.value == 0.0
    assert result.at_scale_limit is True
    assert "at_scale_min" in result.flags

    # Mid-scale: 135° (90° into sweep)
    result = angle_to_value(135.0, profile)
    # offset = (135 - 45) % 360 = 90
    # fraction = 90 / 180 = 0.5
    # value = 0 + 0.5 * 50 = 25
    assert result.value == pytest.approx(25.0, abs=0.001)
    assert result.in_bounds is True
    assert result.at_scale_limit is False

    # At maximum: 225°
    result = angle_to_value(225.0, profile)
    # offset = (225 - 45) % 360 = 180
    # fraction = 180 / 180 = 1.0
    # value = 50
    assert result.value == 50.0
    assert result.at_scale_limit is True
    assert "at_scale_max" in result.flags

    # Just past max (within margin): 230°
    result = angle_to_value(230.0, profile)
    # offset = (230 - 45) % 360 = 185
    # 185 is in (180, 180+10] = (180, 190]
    # Clamped to 50
    assert result.value == 50.0
    assert result.at_scale_limit is True
    assert "at_scale_max" in result.flags

    # Dead zone: 270°
    result = angle_to_value(270.0, profile)
    # offset = (270 - 45) % 360 = 225
    # 225 > 180 + 10 = 190, not in [360-10, 360) = [350, 360)
    # Dead zone
    assert result.value is None
    assert result.in_bounds is False
    assert "angle_outside_scale" in result.flags

    # Just before min (within margin): 40°
    result = angle_to_value(40.0, profile)
    # offset = (40 - 45) % 360 = 355
    # 355 is in [360 - 10, 360) = [350, 360)
    # Clamped to 0
    assert result.value == 0.0
    assert result.at_scale_limit is True
    assert "at_scale_min" in result.flags

    # Dead zone before margin: 20°
    result = angle_to_value(20.0, profile)
    # offset = (20 - 45) % 360 = 335
    # 335 is not in [0, 180] or (180, 190] or [350, 360)
    # Dead zone
    assert result.value is None
    assert result.in_bounds is False


# === CalibrationResult tests ===


def test_calibration_result_to_dict():
    """CalibrationResult.to_dict() should return JSON-serializable dict."""
    result = CalibrationResult(
        value=42.5,
        unit="PSI",
        in_bounds=True,
        at_scale_limit=False,
        within_operating_limits=True,
        value_low=40.0,
        value_high=45.0,
        flags=("linear_scale_assumed", "test_flag"),
        reason="Test reason",
    )

    d = result.to_dict()

    assert isinstance(d, dict)
    assert d["value"] == 42.5
    assert d["unit"] == "PSI"
    assert d["in_bounds"] is True
    assert d["at_scale_limit"] is False
    assert d["within_operating_limits"] is True
    assert d["value_low"] == 40.0
    assert d["value_high"] == 45.0
    assert isinstance(d["flags"], list)  # tuple converted to list
    assert d["flags"] == ["linear_scale_assumed", "test_flag"]
    assert d["reason"] == "Test reason"


def test_calibration_result_to_dict_with_none():
    """CalibrationResult.to_dict() should handle None values."""
    result = CalibrationResult(
        value=None,
        unit="units",
        in_bounds=False,
        at_scale_limit=False,
        within_operating_limits=False,
        value_low=None,
        value_high=None,
        flags=("invalid_angle",),
        reason="Angle is None",
    )

    d = result.to_dict()

    assert d["value"] is None
    assert d["value_low"] is None
    assert d["value_high"] is None


def test_angle_to_value_always_has_linear_scale_flag():
    """All results should have 'linear_scale_assumed' flag."""
    profile = get_profile("EXAMPLE_270")

    # Valid angle
    result = angle_to_value(225.0, profile)
    assert "linear_scale_assumed" in result.flags

    # Invalid angle
    result = angle_to_value(None, profile)
    assert "linear_scale_assumed" in result.flags

    # Dead zone
    result = angle_to_value(180.0, profile)
    assert "linear_scale_assumed" in result.flags
