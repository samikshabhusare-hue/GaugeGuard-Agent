"""Tests for gaugeguard.agent.inspect_upload integration."""

from __future__ import annotations

import math
import cv2
import numpy as np
import pytest

from gaugeguard.agent import inspect_upload
from gaugeguard.calibration import GaugeProfile, get_profile


# A fully-specified, verified industrial profile for testing accepted readings
_VERIFIED_PROFILE = GaugeProfile(
    name="VERIFIED_270_PSI",
    unit="psi",
    minimum_value=0.0,
    maximum_value=100.0,
    minimum_angle_degrees=225.0,
    maximum_angle_degrees=135.0,
    direction="cw",
    bounds_margin_deg=5.0,
    operating_min=10.0,
    operating_max=90.0,
    agreement_tolerance_fraction=0.02,
    eval_tolerance_fraction=0.05,
)


def _create_synthetic_gauge_bytes(
    angle_deg: float,
    dial_radius: int = 120,
    size: int = 320,
) -> bytes:
    """Generate JPEG encoded bytes of a synthetic gauge for agent inspection."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    cx, cy = size // 2, size // 2

    # Dial casing and face
    cv2.circle(img, (cx, cy), dial_radius + 12, (50, 50, 50), -1)
    cv2.circle(img, (cx, cy), dial_radius, (250, 250, 250), -1)
    cv2.circle(img, (cx, cy), dial_radius, (80, 80, 80), 3)

    # Scale ticks
    for tick_angle in range(0, 360, 30):
        rad = math.radians(tick_angle)
        dx = math.sin(rad)
        dy = -math.cos(rad)
        p1 = (int(cx + dx * (dial_radius - 14)), int(cy + dy * (dial_radius - 14)))
        p2 = (int(cx + dx * (dial_radius - 4)), int(cy + dy * (dial_radius - 4)))
        cv2.line(img, p1, p2, (100, 100, 100), 2)

    # Draw needle
    rad = math.radians(angle_deg)
    dx = math.sin(rad)
    dy = -math.cos(rad)
    needle_len = dial_radius * 0.75
    tip_x = int(cx + dx * needle_len)
    tip_y = int(cy + dy * needle_len)

    tail_len = dial_radius * 0.12
    tail_x = int(cx - dx * tail_len)
    tail_y = int(cy - dy * tail_len)
    cv2.line(img, (tail_x, tail_y), (tip_x, tip_y), (25, 25, 25), 4)
    cv2.circle(img, (cx, cy), 8, (45, 45, 45), -1)

    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_inspect_upload_empty_raises():
    """Empty byte input must raise ValueError."""
    with pytest.raises(ValueError):
        inspect_upload(b"")


def test_inspect_upload_synthetic_gauge_accepted():
    """Clean synthetic gauge with verified profile at 0° (midpoint of 270° scale) must return accepted reading ~50.0 psi."""
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    result = inspect_upload(img_bytes, profile=_VERIFIED_PROFILE)

    assert result.reading.value is not None
    assert abs(result.reading.value - 50.0) <= 2.5
    assert result.reading.unit == "psi"
    assert result.reading.angle_degrees is not None
    assert abs(result.reading.angle_degrees - 0.0) <= 2.0 or abs(result.reading.angle_degrees - 360.0) <= 2.0
    assert result.reading.status == "accepted"
    assert not result.needs_human
    assert result.annotated_image is not None
    assert len(result.action_log) >= 4
    step_names = [s.step_name for s in result.action_log]
    assert "image_quality" in step_names
    assert "initial_detection" in step_names
    assert "calibration_mapping" in step_names
    assert "final_decision" in step_names
    assert len(result.attempts) >= 1


def test_inspect_upload_dead_zone_rejects_numeric_reading():
    """Angle in dead zone (e.g. 180° for EXAMPLE_270) must NEVER return a numeric reading."""
    profile = get_profile("EXAMPLE_270")
    # 180° is in the dead zone (140° to 220°)
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=180.0)

    result = inspect_upload(img_bytes, profile=profile)

    assert result.reading.value is None, "Must never return numeric reading for out-of-scale dead zone"
    assert result.reading.status == "needs_human"
    assert result.needs_human
    assert any("angle_outside_scale" in step.output_summary.get("flags", []) for step in result.action_log)


def test_inspect_upload_detection_failure_rejects_numeric_reading():
    """Blank image with no circular dial must return no numeric reading and require human review/retake."""
    blank = np.full((300, 300, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok

    result = inspect_upload(buf.tobytes())

    assert result.reading.value is None, "Must never return numeric reading when detection fails"
    assert result.needs_human
    assert result.reading.status in ("needs_human", "needs_retake")
    # Action log must track attempts and final decision
    step_names = [s.step_name for s in result.action_log]
    assert "final_decision" in step_names
    assert len(result.attempts) >= 1


def test_inspect_upload_retry_and_comparison_disagreement(monkeypatch):
    """When initial and retry detections yield conflicting angles, readings_agree() rejects acceptance."""
    profile = get_profile("EXAMPLE_270")
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    # Simulate an initial detection that has low confidence (triggering retry)
    from gaugeguard import agent
    from gaugeguard.vision import NeedleDetectionResult

    real_detect = agent.detect_dial_and_needle
    call_count = 0

    def mock_detect(image):
        nonlocal call_count
        call_count += 1
        real_res = real_detect(image)
        if call_count == 1:
            # First pass: low confidence, angle at 0.0° (value ~50.0)
            return NeedleDetectionResult(
                success=True,
                center=real_res.center,
                radius=real_res.radius,
                needle_tip=real_res.needle_tip,
                angle_deg=0.0,
                confidence=0.50,  # Low confidence triggers retry
                quality_flags=("low_confidence",),
                annotated_image=real_res.annotated_image,
                reason="Simulated marginal confidence.",
            )
        else:
            # Retry pass: returns a substantially conflicting angle at 90.0° (value ~83.3)
            return NeedleDetectionResult(
                success=True,
                center=real_res.center,
                radius=real_res.radius,
                needle_tip=real_res.needle_tip,
                angle_deg=90.0,
                confidence=0.85,
                quality_flags=(),
                annotated_image=real_res.annotated_image,
                reason="Simulated conflicting retry detection.",
            )

    monkeypatch.setattr(agent, "detect_dial_and_needle", mock_detect)

    result = inspect_upload(img_bytes, profile=profile)

    # Disagreement must reject reading and require human review
    assert result.needs_human
    assert result.reading.value is None
    assert result.reading.status == "needs_human"
    assert "Readings disagree" in result.reading.reason

    # Verify action_log contains reading_comparison step
    step_names = [s.step_name for s in result.action_log]
    assert "reading_comparison" in step_names
    cmp_step = next(s for s in result.action_log if s.step_name == "reading_comparison")
    assert not cmp_step.output_summary["agree"]
    assert cmp_step.output_summary["difference"] is not None


def test_inspect_upload_retry_and_comparison_agreement(monkeypatch):
    """When initial and retry detections agree within tolerance on verified profile, reading is accepted with corroboration."""
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    from gaugeguard import agent
    from gaugeguard.vision import NeedleDetectionResult

    real_detect = agent.detect_dial_and_needle
    call_count = 0

    def mock_detect(image):
        nonlocal call_count
        call_count += 1
        real_res = real_detect(image)
        if call_count == 1:
            # First pass: low confidence, angle at 0.0°
            return NeedleDetectionResult(
                success=True,
                center=real_res.center,
                radius=real_res.radius,
                needle_tip=real_res.needle_tip,
                angle_deg=0.0,
                confidence=0.55,  # Low confidence triggers retry
                quality_flags=("low_confidence",),
                annotated_image=real_res.annotated_image,
                reason="Simulated low confidence initial pass.",
            )
        else:
            # Retry pass: agrees within tolerance (e.g. 0.5° difference)
            return NeedleDetectionResult(
                success=True,
                center=real_res.center,
                radius=real_res.radius,
                needle_tip=real_res.needle_tip,
                angle_deg=0.5,
                confidence=0.88,
                quality_flags=(),
                annotated_image=real_res.annotated_image,
                reason="Simulated consistent retry pass.",
            )

    monkeypatch.setattr(agent, "detect_dial_and_needle", mock_detect)

    result = inspect_upload(img_bytes, profile=_VERIFIED_PROFILE)

    # Corroborated reading must be accepted
    assert not result.needs_human
    assert result.reading.status == "accepted"
    assert result.reading.value is not None
    assert abs(result.reading.value - 50.0) <= 2.5

    # Verify action_log contains comparison step confirming agreement
    cmp_step = next(s for s in result.action_log if s.step_name == "reading_comparison")
    assert cmp_step.output_summary["agree"]


# ---------------------------------------------------------------------------
# Additional audit tests: edge cases, invariants, and escalation paths
# ---------------------------------------------------------------------------


def test_inspect_upload_corrupted_bytes_raises():
    """Non-image bytes (corrupted data) must raise ValueError, not crash."""
    garbage = b"\x00\x01\x02\xff" * 100
    with pytest.raises(ValueError, match="could not be decoded"):
        inspect_upload(garbage)


def test_inspect_upload_truncated_jpeg_raises():
    """Truncated JPEG header must raise ValueError."""
    truncated = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    with pytest.raises(ValueError):
        inspect_upload(truncated)


def test_inspect_upload_scale_limit_escalates_human_review():
    """Needle near scale minimum (within margin) must set needs_human=True even though a value is returned."""
    profile = get_profile("EXAMPLE_270")
    # Angle 225° is exactly at minimum_angle_degrees (value 0.0, at_scale_min)
    # Offset = (225 - 225) % 360 = 0, which is <= margin (5.0)
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=225.0)

    result = inspect_upload(img_bytes, profile=profile)

    # at_scale_limit triggers human review but may still have a value
    assert result.needs_human
    assert result.reading.status == "needs_human"


def test_inspect_upload_outside_operating_limits_escalates():
    """A reading outside operating_min..operating_max must escalate to human review."""
    profile = get_profile("EXAMPLE_270")
    # EXAMPLE_270: operating_min=10, operating_max=90
    # At angle 228° (3° past min_angle=225°), value ≈ 1.11 which is below operating_min=10
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=228.0)

    result = inspect_upload(img_bytes, profile=profile)

    # Value may be computed, but must be flagged for human review
    if result.reading.value is not None:
        assert result.needs_human, "Reading outside operating limits must escalate"
        assert result.reading.status == "needs_human"


def test_inspect_upload_retry_steps_in_action_log(monkeypatch):
    """When initial detection has low confidence, retry steps must appear in action_log."""
    profile = get_profile("EXAMPLE_270")
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    from gaugeguard import agent
    from gaugeguard.vision import NeedleDetectionResult

    real_detect = agent.detect_dial_and_needle
    call_count = 0

    def mock_detect(image):
        nonlocal call_count
        call_count += 1
        real_res = real_detect(image)
        if call_count == 1:
            # First pass: force low confidence to trigger retry
            return NeedleDetectionResult(
                success=True,
                center=real_res.center,
                radius=real_res.radius,
                needle_tip=real_res.needle_tip,
                angle_deg=0.0,
                confidence=0.50,
                quality_flags=("low_confidence",),
                annotated_image=real_res.annotated_image,
                reason="Simulated low confidence.",
            )
        return real_res

    monkeypatch.setattr(agent, "detect_dial_and_needle", mock_detect)

    result = inspect_upload(img_bytes, profile=profile)

    # Verify retry step(s) are logged
    step_names = [s.step_name for s in result.action_log]
    retry_steps = [n for n in step_names if n.startswith("retry_")]
    assert len(retry_steps) >= 1, f"Expected retry steps, got: {step_names}"

    # Verify attempts list tracks >1 attempt
    assert len(result.attempts) >= 2


def test_accepted_reading_invariants():
    """If status == 'accepted', then value must be non-None and needs_human must be False."""
    profile = get_profile("EXAMPLE_270")
    # 0° => value ~50.0, within operating limits, high confidence on synthetic gauge
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    result = inspect_upload(img_bytes, profile=profile)

    if result.reading.status == "accepted":
        assert result.reading.value is not None, "Accepted reading must have a value"
        assert not result.needs_human, "Accepted reading must not require human review"
        assert result.reading.confidence > 0.0, "Accepted reading must have positive confidence"
        assert result.reading.angle_degrees is not None, "Accepted reading must have detected angle"


def test_needs_human_never_accepted_with_none_value():
    """Cross-check: if value is None, status must never be 'accepted'."""
    profile = get_profile("EXAMPLE_270")
    # Dead zone angle: detection succeeds but calibration rejects
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=180.0)

    result = inspect_upload(img_bytes, profile=profile)

    if result.reading.value is None:
        assert result.reading.status != "accepted", "None value must never have 'accepted' status"
        assert result.needs_human, "None value must require human review"


def test_action_log_always_has_final_decision():
    """Every inspection must end with a final_decision step, regardless of outcome."""
    profile = get_profile("EXAMPLE_270")

    # Case 1: successful detection
    good_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)
    result_good = inspect_upload(good_bytes, profile=profile)
    step_names_good = [s.step_name for s in result_good.action_log]
    assert "final_decision" in step_names_good

    # Case 2: detection failure (blank image)
    blank = np.full((300, 300, 3), 200, dtype=np.uint8)
    ok, buf = cv2.imencode(".png", blank)
    assert ok
    result_fail = inspect_upload(buf.tobytes(), profile=profile)
    step_names_fail = [s.step_name for s in result_fail.action_log]
    assert "final_decision" in step_names_fail


def test_inspect_upload_all_flags_never_fabricated():
    """Verify flags in final_decision step are sourced from actual quality/detection/calibration, not invented."""
    profile = get_profile("EXAMPLE_270")
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    result = inspect_upload(img_bytes, profile=profile)

    dec_step = next(s for s in result.action_log if s.step_name == "final_decision")
    reported_flags = dec_step.output_summary.get("flags", [])

    # Every reported flag must come from quality, detection, or calibration
    valid_sources = (
        set(result.quality.flags)
        | set(result.action_log[1].output_summary.get("flags", []))  # initial_detection
    )
    # Also include calibration flags (minus linear_scale_assumed)
    cal_step = next((s for s in result.action_log if s.step_name == "calibration_mapping"), None)
    if cal_step:
        valid_sources |= {f for f in cal_step.output_summary.get("flags", []) if f != "linear_scale_assumed"}

    for flag in reported_flags:
        assert flag in valid_sources, f"Flag '{flag}' not traceable to quality/detection/calibration"


def test_inspect_upload_demo_profile_always_requires_human_review():
    """EXAMPLE_270 uses placeholder units and must always require human review (cannot be marked accepted)."""
    profile = get_profile("EXAMPLE_270")
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    result = inspect_upload(img_bytes, profile=profile)

    # Reading value must be preserved as provisional
    assert result.reading.value is not None
    assert abs(result.reading.value - 50.0) <= 2.5
    assert result.reading.unit == "units"
    assert result.reading.angle_degrees is not None

    # Must require human review and cannot be marked accepted
    assert result.needs_human is True
    assert result.reading.status == "needs_human"
    assert "unverified_demo_profile" in result.reading.reason
    dec_step = next(s for s in result.action_log if s.step_name == "final_decision")
    assert "unverified_demo_profile" in dec_step.output_summary.get("flags", [])


def test_inspect_upload_default_profile_escalates_as_unverified_demo():
    """Default profile (when profile=None is passed) must safely escalate to human review."""
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    result = inspect_upload(img_bytes, profile=None)

    assert result.reading.value is not None
    assert result.needs_human is True
    assert result.reading.status == "needs_human"
    assert "unverified_demo_profile" in result.reading.reason


def test_inspect_upload_custom_profile_with_placeholder_units_escalates():
    """Any custom profile with placeholder units ('units') must escalate to human review."""
    placeholder_profile = GaugeProfile(
        name="CUSTOM_UNVERIFIED_GAUGE",
        unit="units",
        minimum_value=0.0,
        maximum_value=100.0,
        minimum_angle_degrees=225.0,
        maximum_angle_degrees=135.0,
        direction="cw",
    )
    img_bytes = _create_synthetic_gauge_bytes(angle_deg=0.0)

    result = inspect_upload(img_bytes, profile=placeholder_profile)

    assert result.reading.value is not None
    assert result.needs_human is True
    assert result.reading.status == "needs_human"
    assert "unverified_demo_profile" in result.reading.reason

