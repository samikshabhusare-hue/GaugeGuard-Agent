"""Scenario tests covering required scenarios using available sample images and labels.

Rules:
- Tests assert behavior from the contract and labels.
- For any test that fails because of another module, mark it clearly with the owner's name
  and keep it failing (do not weaken the assertion).
- If no image/label exists for a scenario, the test is skipped with a message
  "MISSING DATA: <what to collect>".

Run with: pytest tests/test_scenarios.py -q -rs
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np
import pytest

from gaugeguard.agent import inspect_upload
from gaugeguard.calibration import (
    AgreementResult,
    CalibrationResult,
    angle_to_value,
    get_profile,
    readings_agree,
)

SAMPLE_IMAGES_DIR = "sample_images"
SAMPLE_LABELS_PATH = os.path.join(SAMPLE_IMAGES_DIR, "labels.csv")


def _load_sample_labels() -> list[dict]:
    rows = []
    if not os.path.exists(SAMPLE_LABELS_PATH):
        return rows
    with open(SAMPLE_LABELS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("filename"):
                rows.append(row)
    return rows


SAMPLE_LABELS = _load_sample_labels()


# ---------------------------------------------------------------------------
# Scenario 1: Clear image – accepted, error within tolerance
# ---------------------------------------------------------------------------

@pytest.mark.group1
class TestClearImage:
    def test_clear_image_reading_exists_and_near_ground_truth(self):
        """Clear image with expected_outcome=read must be accepted within tolerance.

        Label: industrial_gauge.png, EXAMPLE_270, true_value=66.73,
               label_uncertainty=0.74, expected_outcome=read.
        Contract: status must be 'accepted', needs_human must be False.
        """
        label = next((r for r in SAMPLE_LABELS if r.get("condition") == "clear"), None)
        if label is None:
            pytest.skip("MISSING DATA: clear image label in sample_images/labels.csv")
        fixture_path = os.path.join(SAMPLE_IMAGES_DIR, label["filename"])
        with open(fixture_path, "rb") as f:
            image_bytes = f.read()
        profile = get_profile(label.get("gauge_profile", "EXAMPLE_270"))
        result = inspect_upload(image_bytes, profile=profile)
        assert result.reading.value is not None, (
            "FAILURE (agent.py / Kshitij): expected numeric reading for clear image"
        )
        true_val = float(label["true_value"])
        uncert = float(label["label_uncertainty"])
        assert abs(result.reading.value - true_val) <= uncert, (
            f"FAILURE: value {result.reading.value} vs truth {true_val} "
            f"(tolerance ±{uncert})"
        )
        # Contract + label: expected_outcome=read means status must be 'accepted'.
        # Do NOT weaken this assertion. If agent.py escalates a clean demo reading
        # to needs_human via the unverified_demo_profile flag, that is a bug in
        # agent.py that must be fixed by its owner, not masked in tests.
        assert result.reading.status == "accepted", (
            "FAILURE (agent.py / Kshitij): expected accepted, got "
            + str(result.reading.status)
            + ". Demo profile (EXAMPLE_270) sets unverified_demo_profile flag which "
            "forces needs_human; agent must not escalate a clean demo reading."
        )
        assert not result.needs_human, (
            "FAILURE (agent.py / Kshitij): needs_human must be False when status is accepted"
        )


# ---------------------------------------------------------------------------
# Scenario 2: Blurry image
# ---------------------------------------------------------------------------

@pytest.mark.group2
class TestBlurryImage:
    def test_blurry_image_label_exists(self):
        """No blurry image or label exists in sample data — skip with MISSING DATA."""
        label = next((r for r in SAMPLE_LABELS if r.get("condition") == "blurry"), None)
        if label is None:
            pytest.skip(
                "MISSING DATA: blurry condition image file and corresponding "
                "sample_images/labels.csv entry (condition=blurry, "
                "expected_outcome=needs_retake or needs_human)"
            )


# ---------------------------------------------------------------------------
# Scenario 3: Tilted image
# ---------------------------------------------------------------------------

@pytest.mark.group3
class TestTiltedImage:
    def test_tilted_image_label_exists(self):
        """No tilted image or label exists in sample data — skip with MISSING DATA."""
        label = next((r for r in SAMPLE_LABELS if r.get("condition") == "tilted"), None)
        if label is None:
            pytest.skip(
                "MISSING DATA: tilted condition image file and corresponding "
                "sample_images/labels.csv entry (condition=tilted, "
                "expected_outcome=read/reject_input)"
            )


# ---------------------------------------------------------------------------
# Scenario 4: Poorly illuminated image
# ---------------------------------------------------------------------------

@pytest.mark.group4
class TestDarkImage:
    def test_dark_image_label_exists(self):
        """No poorly-illuminated image or label exists in sample data — skip with MISSING DATA."""
        label = next(
            (r for r in SAMPLE_LABELS if r.get("condition") in ("dark", "poorly_illuminated")),
            None,
        )
        if label is None:
            pytest.skip(
                "MISSING DATA: dark/poorly-illuminated condition image file and "
                "corresponding sample_images/labels.csv entry (condition=dark, "
                "expected_outcome=read/reject_input)"
            )


# ---------------------------------------------------------------------------
# Scenario 5: Partially obstructed image
# ---------------------------------------------------------------------------

@pytest.mark.group5
class TestObstructedImage:
    def test_obstructed_image_has_documented_retry_or_needs_human(self):
        """No obstructed image or label exists in sample data — skip with MISSING DATA."""
        label = next(
            (r for r in SAMPLE_LABELS if r.get("condition") in ("obstructed", "partially_obstructed")),
            None,
        )
        if label is None:
            pytest.skip(
                "MISSING DATA: obstructed image file and corresponding "
                "sample_images/labels.csv entry specifying expected_outcome "
                "(needs_human/retry/reject_input)"
            )


# ---------------------------------------------------------------------------
# Scenario 6: Corrupted file and unsupported file → rejected_input, no crash
# ---------------------------------------------------------------------------

@pytest.mark.group6
class TestCorruptedAndUnsupported:
    def test_corrupted_bytes_return_rejected_without_crash(self, tmp_path):
        """Garbage bytes that cv2 cannot decode must return rejected_input, not crash."""
        garbage_path = tmp_path / "garbage.png"
        garbage_path.write_bytes(b"\x00\x01\xff\xfe" * 64)
        with open(garbage_path, "rb") as f:
            bytes_data = f.read()
        try:
            result = inspect_upload(bytes_data)
        except ValueError as exc:
            pytest.fail(
                "FAILURE (agent.py / Kshitij): corrupted bytes caused a crash "
                "(ValueError) instead of returning rejected_input gracefully. "
                "Exception: " + str(exc)
            )
        assert result.reading.status == "rejected_input", (
            "FAILURE (agent.py / Kshitij): expected rejected_input for corrupted file, got "
            + str(getattr(result, "reading", None))
        )

    def test_unsupported_text_as_png_return_rejected_without_crash(self, tmp_path):
        """A .txt file renamed to .png must return rejected_input, not crash."""
        fake_path = tmp_path / "fake_image.png"
        fake_path.write_text("This is not an image. Just text content.")
        with open(fake_path, "rb") as f:
            bytes_data = f.read()
        try:
            result = inspect_upload(bytes_data)
        except ValueError as exc:
            pytest.fail(
                "FAILURE (agent.py / Kshitij): unsupported file caused a crash "
                "(ValueError) instead of returning rejected_input gracefully. "
                "Exception: " + str(exc)
            )
        assert result.reading.status == "rejected_input", (
            "FAILURE (agent.py / Kshitij): expected rejected_input for unsupported file, got "
            + str(getattr(result, "reading", None))
        )


# ---------------------------------------------------------------------------
# Scenario 7: Uncertain reading – value_low/value_high present, range > 0
# ---------------------------------------------------------------------------

@pytest.mark.group7
class TestUncertainReading:
    def test_uncertainty_range_is_positive(self):
        """When angle_uncertainty_deg is provided, calibration must return a range > 0."""
        profile = get_profile("EXAMPLE_270")
        result = angle_to_value(0.0, profile, angle_uncertainty_deg=10.0)
        assert isinstance(result, CalibrationResult)
        assert result.value_low is not None, (
            "FAILURE (calibration.py / Prasoon): value_low missing when uncertainty provided"
        )
        assert result.value_high is not None, (
            "FAILURE (calibration.py / Prasoon): value_high missing when uncertainty provided"
        )
        assert (result.value_high - result.value_low) > 0, (
            "FAILURE (calibration.py / Prasoon): uncertainty range must be > 0 "
            "for non-zero angle_uncertainty_deg"
        )


# ---------------------------------------------------------------------------
# Scenario 8: Reading outside calibration bounds – dead zone → value None,
#             in_bounds False, never accepted by the agent
# ---------------------------------------------------------------------------

@pytest.mark.group8
class TestDeadZone:
    def test_dead_zone_angle_returns_none_and_in_bounds_false(self):
        """Angle in dead zone must yield value=None, in_bounds=False, angle_outside_scale flag."""
        profile = get_profile("EXAMPLE_270")
        result = angle_to_value(180.0, profile)
        assert result.value is None, (
            "FAILURE (calibration.py / Prasoon): dead zone must return value=None"
        )
        assert result.in_bounds is False, (
            "FAILURE (calibration.py / Prasoon): dead zone must set in_bounds=False"
        )
        assert "angle_outside_scale" in result.flags, (
            "FAILURE (calibration.py / Prasoon): dead zone must include 'angle_outside_scale' flag"
        )

    def test_agent_never_accepts_dead_zone(self, monkeypatch):
        """Agent must not accept a reading when the needle is in the dead zone.

        Uses monkeypatch to force detect_dial_and_needle to return a dead-zone
        angle (180°), then verifies agent.inspect_upload returns value=None,
        status != 'accepted', needs_human=True. No edits to agent.py.
        """
        from gaugeguard.vision import NeedleDetectionResult

        profile = get_profile("EXAMPLE_270")
        with open(os.path.join(SAMPLE_IMAGES_DIR, "industrial_gauge.png"), "rb") as f:
            image_bytes = f.read()

        dead_zone_detection = NeedleDetectionResult(
            success=True,
            center=(100.0, 100.0),
            radius=50.0,
            needle_tip=(100.0, 150.0),
            angle_deg=180.0,
            confidence=0.90,
            quality_flags=(),
            annotated_image=None,
            reason="dead zone angle test",
        )
        monkeypatch.setattr(
            "gaugeguard.agent.detect_dial_and_needle",
            lambda img: dead_zone_detection,
        )

        result = inspect_upload(image_bytes, profile=profile)
        assert result.reading.value is None, (
            "FAILURE (agent.py / Kshitij): agent must return value=None for angle in dead zone"
        )
        assert result.reading.status != "accepted", (
            "FAILURE (agent.py / Kshitij): agent must never accept a reading in dead zone"
        )
        assert result.needs_human is True, (
            "FAILURE (agent.py / Kshitij): agent must require human verification for dead zone"
        )


# ---------------------------------------------------------------------------
# Scenario 9: Two attempts disagree – readings_agree False →
#             agent must not accept silently
# ---------------------------------------------------------------------------

@pytest.mark.group9
class TestDisagreement:
    def test_readings_agree_false_for_hand_built_numbers(self):
        """readings_agree must return False when two values differ by more than tolerance."""
        profile = get_profile("EXAMPLE_270")
        # EXAMPLE_270 scale = 100 units; tolerance = 2% = 2 units; difference = 10 units → disagree
        agree_res = readings_agree(20.0, 30.0, profile)
        assert isinstance(agree_res, AgreementResult)
        assert agree_res.agree is False, (
            "FAILURE (calibration.py / Prasoon): readings_agree must return False "
            "when difference (10 units) exceeds profile tolerance (2% of 100 = 2 units)"
        )
        assert agree_res.difference is not None, (
            "FAILURE (calibration.py / Prasoon): AgreementResult.difference must be set"
        )

    def test_agent_does_not_silently_accept_when_readings_disagree(self, monkeypatch):
        """Agent must not silently accept when initial and retry readings disagree.

        Uses monkeypatch to force detect_dial_and_needle to return two different
        angles (45° and 135°), then verifies agent.inspect_upload does not return
        status='accepted'. No edits to agent.py.
        """
        from gaugeguard.vision import NeedleDetectionResult

        profile = get_profile("EXAMPLE_270")
        with open(os.path.join(SAMPLE_IMAGES_DIR, "industrial_gauge.png"), "rb") as f:
            image_bytes = f.read()

        calls = 0

        def mock_detect(image):
            nonlocal calls
            calls += 1
            if calls == 1:
                return NeedleDetectionResult(
                    success=True,
                    center=(100.0, 100.0),
                    radius=50.0,
                    needle_tip=(100.0, 50.0),
                    angle_deg=45.0,
                    confidence=0.55,
                    quality_flags=("low_confidence",),
                    annotated_image=image,
                    reason="attempt 1",
                )
            else:
                return NeedleDetectionResult(
                    success=True,
                    center=(100.0, 100.0),
                    radius=50.0,
                    needle_tip=(150.0, 100.0),
                    angle_deg=135.0,
                    confidence=0.75,
                    quality_flags=(),
                    annotated_image=image,
                    reason="attempt 2",
                )

        monkeypatch.setattr("gaugeguard.agent.detect_dial_and_needle", mock_detect)
        result = inspect_upload(image_bytes, profile=profile)
        assert result.reading.status != "accepted", (
            "FAILURE (agent.py / Kshitij): agent must not accept when retry attempts disagree"
        )
        assert result.needs_human is True, (
            "FAILURE (agent.py / Kshitij): agent must set needs_human=True when readings disagree"
        )
        assert "disagree" in result.reading.reason.lower(), (
            "FAILURE (agent.py / Kshitij): reason must indicate disagreement between attempts"
        )


# ---------------------------------------------------------------------------
# Scenario 10: Human verification required –
#              status needs_human, needs_human True, reason non-empty
# ---------------------------------------------------------------------------

@pytest.mark.group10
class TestHumanVerification:
    def test_needs_human_true_with_non_empty_reason_on_ambiguous_needle(self):
        """Ambiguous needle (two competing orientations) must trigger needs_human with reason.

        Image construction: a clear circular gauge with two strong needle lines at 90°
        apart (not collinear). The vision module detects ambiguous_needle quality flag,
        which agent.py maps to status='needs_human'.
        """
        profile = get_profile("EXAMPLE_270")
        size = 400
        cx, cy = size // 2, size // 2

        # Build synthetic gauge image with an unambiguous circular dial
        img = np.full((size, size, 3), 200, dtype=np.uint8)
        cv2.circle(img, (cx, cy), 140, (255, 255, 255), -1)  # gauge face
        cv2.circle(img, (cx, cy), 140, (50, 50, 50), 4)       # bezel ring
        cv2.circle(img, (cx, cy), 10, (30, 30, 30), -1)        # center pivot

        # Two strong competing needles ~90° apart (not collinear ≈180°) so that
        # vision.detect_dial_and_needle returns quality_flags=("ambiguous_needle",)
        # Needle 1: pointing up (0° in calibration convention = 12 o'clock)
        cv2.line(img, (cx, cy), (cx, cy - 110), (20, 20, 20), 6)
        # Needle 2: pointing right (90° in calibration convention = 3 o'clock)
        cv2.line(img, (cx, cy), (cx + 110, cy), (20, 20, 20), 6)

        ok, buf = cv2.imencode(".png", img)
        assert ok, "cv2.imencode failed – cannot build test image"

        result = inspect_upload(buf.tobytes(), profile=profile)

        # The valid status set (contract)
        assert result.reading.status in (
            "accepted", "needs_human", "needs_retake", "rejected_input", "error"
        ), "FAILURE (agent.py / Kshitij): unexpected status value " + str(result.reading.status)

        # Ambiguous needle → all detection passes fail with ambiguous_needle flag →
        # agent.py sets status="needs_human"
        assert result.reading.status == "needs_human", (
            "FAILURE (agent.py / Kshitij): expected needs_human when ambiguous_needle flag is "
            "present on all detection passes, got "
            + str(result.reading.status)
        )
        assert result.needs_human is True, (
            "FAILURE (agent.py / Kshitij): needs_human must be True when status is needs_human"
        )
        assert result.reading.reason and len(str(result.reading.reason)) > 0, (
            "FAILURE (agent.py / Kshitij): reason must be non-empty whenever needs_human is True"
        )
