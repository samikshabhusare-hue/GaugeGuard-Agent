
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
from gaugeguard.vision import inspect_image_quality
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

    with open(SAMPLE_LABELS_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("filename"):
                rows.append(row)

    return rows


SAMPLE_LABELS = _load_sample_labels()


# ---------------------------------------------------------------------------
# Scenario 1: Clear image – numeric reading exists and error is within tolerance.
# An unverified demo calibration profile must still require human review.
# ---------------------------------------------------------------------------

@pytest.mark.group1
class TestClearImage:
    def test_clear_image_reading_exists_and_near_ground_truth(self):
        label = next(
            (row for row in SAMPLE_LABELS if row.get("condition") == "clear"),
            None,
        )

        if label is None:
            pytest.skip(
                "MISSING DATA: clear image label in sample_images/labels.csv"
            )

        fixture_path = os.path.join(SAMPLE_IMAGES_DIR, label["filename"])

        if not os.path.exists(fixture_path):
            pytest.skip(
                f"MISSING DATA: clear sample image {fixture_path}"
            )

        with open(fixture_path, "rb") as f:
            image_bytes = f.read()

        profile_name = label.get("gauge_profile") or "EXAMPLE_270"
        profile = get_profile(profile_name)
        result = inspect_upload(image_bytes, profile=profile)

        assert result.reading.value is not None, (
            "FAILURE (agent.py / Kshitij): expected numeric reading for clear image"
        )

        true_val = float(label["true_value"])
        uncertainty = float(label["label_uncertainty"])

        assert abs(result.reading.value - true_val) <= uncertainty, (
            f"FAILURE: value {result.reading.value} vs truth {true_val} "
            f"(tolerance ±{uncertainty})"
        )

        calibration_steps = [
            step
            for step in result.action_log
            if step.step_name == "calibration_mapping"
        ]

        assert calibration_steps, (
            "FAILURE (agent.py / Kshitij): calibration decision must be logged"
        )

        flags = calibration_steps[-1].output_summary.get("flags", [])

        if profile_name.upper().startswith(("EXAMPLE", "DEMO", "SAMPLE", "MOCK", "TEST_")):
            assert result.reading.status == "needs_human", (
                "An unverified demo profile must require human review; "
                f"got {result.reading.status}"
            )
            assert result.needs_human is True, (
                "An unverified demo profile must set needs_human=True"
            )
            assert "unverified_demo_profile" in flags, (
                "The calibration log must identify the unverified demo profile"
            )
        else:
            # For a non-demo profile, this test does not assume that the profile
            # is actually verified. The agent's own calibration flags determine
            # whether automatic acceptance is safe.
            if "unverified_demo_profile" in flags:
                assert result.reading.status == "needs_human"
                assert result.needs_human is True


# ---------------------------------------------------------------------------
# Scenario 2: Blurry image
# ---------------------------------------------------------------------------

@pytest.mark.group2
class TestBlurryImage:
    def test_blurry_image_requires_review(self):
        """A heavily blurred synthetic sample must not be auto-accepted."""
        source_path = os.path.join(
            SAMPLE_IMAGES_DIR, "industrial_gauge.png"
        )
        source = cv2.imread(source_path)

        if source is None:
            pytest.skip(
                "MISSING DATA: sample_images/industrial_gauge.png"
            )

        blurred = cv2.GaussianBlur(source, (51, 51), 0)
        ok, encoded = cv2.imencode(".png", blurred)
        assert ok, "Could not encode synthetic blurred image"

        image_bytes = encoded.tobytes()
        quality = inspect_image_quality(image_bytes)
        result = inspect_upload(
            image_bytes,
            profile=get_profile("EXAMPLE_270"),
        )

        assert "possible_blur" in quality.flags, (
            "FAILURE (vision.py): heavily blurred image must have possible_blur flag"
        )
        assert result.reading.status != "accepted"
        assert (
            result.needs_human
            or result.reading.status == "needs_retake"
        )


# ---------------------------------------------------------------------------
# Scenario 3: Tilted image
# ---------------------------------------------------------------------------

@pytest.mark.group3
class TestTiltedImage:
    def test_tilted_image_requires_review(self):
        """A rotated gauge must remain untrusted under the demo profile."""
        source_path = os.path.join(
            SAMPLE_IMAGES_DIR, "industrial_gauge.png"
        )
        source = cv2.imread(source_path)

        if source is None:
            pytest.skip(
                "MISSING DATA: sample_images/industrial_gauge.png"
            )

        height, width = source.shape[:2]
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2), 18, 1.0
        )
        tilted = cv2.warpAffine(
            source,
            matrix,
            (width, height),
            borderValue=(30, 30, 30),
        )

        ok, encoded = cv2.imencode(".png", tilted)
        assert ok, "Could not encode synthetic tilted image"

        result = inspect_upload(
            encoded.tobytes(),
            profile=get_profile("EXAMPLE_270"),
        )

        assert result.reading.status != "accepted"
        assert (
            result.needs_human
            or result.reading.status in ("needs_retake", "rejected_input")
        )


# ---------------------------------------------------------------------------
# Scenario 4: Poorly illuminated image
# ---------------------------------------------------------------------------

@pytest.mark.group4
class TestDarkImage:
    def test_dark_image_is_flagged_and_not_auto_accepted(self):
        """A dark synthetic sample must be flagged and not auto-accepted."""
        source_path = os.path.join(
            SAMPLE_IMAGES_DIR, "industrial_gauge.png"
        )
        source = cv2.imread(source_path)

        if source is None:
            pytest.skip(
                "MISSING DATA: sample_images/industrial_gauge.png"
            )

        dark = cv2.convertScaleAbs(source, alpha=0.08, beta=0)
        ok, encoded = cv2.imencode(".png", dark)
        assert ok, "Could not encode synthetic dark image"

        image_bytes = encoded.tobytes()
        quality = inspect_image_quality(image_bytes)
        result = inspect_upload(
            image_bytes,
            profile=get_profile("EXAMPLE_270"),
        )

        assert (
            "low_contrast" in quality.flags
            or "possible_blur" in quality.flags
        ), "FAILURE (vision.py): dark image was not flagged as poor quality"

        assert result.reading.status != "accepted"


# ---------------------------------------------------------------------------
# Scenario 5: Partially obstructed image
# ---------------------------------------------------------------------------

@pytest.mark.group5
class TestObstructedImage:
    def test_obstructed_image_is_not_auto_accepted(self):
        """A synthetic obstruction must not be treated as a verified reading."""
        source_path = os.path.join(
            SAMPLE_IMAGES_DIR, "industrial_gauge.png"
        )
        source = cv2.imread(source_path)

        if source is None:
            pytest.skip(
                "MISSING DATA: sample_images/industrial_gauge.png"
            )

        obstructed = source.copy()
        height, width = obstructed.shape[:2]

        cv2.rectangle(
            obstructed,
            (width // 2 - 45, height // 2 - 45),
            (width // 2 + 45, height // 2 + 45),
            (0, 0, 0),
            -1,
        )

        ok, encoded = cv2.imencode(".png", obstructed)
        assert ok, "Could not encode synthetic obstructed image"

        result = inspect_upload(
            encoded.tobytes(),
            profile=get_profile("EXAMPLE_270"),
        )

        assert result.reading.status != "accepted"
        assert (
            result.needs_human
            or result.reading.status in ("needs_retake", "rejected_input")
        )


# ---------------------------------------------------------------------------
# Scenario 6: Corrupted file and unsupported file
# ---------------------------------------------------------------------------

@pytest.mark.group6
class TestCorruptedAndUnsupported:
    def test_corrupted_bytes_return_rejected_without_crash(self, tmp_path):
        """Undecodable bytes must return rejected_input without crashing."""
        garbage_path = tmp_path / "garbage.png"
        garbage_path.write_bytes(b"\x00\x01\xff\xfe" * 64)

        bytes_data = garbage_path.read_bytes()

        try:
            result = inspect_upload(bytes_data)
        except Exception as exc:
            pytest.fail(
                "FAILURE (agent.py / Kshitij): corrupted bytes caused a crash "
                f"instead of returning rejected_input. Exception: {exc}"
            )

        assert result.reading.status == "rejected_input", (
            "FAILURE (agent.py / Kshitij): expected rejected_input for corrupted "
            f"file, got {result.reading.status}"
        )

    def test_unsupported_text_as_png_return_rejected_without_crash(self, tmp_path):
        """Text content renamed to PNG must return rejected_input without crashing."""
        fake_path = tmp_path / "fake_image.png"
        fake_path.write_text(
            "This is not an image. Just text content.",
            encoding="utf-8",
        )

        bytes_data = fake_path.read_bytes()

        try:
            result = inspect_upload(bytes_data)
        except Exception as exc:
            pytest.fail(
                "FAILURE (agent.py / Kshitij): unsupported file caused a crash "
                f"instead of returning rejected_input. Exception: {exc}"
            )

        assert result.reading.status == "rejected_input", (
            "FAILURE (agent.py / Kshitij): expected rejected_input for unsupported "
            f"file, got {result.reading.status}"
        )


# ---------------------------------------------------------------------------
# Scenario 7: Uncertain reading – value_low/value_high present, range > 0
# ---------------------------------------------------------------------------

@pytest.mark.group7
class TestUncertainReading:
    def test_uncertainty_range_is_positive(self):
        """Non-zero angular uncertainty must produce a positive value range."""
        profile = get_profile("EXAMPLE_270")
        result = angle_to_value(
            0.0,
            profile,
            angle_uncertainty_deg=10.0,
        )

        assert isinstance(result, CalibrationResult)
        assert result.value_low is not None, (
            "FAILURE (calibration.py / Prasoon): value_low missing "
            "when uncertainty is provided"
        )
        assert result.value_high is not None, (
            "FAILURE (calibration.py / Prasoon): value_high missing "
            "when uncertainty is provided"
        )
        assert result.value_high - result.value_low > 0, (
            "FAILURE (calibration.py / Prasoon): uncertainty range must be "
            "positive for non-zero angle_uncertainty_deg"
        )


# ---------------------------------------------------------------------------
# Scenario 8: Reading outside calibration bounds – dead zone
# ---------------------------------------------------------------------------

@pytest.mark.group8
class TestDeadZone:
    def test_dead_zone_angle_returns_none_and_in_bounds_false(self):
        """Dead-zone angle must return no value and an out-of-scale flag."""
        profile = get_profile("EXAMPLE_270")
        result = angle_to_value(180.0, profile)

        assert result.value is None, (
            "FAILURE (calibration.py / Prasoon): dead zone must return value=None"
        )
        assert result.in_bounds is False, (
            "FAILURE (calibration.py / Prasoon): dead zone must set in_bounds=False"
        )
        assert "angle_outside_scale" in result.flags, (
            "FAILURE (calibration.py / Prasoon): missing angle_outside_scale flag"
        )

    def test_agent_never_accepts_dead_zone(self):
        """Calibration must mark a dead-zone angle as out of bounds."""
        profile = get_profile("EXAMPLE_270")
        result = angle_to_value(180.0, profile)

        assert result.value is None, (
            "FAILURE (calibration.py / Prasoon): dead-zone value must be None"
        )
        assert result.in_bounds is False, (
            "FAILURE (calibration.py / Prasoon): dead-zone result must be out of bounds"
        )

        # This confirms the calibration-side invariant only.
        # An integration test using a real image should verify the agent response.
        assert "angle_outside_scale" in result.flags


# ---------------------------------------------------------------------------
# Scenario 9: Two attempts disagree
# ---------------------------------------------------------------------------

@pytest.mark.group9
class TestDisagreement:
    def test_readings_agree_false_for_hand_built_numbers(self):
        """Values differing beyond tolerance must not agree."""
        profile = get_profile("EXAMPLE_270")
        agree_result = readings_agree(20.0, 30.0, profile)

        assert isinstance(agree_result, AgreementResult)
        assert agree_result.agree is False, (
            "FAILURE (calibration.py / Prasoon): readings must disagree when "
            "their difference exceeds profile tolerance"
        )
        assert agree_result.difference is not None, (
            "FAILURE (calibration.py / Prasoon): difference must be set"
        )

    def test_agent_does_not_silently_accept_when_readings_disagree(self):
        """Document the disagreement contract without pretending to test integration."""
        profile = get_profile("EXAMPLE_270")
        agree_result = readings_agree(20.0, 30.0, profile)

        assert agree_result.agree is False, (
            "FAILURE (calibration.py / Prasoon): disagreement invariant failed"
        )

        # This tests calibration behavior, not the agent's integration path.
        # A true integration test needs an image that produces two valid passes.


# ---------------------------------------------------------------------------
# Scenario 10: Human verification required for ambiguous needle
# ---------------------------------------------------------------------------

@pytest.mark.group10
class TestHumanVerification:
    def test_needs_human_true_with_non_empty_reason_on_ambiguous_needle(self):
        """Two competing needle orientations must trigger human review."""
        profile = get_profile("EXAMPLE_270")
        size = 400
        center_x = size // 2
        center_y = size // 2

        img = np.full((size, size, 3), 200, dtype=np.uint8)
        cv2.circle(img, (center_x, center_y), 140, (255, 255, 255), -1)
        cv2.circle(img, (center_x, center_y), 140, (50, 50, 50), 4)
        cv2.circle(img, (center_x, center_y), 10, (30, 30, 30), -1)

        # Two competing needle lines: up and right.
        cv2.line(
            img,
            (center_x, center_y),
            (center_x, center_y - 110),
            (20, 20, 20),
            6,
        )
        cv2.line(
            img,
            (center_x, center_y),
            (center_x + 110, center_y),
            (20, 20, 20),
            6,
        )

        ok, encoded = cv2.imencode(".png", img)
        assert ok, "Could not encode synthetic ambiguous gauge image"

        result = inspect_upload(encoded.tobytes(), profile=profile)

        assert result.reading.status in (
            "accepted",
            "needs_human",
            "needs_retake",
            "rejected_input",
            "error",
        ), (
            "FAILURE (agent.py / Kshitij): unexpected status value "
            f"{result.reading.status}"
        )

        assert result.reading.status == "needs_human", (
            "FAILURE (agent.py / Kshitij): expected needs_human for ambiguous "
            f"needle, got {result.reading.status}"
        )
        assert result.needs_human is True, (
            "FAILURE (agent.py / Kshitij): needs_human must be True"
        )
        assert result.reading.reason and str(result.reading.reason).strip(), (
            "FAILURE (agent.py / Kshitij): human review must have a non-empty reason"
        )
