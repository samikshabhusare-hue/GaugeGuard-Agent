"""Tests for gaugeguard.vision.detect_dial_and_needle."""

from __future__ import annotations

import math
from pathlib import Path
import cv2
import numpy as np
import pytest

from gaugeguard.vision import NeedleDetectionResult, detect_dial_and_needle


def _create_synthetic_gauge(
    angle_deg: float,
    dial_radius: int = 120,
    size: int = 320,
) -> tuple[np.ndarray, tuple[int, int], int]:
    """Helper to generate a clean synthetic gauge image with known needle orientation."""
    img = np.full((size, size, 3), 240, dtype=np.uint8)
    cx, cy = size // 2, size // 2

    # Outer casing & dial face
    outer_r = dial_radius + 12
    cv2.circle(img, (cx, cy), outer_r, (50, 50, 50), -1)
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

    return img, (cx, cy), outer_r



def test_invalid_and_empty_images():
    """Missing or invalid image arrays must return safe failure without raising."""
    res_none = detect_dial_and_needle(None)
    assert not res_none.success
    assert res_none.angle_deg is None
    assert "invalid_image" in res_none.quality_flags

    res_empty = detect_dial_and_needle(np.array([]))
    assert not res_empty.success
    assert res_empty.angle_deg is None
    assert "invalid_image" in res_empty.quality_flags

    res_small = detect_dial_and_needle(np.zeros((20, 20), dtype=np.uint8))
    assert not res_small.success
    assert "image_too_small" in res_small.quality_flags


def test_no_dial_detected_on_blank():
    """An image with no circular dial must return safe failure and quality flag."""
    blank = np.full((300, 300, 3), 220, dtype=np.uint8)
    res = detect_dial_and_needle(blank)
    assert not res.success
    assert res.center is None
    assert res.radius is None
    assert res.angle_deg is None
    assert "no_dial_detected" in res.quality_flags
    assert res.annotated_image is not None


def test_no_needle_detected_on_empty_dial():
    """A dial with no needle line must report located dial but no needle."""
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (50, 50, 50), -1)
    cv2.circle(img, (150, 150), 90, (250, 250, 250), -1)
    cv2.circle(img, (150, 150), 8, (45, 45, 45), -1)

    res = detect_dial_and_needle(img)
    assert not res.success
    assert res.center is not None
    assert res.radius is not None
    assert res.angle_deg is None
    assert "no_needle_detected" in res.quality_flags


def test_ambiguous_needles():
    """Conflicting candidate needles pointing in different directions must be flagged."""
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (50, 50, 50), -1)
    cv2.circle(img, (150, 150), 90, (250, 250, 250), -1)
    # Two competing lines at 0° and 90°
    cv2.line(img, (150, 150), (150, 80), (20, 20, 20), 4)
    cv2.line(img, (150, 150), (220, 150), (20, 20, 20), 4)

    res = detect_dial_and_needle(img)
    assert not res.success
    assert res.angle_deg is None
    assert "ambiguous_needle" in res.quality_flags


@pytest.mark.parametrize("target_angle", [0.0, 90.0, 180.0, 270.0])
def test_synthetic_gauge_cardinal_angles(target_angle: float):
    """Accurately detect cardinal needle angles on synthetic gauge images."""
    img, (expected_cx, expected_cy), expected_r = _create_synthetic_gauge(target_angle)
    res = detect_dial_and_needle(img)

    assert res.success, f"Failed for angle {target_angle}: {res.reason}"
    assert res.angle_deg is not None
    # Verify angular distance on circle is within 3.0 degrees
    diff = abs(res.angle_deg - target_angle)
    diff = min(diff, 360.0 - diff)
    assert diff <= 3.0, f"Detected {res.angle_deg}° but expected {target_angle}° (diff {diff}°)"

    # Verify dial center and radius estimates
    assert abs(res.center[0] - expected_cx) <= 3.0
    assert abs(res.center[1] - expected_cy) <= 3.0
    assert abs(res.radius - expected_r) <= 6.0

    # Verify annotated image
    assert isinstance(res.annotated_image, np.ndarray)
    assert res.annotated_image.shape == img.shape


def test_structured_return_format():
    """Verify return format adheres to NeedleDetectionResult interface."""
    img, (cx, cy), r = _create_synthetic_gauge(45.0)
    res = detect_dial_and_needle(img)

    assert isinstance(res, NeedleDetectionResult)
    assert isinstance(res.success, bool)
    assert isinstance(res.center, tuple)
    assert isinstance(res.radius, float)
    assert isinstance(res.needle_tip, tuple)
    assert isinstance(res.angle_deg, float)
    assert isinstance(res.confidence, float)
    assert isinstance(res.quality_flags, tuple)
    assert isinstance(res.reason, str)

    # Test dictionary export
    d = res.to_dict()
    assert "success" in d and "center" in d and "angle_deg" in d and "confidence" in d

    # Test backwards-compatible 2-tuple unpacking
    unpacked_center, unpacked_tip = res
    assert unpacked_center == res.center
    assert unpacked_tip == res.needle_tip


def test_houghlinesP_coordinate_extraction_opencv5_compat():
    """Regression test: cv2.HoughLinesP coordinate extraction must work under both
    OpenCV 4 (line shape (1,4)) and OpenCV 5 (line shape (4,)).

    OpenCV 5 changed HoughLinesP output from (N,1,4) to (N,4).  The previous
    ``line[0]`` unpack raised ``TypeError: cannot unpack non-iterable
    numpy.int32 object`` on OpenCV 5 because ``line[0]`` became a scalar.
    ``line.flat[:4]`` is shape-agnostic and must succeed on any version.
    """
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(img, (20, 100), (180, 100), 255, 3)
    edges = cv2.Canny(img, 50, 150)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=15,
        minLineLength=20, maxLineGap=5,
    )
    assert lines is not None, "HoughLinesP returned None on a clear synthetic line"

    # Shape must be 2-D (OpenCV 5) or 3-D (OpenCV 4) — both are acceptable.
    assert lines.ndim in (2, 3), f"Unexpected HoughLinesP output ndim: {lines.ndim}"

    for line in lines:
        # This is the exact extraction used in vision.py — must not raise.
        coords = [int(v) for v in line.flat[:4]]
        assert len(coords) == 4, "Expected exactly 4 coordinates per line"
        x1, y1, x2, y2 = coords
        # All coordinates must be valid pixel positions within the test image.
        for coord in (x1, x2):
            assert 0 <= coord < 200, f"x coordinate {coord} out of image bounds"
        for coord in (y1, y2):
            assert 0 <= coord < 200, f"y coordinate {coord} out of image bounds"


def test_needle_with_prominent_counterweight_tail_accepted():
    """A needle with an extended counterweight tail (~180° opposite) must be correctly detected."""
    size = 320
    cx, cy = size // 2, size // 2
    dial_radius = 120
    outer_r = dial_radius + 12

    img = np.full((size, size, 3), 240, dtype=np.uint8)
    cv2.circle(img, (cx, cy), outer_r, (50, 50, 50), -1)
    cv2.circle(img, (cx, cy), dial_radius, (250, 250, 250), -1)
    cv2.circle(img, (cx, cy), dial_radius, (80, 80, 80), 3)

    for tick_angle in range(0, 360, 30):
        rad = math.radians(tick_angle)
        dx, dy = math.sin(rad), -math.cos(rad)
        p1 = (int(cx + dx * (dial_radius - 14)), int(cy + dy * (dial_radius - 14)))
        p2 = (int(cx + dx * (dial_radius - 4)), int(cy + dy * (dial_radius - 4)))
        cv2.line(img, p1, p2, (100, 100, 100), 2)

    # Needle pointer at 45.0°
    target_angle = 45.0
    rad = math.radians(target_angle)
    dx, dy = math.sin(rad), -math.cos(rad)
    tip_len = dial_radius * 0.75
    tip_pt = (int(cx + dx * tip_len), int(cy + dy * tip_len))

    # Prominent counterweight tail in the opposite direction (225°), length 0.35 * dial_radius
    tail_len = dial_radius * 0.35
    tail_pt = (int(cx - dx * tail_len), int(cy - dy * tail_len))

    # Draw needle shaft and prominent tail
    cv2.line(img, (cx, cy), tip_pt, (25, 25, 25), 4)
    cv2.line(img, (cx, cy), tail_pt, (20, 20, 20), 7)
    cv2.circle(img, (cx, cy), 10, (40, 40, 40), -1)

    res = detect_dial_and_needle(img)
    assert res.success, f"Detection failed: {res.reason}"
    assert "ambiguous_needle" not in res.quality_flags
    diff = abs(res.angle_deg - target_angle)
    diff = min(diff, 360.0 - diff)
    assert diff <= 3.0, f"Detected angle {res.angle_deg}° differs from target {target_angle}°"


def test_non_collinear_competing_needles_remain_ambiguous():
    """Genuinely non-collinear competing needle lines (e.g. 45° and 135°) must remain flagged ambiguous."""
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (50, 50, 50), -1)
    cv2.circle(img, (150, 150), 90, (250, 250, 250), -1)

    # Competing lines at 45° and 135° (perpendicular / non-collinear)
    for ang in [45.0, 135.0]:
        rad = math.radians(ang)
        p2 = (int(150 + math.sin(rad) * 70), int(150 - math.cos(rad) * 70))
        cv2.line(img, (150, 150), p2, (20, 20, 20), 4)

    res = detect_dial_and_needle(img)
    assert not res.success
    assert res.angle_deg is None
    assert "ambiguous_needle" in res.quality_flags
    assert res.confidence == 0.20


def test_uploaded_industrial_gauge_fixture():
    """Real industrial pressure gauge fixture with counterweight tail must be correctly detected."""
    fixture_path = Path(__file__).parent / "fixtures" / "industrial_gauge.png"
    if not fixture_path.exists():
        pytest.skip(f"Fixture image not found at {fixture_path}")

    img = cv2.imread(str(fixture_path))
    assert img is not None, "Failed to decode fixture image"

    res = detect_dial_and_needle(img)
    assert res.success, f"Detection failed: {res.reason}"
    assert "ambiguous_needle" not in res.quality_flags
    assert abs(res.angle_deg - 45.2) <= 2.0, f"Detected angle was {res.angle_deg}°"
    assert res.confidence >= 0.70
    assert res.needle_tip is not None
