from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from gaugeguard.models import ImageQuality


@dataclass(frozen=True)
class NeedleDetectionResult:
    """Structured result from dial and needle detection."""

    success: bool
    center: tuple[float, float] | None
    radius: float | None
    needle_tip: tuple[float, float] | None
    angle_deg: float | None
    confidence: float
    quality_flags: tuple[str, ...] = ()
    annotated_image: np.ndarray | None = None
    reason: str = ""

    def __iter__(self):
        """Allow backwards-compatible unpacking as (center, needle_tip)."""
        return iter((self.center, self.needle_tip))

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return {
            "success": self.success,
            "center": self.center,
            "radius": self.radius,
            "needle_tip": self.needle_tip,
            "angle_deg": self.angle_deg,
            "confidence": self.confidence,
            "quality_flags": list(self.quality_flags),
            "reason": self.reason,
        }


def decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode uploaded image bytes into an OpenCV BGR image."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The image could not be decoded. Try a PNG or JPEG image.")
    return image


def inspect_image_quality(image_bytes: bytes) -> ImageQuality:
    """Return basic blur and contrast indicators; these are not calibrated confidence."""
    image = decode_image(image_bytes)
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast_score = float(gray.std())
    flags: list[str] = []
    if min(width, height) < 160:
        flags.append("low_resolution")
    if blur_score < 45.0:
        flags.append("possible_blur")
    if contrast_score < 18.0:
        flags.append("low_contrast")
    return ImageQuality(
        width=width,
        height=height,
        blur_score=blur_score,
        contrast_score=contrast_score,
        flags=tuple(flags),
    )


def preprocess_for_retry(image: np.ndarray, method: str) -> np.ndarray:
    """Apply one documented alternate preprocessing method for a future retry pass."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if method == "adaptive_threshold":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    raise ValueError(f"Unknown retry preprocessing method: {method}")


def detect_dial_and_needle(image: np.ndarray) -> NeedleDetectionResult:
    """Detect circular gauge dial, estimate center, radius, needle vector, and angle.

    Angle convention matches gaugeguard.calibration:
    0° = 12 o'clock, measured clockwise in [0, 360) degrees.

    Returns:
        NeedleDetectionResult containing detection status, coordinates, angle,
        quality flags, confidence score, and annotated visualization.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return NeedleDetectionResult(
            success=False,
            center=None,
            radius=None,
            needle_tip=None,
            angle_deg=None,
            confidence=0.0,
            quality_flags=("invalid_image",),
            annotated_image=None,
            reason="Input image is empty or invalid.",
        )

    # Determine color channels and prepare grayscale + BGR canvas
    if len(image.shape) == 2:
        gray = image
        annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        annotated = image.copy()
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        annotated = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        return NeedleDetectionResult(
            success=False,
            center=None,
            radius=None,
            needle_tip=None,
            angle_deg=None,
            confidence=0.0,
            quality_flags=("unsupported_channels",),
            annotated_image=None,
            reason="Image channels must be 1 (gray), 3 (BGR), or 4 (BGRA).",
        )

    h, w = gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 30:
        return NeedleDetectionResult(
            success=False,
            center=None,
            radius=None,
            needle_tip=None,
            angle_deg=None,
            confidence=0.0,
            quality_flags=("image_too_small",),
            annotated_image=annotated,
            reason="Image dimension is too small for circular dial detection.",
        )

    # 1. Circular dial localization via Hough Circle Transform
    blurred = cv2.medianBlur(gray, 5)
    min_r = int(min_dim * 0.15)
    max_r = int(min_dim * 0.55)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 3,
        param1=100,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )

    used_retry_clahe = False
    if circles is None:
        # Fallback using existing CLAHE helper
        enhanced = preprocess_for_retry(annotated, "clahe")
        circles = cv2.HoughCircles(
            cv2.medianBlur(enhanced, 5),
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 3,
            param1=100,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )
        used_retry_clahe = True

    if circles is None:
        return NeedleDetectionResult(
            success=False,
            center=None,
            radius=None,
            needle_tip=None,
            angle_deg=None,
            confidence=0.0,
            quality_flags=("no_dial_detected",),
            annotated_image=annotated,
            reason="No circular gauge dial was detected.",
        )

    # Select circle closest to image center
    detected_circles = circles[0]
    img_cx, img_cy = w / 2.0, h / 2.0
    best_c = min(
        detected_circles,
        key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2,
    )
    cx, cy, r = float(best_c[0]), float(best_c[1]), float(best_c[2])

    if cx < 0 or cx >= w or cy < 0 or cy >= h or r < 10:
        return NeedleDetectionResult(
            success=False,
            center=None,
            radius=None,
            needle_tip=None,
            angle_deg=None,
            confidence=0.0,
            quality_flags=("invalid_dial_geometry",),
            annotated_image=annotated,
            reason="Detected dial coordinates fall outside image bounds.",
        )

    # Draw dial boundary and pivot center
    cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)

    # 2. Extract needle candidate vectors within the dial face
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(r * 0.90), 255, -1)
    dial_gray = cv2.bitwise_and(gray, gray, mask=mask)

    edges = cv2.Canny(dial_gray, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    min_line_len = int(r * 0.22)
    max_line_gap = int(r * 0.08)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=15,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    candidates: list[tuple[float, float, tuple[float, float], float, float]] = []
    if lines is not None:
        for line in lines:
            # OpenCV 4 returns shape (N,1,4); OpenCV 5 returns (N,4).
            # line.flat[:4] is shape-agnostic and works for both.
            x1, y1, x2, y2 = (int(v) for v in line.flat[:4])
            line_len = math.hypot(x2 - x1, y2 - y1)
            if line_len < min_line_len:
                continue

            # Perpendicular distance from dial center to infinite line
            dist_to_center = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / line_len
            if dist_to_center > 0.25 * r:
                continue

            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            d_near = min(d1, d2)
            d_far = max(d1, d2)

            if d_near > 0.40 * r or d_far < 0.35 * r:
                continue

            tip_x = float(x1 if d1 > d2 else x2)
            tip_y = float(y1 if d1 > d2 else y2)
            dx = tip_x - cx
            dy = tip_y - cy
            # 0° = 12 o'clock, clockwise positive
            ang = (math.degrees(math.atan2(dx, -dy))) % 360.0
            score = line_len * (d_far / r) / (1.0 + dist_to_center)
            candidates.append((ang, score, (tip_x, tip_y), line_len, d_far))

    # Fallback: Contour analysis on thresholded dial face
    if not candidates:
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(ring_mask, (int(cx), int(cy)), int(r * 0.75), 255, -1)
        cv2.circle(ring_mask, (int(cx), int(cy)), int(r * 0.15), 0, -1)
        mean_v = cv2.mean(gray, mask=ring_mask)[0]

        if mean_v > 120:
            _, bin_img = cv2.threshold(dial_gray, int(mean_v * 0.72), 255, cv2.THRESH_BINARY_INV)
        else:
            _, bin_img = cv2.threshold(dial_gray, int(mean_v * 1.28), 255, cv2.THRESH_BINARY)

        bin_img = cv2.bitwise_and(bin_img, bin_img, mask=mask)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (r * 0.08) ** 2:
                continue
            pts = cnt.reshape(-1, 2)
            dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
            min_d = float(np.min(dists))
            max_d = float(np.max(dists))
            if min_d < 0.35 * r and max_d > 0.40 * r:
                far_idx = int(np.argmax(dists))
                tip_x = float(pts[far_idx, 0])
                tip_y = float(pts[far_idx, 1])
                dx = tip_x - cx
                dy = tip_y - cy
                ang = (math.degrees(math.atan2(dx, -dy))) % 360.0
                score = area * (max_d / r)
                candidates.append((ang, score, (tip_x, tip_y), max_d, max_d))

    if not candidates:
        cv2.putText(
            annotated,
            "Needle: Not detected",
            (max(10, int(cx - r)), max(30, int(cy - r - 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return NeedleDetectionResult(
            success=False,
            center=(round(cx, 1), round(cy, 1)),
            radius=round(r, 1),
            needle_tip=None,
            angle_deg=None,
            confidence=0.0,
            quality_flags=("no_needle_detected",),
            annotated_image=annotated,
            reason="Dial circle localized, but no needle lines or elongated structures detected.",
        )

    # 3. Cluster candidates by circular angle to evaluate dominant orientation
    clusters: list[dict[str, Any]] = []
    for cand in candidates:
        ang, score, tip, _, d_far = cand
        placed = False
        for cluster in clusters:
            diff = abs(ang - cluster["mean_angle"])
            diff = min(diff, 360.0 - diff)
            if diff <= 20.0:
                cluster["candidates"].append(cand)
                cluster["total_score"] += score
                rad = math.radians(ang)
                cluster["sin_sum"] += score * math.sin(rad)
                cluster["cos_sum"] += score * math.cos(rad)
                cluster["mean_angle"] = (math.degrees(math.atan2(cluster["sin_sum"], cluster["cos_sum"]))) % 360.0
                if d_far > cluster["best_tip_dist"]:
                    cluster["best_tip"] = tip
                    cluster["best_tip_dist"] = d_far
                placed = True
                break
        if not placed:
            rad = math.radians(ang)
            clusters.append({
                "mean_angle": ang,
                "total_score": score,
                "sin_sum": score * math.sin(rad),
                "cos_sum": score * math.cos(rad),
                "best_tip": tip,
                "best_tip_dist": d_far,
                "candidates": [cand],
            })

    clusters.sort(key=lambda c: c["total_score"], reverse=True)
    best_cluster = clusters[0]
    best_ang = round(best_cluster["mean_angle"] % 360.0, 2)
    best_tip = (
        round(float(best_cluster["best_tip"][0]), 1),
        round(float(best_cluster["best_tip"][1]), 1),
    )

    # Check for ambiguity: competing cluster > 25° away with >= 35% of top score
    for other_cluster in clusters[1:]:
        diff = abs(best_ang - other_cluster["mean_angle"])
        diff = min(diff, 360.0 - diff)
        # Exclude collinear opposite cluster (~180° away) corresponding to needle counterweight/tail
        if abs(diff - 180.0) <= 25.0:
            continue
        if diff > 25.0 and other_cluster["total_score"] >= 0.35 * best_cluster["total_score"]:
            cv2.putText(
                annotated,
                "Needle: Ambiguous",
                (max(10, int(cx - r)), max(30, int(cy - r - 10))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return NeedleDetectionResult(
                success=False,
                center=(round(cx, 1), round(cy, 1)),
                radius=round(r, 1),
                needle_tip=None,
                angle_deg=None,
                confidence=0.20,
                quality_flags=("ambiguous_needle",),
                annotated_image=annotated,
                reason="Multiple conflicting needle orientations detected; cannot reliably determine needle angle.",
            )

    # 4. Success: compute confidence and render annotations
    tip_reach = min(1.0, best_cluster["best_tip_dist"] / r)
    raw_confidence = tip_reach * 0.85 + (0.10 if len(best_cluster["candidates"]) > 1 else 0.0)
    confidence = round(min(0.95, max(0.40, raw_confidence)), 2)

    quality_flags: list[str] = []
    if confidence < 0.60:
        quality_flags.append("low_confidence")
    if used_retry_clahe:
        quality_flags.append("clahe_enhanced")

    # Render needle line, tip, and angle text
    cv2.line(
        annotated,
        (int(cx), int(cy)),
        (int(best_tip[0]), int(best_tip[1])),
        (0, 140, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.circle(annotated, (int(best_tip[0]), int(best_tip[1])), 5, (0, 255, 255), -1)
    cv2.putText(
        annotated,
        f"Angle: {best_ang:.1f} deg",
        (max(10, int(cx - r)), max(30, int(cy - r - 10))),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return NeedleDetectionResult(
        success=True,
        center=(round(cx, 1), round(cy, 1)),
        radius=round(r, 1),
        needle_tip=best_tip,
        angle_deg=best_ang,
        confidence=confidence,
        quality_flags=tuple(quality_flags),
        annotated_image=annotated,
        reason=f"Needle detected at angle {best_ang:.1f}° with center ({cx:.1f}, {cy:.1f}) and radius {r:.1f}px.",
    )

