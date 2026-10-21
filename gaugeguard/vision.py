from __future__ import annotations

import math

import cv2
import numpy as np

from gaugeguard.models import ImageQuality, VisionResult


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
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 1:
        gray = image[:, :, 0]
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if method == "adaptive_threshold":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
        )
    raise ValueError(f"Unknown retry preprocessing method: {method}")


def detect_dial_and_needle(image: np.ndarray) -> VisionResult:
    """Detect the gauge dial and needle, calculating needle angle clockwise from +X axis (0-360 deg).

    Accepts BGR or grayscale numpy arrays.
    Returns a VisionResult containing geometry, angle, confidence, and a BGR annotated image.
    """
    if image is None:
        return VisionResult(success=False, error_message="Image input is None.")

    # Reuse decode_image if bytes are passed
    if isinstance(image, bytes):
        try:
            image = decode_image(image)
        except Exception as exc:
            return VisionResult(success=False, error_message=f"Failed to decode image bytes: {exc}")

    if not isinstance(image, np.ndarray) or image.size == 0:
        return VisionResult(success=False, error_message="Image must be a non-empty NumPy array.")

    if image.dtype != np.uint8:
        return VisionResult(
            success=False,
            error_message=f"Unsupported image dtype: {image.dtype}. Expected uint8.",
        )

    # Normalize input into gray and BGR annotated copy
    if image.ndim == 2:
        gray = image.copy()
        annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        bgr = annotated.copy()
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 3:
            bgr = image.copy()
            annotated = image.copy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif channels == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            annotated = bgr.copy()
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        elif channels == 1:
            gray = image[:, :, 0].copy()
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            bgr = annotated.copy()
        else:
            return VisionResult(
                success=False,
                error_message=f"Unsupported number of image channels: {channels}.",
            )
    else:
        return VisionResult(
            success=False,
            error_message=f"Unsupported image shape: {image.shape}.",
        )

    height, width = gray.shape[:2]
    min_dim = min(height, width)
    if min_dim < 30:
        cv2.putText(
            annotated,
            "Image too small",
            (5, min(height - 5, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
        return VisionResult(
            success=False,
            annotated_image=annotated,
            error_message=f"Image resolution too low for gauge detection ({width}x{height} px).",
        )

    try:
        # 1. Dial circle localization
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        min_radius = max(10, int(min_dim * 0.15))
        max_radius = int(min_dim * 0.55)
        img_cx, img_cy = width / 2.0, height / 2.0
        diag = math.hypot(width, height)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min_dim // 3,
            param1=100,
            param2=45,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dim // 3,
                param1=80,
                param2=28,
                minRadius=min_radius,
                maxRadius=max_radius,
            )

        best_circle: tuple[float, float, float] | None = None
        best_circle_score = -1.0

        if circles is not None:
            circle_list = circles[0] if circles.ndim == 3 else circles
            for circ in circle_list:
                c_x, c_y, rad = float(circ[0]), float(circ[1]), float(circ[2])
                if 0 <= c_x < width and 0 <= c_y < height:
                    dist = math.hypot(c_x - img_cx, c_y - img_cy)
                    score = (rad / max_radius) * (1.0 - 0.4 * (dist / diag))
                    if score > best_circle_score:
                        best_circle_score = score
                        best_circle = (c_x, c_y, rad)

        # Contour fallback for dial if HoughCircles did not find a circle
        if best_circle is None:
            edges = cv2.Canny(blurred, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                peri = cv2.arcLength(cnt, True)
                if peri > 0:
                    circ_metric = 4.0 * math.pi * area / (peri * peri)
                    (c_x, c_y), rad = cv2.minEnclosingCircle(cnt)
                    if circ_metric > 0.4 and min_radius <= rad <= max_radius:
                        dist = math.hypot(c_x - img_cx, c_y - img_cy)
                        score = (rad / max_radius) * circ_metric * (1.0 - 0.4 * (dist / diag))
                        if score > best_circle_score:
                            best_circle_score = score
                            best_circle = (float(c_x), float(c_y), float(rad))

        if best_circle is None:
            cv2.putText(
                annotated,
                "Dial not detected",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return VisionResult(
                success=False,
                annotated_image=annotated,
                error_message="Could not localize dial boundary.",
            )

        cx, cy, r = best_circle

        # 2. Needle localization within dial ROI
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r * 0.88), 255, -1)
        dial_roi = cv2.bitwise_and(gray, gray, mask=mask)
        dial_smooth = cv2.medianBlur(dial_roi, 5)

        def extract_candidate_lines(edge_map: np.ndarray) -> list[tuple[float, float, float, float, float]]:
            # Exclude edges near the outer ring to prevent tick arcs from interfering
            cv2.circle(edge_map, (int(cx), int(cy)), int(r * 0.88), 0, max(2, int(r * 0.12)))
            lines = cv2.HoughLinesP(
                edge_map,
                rho=1,
                theta=np.pi / 180,
                threshold=15,
                minLineLength=max(8, int(r * 0.22)),
                maxLineGap=max(3, int(r * 0.08)),
            )
            candidates = []
            if lines is not None:
                for l in lines:
                    x1, y1, x2, y2 = [float(v) for v in l.reshape(-1)[:4]]
                    line_len = math.hypot(x2 - x1, y2 - y1)
                    if line_len < 1e-3:
                        continue
                    # Line distance to center
                    d_center = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / line_len
                    # Endpoints distance to center
                    d1 = math.hypot(x1 - cx, y1 - cy)
                    d2 = math.hypot(x2 - cx, y2 - cy)
                    # Needle line must pass close to center and reach outward
                    if d_center < 0.22 * r and min(d1, d2) < 0.55 * r and max(d1, d2) > 0.30 * r:
                        candidates.append((x1, y1, x2, y2, line_len))
            return candidates

        edges = cv2.Canny(dial_smooth, 30, 100)
        candidates = extract_candidate_lines(edges)

        # Fallback with CLAHE retry preprocessing if edge pass found no radial needle lines
        if not candidates:
            try:
                clahe_gray = preprocess_for_retry(bgr, "clahe")
                clahe_masked = cv2.bitwise_and(clahe_gray, clahe_gray, mask=mask)
                clahe_edges = cv2.Canny(cv2.medianBlur(clahe_masked, 5), 30, 100)
                candidates = extract_candidate_lines(clahe_edges)
            except Exception:
                pass

        # Fallback to Otsu threshold & contour if lines still empty
        if not candidates:
            mean_val = float(cv2.mean(gray, mask=mask)[0])
            thresh_type = cv2.THRESH_BINARY_INV if mean_val > 120 else cv2.THRESH_BINARY
            _, thresh = cv2.threshold(dial_smooth, 0, 255, thresh_type + cv2.THRESH_OTSU)
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
            # Suppress center hub to decouple shaft from pivot disk
            cv2.circle(thresh, (int(cx), int(cy)), max(3, int(r * 0.15)), 0, -1)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in cnts:
                if len(cnt) >= 5 and cv2.contourArea(cnt) > 20:
                    rect = cv2.minAreaRect(cnt)
                    rw, rh = rect[1]
                    aspect = max(rw, rh) / (min(rw, rh) + 1e-3)
                    if aspect > 2.0:
                        pts = cnt.reshape(-1, 2)
                        dists = [math.hypot(px - cx, py - cy) for px, py in pts]
                        max_idx = int(np.argmax(dists))
                        min_idx = int(np.argmin(dists))
                        x_far, y_far = float(pts[max_idx][0]), float(pts[max_idx][1])
                        x_near, y_near = float(pts[min_idx][0]), float(pts[min_idx][1])
                        if dists[max_idx] > 0.30 * r and dists[min_idx] < 0.55 * r:
                            c_len = math.hypot(x_far - x_near, y_far - y_near)
                            candidates.append((x_near, y_near, x_far, y_far, c_len))

        if not candidates:
            # Draw dial overlay for partial failure feedback
            cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 165, 255), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                "Needle not detected",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return VisionResult(
                success=False,
                center=(round(cx, 2), round(cy, 2)),
                radius=round(r, 2),
                annotated_image=annotated,
                error_message="Could not detect needle line on dial face.",
            )

        # 3. Angle calculation (clockwise from +X axis) & Tip Disambiguation
        total_vx, total_vy = 0.0, 0.0
        max_reach = 0.0

        for x1, y1, x2, y2, length in candidates:
            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            # The tip is further from center than the pivot/counterweight
            tx, ty = (x2, y2) if d2 > d1 else (x1, y1)
            vx = tx - cx
            vy = ty - cy
            dist = math.hypot(vx, vy)
            if dist > 1e-3:
                weight = length
                total_vx += weight * (vx / dist)
                total_vy += weight * (vy / dist)
                if dist > max_reach:
                    max_reach = dist

        vector_magnitude = math.hypot(total_vx, total_vy)
        if vector_magnitude < 1e-3:
            cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 165, 255), 2)
            cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                "Ambiguous needle",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            return VisionResult(
                success=False,
                center=(round(cx, 2), round(cy, 2)),
                radius=round(r, 2),
                annotated_image=annotated,
                error_message="Could not resolve unique needle direction; candidate vectors canceled out.",
            )

        angle_rad = math.atan2(total_vy, total_vx)
        angle_degrees = float((math.degrees(angle_rad)) % 360.0)

        # Projected tip along the fused needle vector
        tip_dist = min(max_reach, r * 0.85)
        tip_x = cx + tip_dist * math.cos(angle_rad)
        tip_y = cy + tip_dist * math.sin(angle_rad)

        # 4. Confidence calculation
        reach_ratio = tip_dist / r
        confidence = 0.65
        if reach_ratio >= 0.45:
            confidence += 0.15
        if len(candidates) >= 2:
            confidence += 0.10
        if best_circle_score > 0.5:
            confidence += 0.05
        confidence = float(min(0.95, max(0.10, confidence)))

        # 5. Annotation overlay
        cv2.circle(annotated, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
        cv2.circle(annotated, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        cv2.line(annotated, (int(cx), int(cy)), (int(tip_x), int(tip_y)), (255, 0, 0), 2)
        cv2.circle(annotated, (int(tip_x), int(tip_y)), 5, (0, 165, 255), -1)
        label = f"Angle: {angle_degrees:.1f} deg | Conf: {confidence:.2f}"
        cv2.putText(
            annotated,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        return VisionResult(
            success=True,
            angle_degrees=round(angle_degrees, 2),
            confidence=round(confidence, 2),
            center=(round(cx, 2), round(cy, 2)),
            radius=round(r, 2),
            needle_tip=(round(tip_x, 2), round(tip_y, 2)),
            annotated_image=annotated,
            error_message=None,
        )
    except Exception as exc:
        return VisionResult(
            success=False,
            annotated_image=annotated if "annotated" in locals() and annotated is not None else None,
            error_message=f"Vision processing failed: {exc}",
        )
