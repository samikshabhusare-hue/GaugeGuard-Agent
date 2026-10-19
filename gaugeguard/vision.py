from __future__ import annotations

import cv2
import numpy as np

from gaugeguard.models import ImageQuality


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


def detect_dial_and_needle(image: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    """TODO: implement and evaluate for the single supported gauge style."""
    raise NotImplementedError("Dial and needle detection has not been implemented yet.")
