from __future__ import annotations

from gaugeguard.models import AgentStep, InspectionResult, ImageQuality, ReadingEstimate
from gaugeguard.vision import inspect_image_quality


def inspect_upload(image_bytes: bytes) -> InspectionResult:
    """Initial orchestration scaffold; gauge detection and retries are TODOs."""
    quality: ImageQuality | None = inspect_image_quality(image_bytes)
    step = AgentStep(
        step_name="image_quality",
        method="opencv_laplacian_and_contrast",
        reason="Check basic image quality before attempting a gauge reading.",
        output_summary={"flags": list(quality.flags), "blur_score": quality.blur_score},
    )
    if quality.flags:
        reading = ReadingEstimate(
            value=None,
            unit=None,
            angle_degrees=None,
            confidence=0.0,
            status="needs_retake",
            reason="Image quality flags need review before a reading can be attempted.",
        )
    else:
        reading = ReadingEstimate(
            value=None,
            unit=None,
            angle_degrees=None,
            confidence=0.0,
            status="not_implemented",
            reason="Image quality passed; dial/needle detection is the next implementation step.",
        )
    return InspectionResult(reading=reading, quality=quality, action_log=[step])
