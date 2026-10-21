from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    blur_score: float
    contrast_score: float
    flags: tuple[str, ...] = ()


@dataclass
class VisionResult:
    success: bool
    angle_degrees: float | None = None
    confidence: float = 0.0
    center: tuple[float, float] | None = None
    radius: float | None = None
    needle_tip: tuple[float, float] | None = None
    annotated_image: np.ndarray | None = field(default=None, repr=False, compare=False)
    error_message: str | None = None


@dataclass(frozen=True)
class ReadingEstimate:
    value: float | None
    unit: str | None
    angle_degrees: float | None
    confidence: float
    status: str
    reason: str


@dataclass
class AgentStep:
    step_name: str
    method: str
    reason: str
    output_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionResult:
    reading: ReadingEstimate
    quality: ImageQuality | None
    action_log: list[AgentStep] = field(default_factory=list)
