from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ImageQuality:
    width: int
    height: int
    blur_score: float
    contrast_score: float
    flags: tuple[str, ...] = ()


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
