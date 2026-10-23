"""GaugeGuard Agent project package."""

from gaugeguard.calibration import (
    GaugeProfile,
    CalibrationError,
    UnknownProfileError,
    CalibrationResult,
    AgreementResult,
    validate_profile,
    normalize_angle_deg,
    pixel_to_angle_deg,
    profile_from_tick_points,
    describe_profile,
    angle_to_value,
    readings_agree,
    get_profile,
)

from gaugeguard.vision import (
    decode_image,
    inspect_image_quality,
    preprocess_for_retry,
    detect_dial_and_needle,
    ImageQuality,
)

from gaugeguard.agent import (
    inspect_upload,
    InspectionResult,
)

from gaugeguard.models import (
    ImageQuality as ModelImageQuality,
    ReadingEstimate,
    AgentStep,
    InspectionResult as ModelInspectionResult,
)

from gaugeguard.evaluation import (
    load_labels,
    validate_labels,
    validate_labels_cli,
    LabelRecord,
    ValidationProblem,
    evaluate_dataset,
    format_evaluation_report,
    EvaluationMetrics,
    EvaluationRecord,
)

__all__ = [
    # Calibration
    "GaugeProfile",
    "CalibrationError",
    "UnknownProfileError",
    "CalibrationResult",
    "AgreementResult",
    "validate_profile",
    "normalize_angle_deg",
    "pixel_to_angle_deg",
    "profile_from_tick_points",
    "describe_profile",
    "angle_to_value",
    "readings_agree",
    "get_profile",
    # Vision
    "decode_image",
    "inspect_image_quality",
    "preprocess_for_retry",
    "detect_dial_and_needle",
    "ImageQuality",
    # Agent
    "inspect_upload",
    "InspectionResult",
    # Models
    "ModelImageQuality",
    "ReadingEstimate",
    "AgentStep",
    "ModelInspectionResult",
    # Evaluation
    "load_labels",
    "validate_labels",
    "validate_labels_cli",
    "LabelRecord",
    "ValidationProblem",
    "evaluate_dataset",
    "format_evaluation_report",
    "EvaluationMetrics",
    "EvaluationRecord",
]