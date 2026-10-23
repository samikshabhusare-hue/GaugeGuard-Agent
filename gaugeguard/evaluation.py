"""Evaluation pipeline for GaugeGuard analog gauge readings.

Calculates Mean Absolute Error (MAE), percentage of readings within configurable tolerance,
retry rate, escalation rate, and incorrect-acceptance rate against verified ground-truth reference labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from gaugeguard.calibration import GaugeProfile, UnknownProfileError, get_profile

logger = logging.getLogger(__name__)


# === Image Loader ===


def load_image_bgr(path: str | Path) -> Any:
    """Load an image from path using cv2.imdecode(np.fromfile(...)).

    Returns None for unreadable or corrupt files without crashing.
    Imports cv2 lazily inside the function.
    """
    try:
        import cv2
        import numpy as np

        path_obj = Path(path)
        if not path_obj.is_file():
            return None
        data = np.fromfile(str(path_obj), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as exc:
        logger.debug("Failed to load image from %s: %s", path, exc)
        return None


# === Label Record and Validation ===


@dataclass(frozen=True)
class LabelRecord:
    """Validated label record from labels.csv."""
    filename: str
    gauge_profile: str
    true_value: float | None
    label_uncertainty: float
    unit: str
    condition: str
    expected_outcome: str
    label_source: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VALID_CONDITIONS = {"clear", "blurry", "tilted", "dark", "obstructed", "other"}
VALID_EXPECTED_OUTCOMES = {"read", "reject_input"}


def load_labels(csv_path: str | Path) -> list[LabelRecord]:
    """Load and parse labels CSV into LabelRecord objects.

    Args:
        csv_path: Path to labels.csv file.

    Returns:
        List of LabelRecord objects (one per valid data row).

    Raises:
        FileNotFoundError: If CSV file does not exist.
        ValueError: If CSV is malformed (reports row number).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")

    records: list[LabelRecord] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        # Filter comments and empty lines, but keep track of original line numbers
        lines = []
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append((line_num, stripped))

    if not lines:
        return records

    # Check header
    header_line_num, header_line = lines[0]
    expected_header = "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes"
    if header_line != expected_header:
        raise ValueError(
            f"Row {header_line_num}: Invalid header. Expected: {expected_header}"
        )

    # Parse data rows
    reader = csv.reader([line for _, line in lines[1:]])
    for idx, row in enumerate(reader):
        orig_line_num = header_line_num + 1 + idx

        # Handle trailing empty columns from trailing commas
        while len(row) > 9 and row[-1] == "":
            row = row[:-1]

        if len(row) < 9:
            raise ValueError(
                f"Row {orig_line_num}: Expected 9 columns, got {len(row)}. "
                f"Columns: filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes"
            )

        filename, gauge_profile, true_value_str, label_uncertainty_str, unit, condition, expected_outcome, label_source, notes = row[:9]

        # Parse true_value (can be empty for reject_input or pending validation)
        true_value: float | None = None
        if true_value_str.strip() != "":
            try:
                true_value = float(true_value_str)
            except ValueError:
                raise ValueError(
                    f"Row {orig_line_num}: true_value '{true_value_str}' is not a valid number"
                )

        # Parse label_uncertainty
        try:
            label_uncertainty = float(label_uncertainty_str)
        except ValueError:
            raise ValueError(
                f"Row {orig_line_num}: label_uncertainty '{label_uncertainty_str}' is not a valid number"
            )

        if label_uncertainty < 0:
            raise ValueError(
                f"Row {orig_line_num}: label_uncertainty must be non-negative, got {label_uncertainty}"
            )

        record = LabelRecord(
            filename=filename.strip(),
            gauge_profile=gauge_profile.strip(),
            true_value=true_value,
            label_uncertainty=label_uncertainty,
            unit=unit.strip(),
            condition=condition.strip(),
            expected_outcome=expected_outcome.strip(),
            label_source=label_source.strip(),
            notes=notes.strip(),
        )
        records.append(record)

    return records


@dataclass
class ValidationProblem:
    """A validation problem found during label validation."""
    severity: str  # "error" or "warning"
    message: str
    row_number: int | None = None
    filename: str | None = None

    def __str__(self) -> str:
        parts = []
        if self.row_number is not None:
            parts.append(f"Row {self.row_number}")
        if self.filename:
            parts.append(f"({self.filename})")
        parts.append(f"[{self.severity.upper()}] {self.message}")
        return " ".join(parts)


def validate_labels(records: list[LabelRecord], images_dir: str | Path) -> list[ValidationProblem]:
    """Validate label records against registered profiles and image files.

    Args:
        records: List of LabelRecord objects from load_labels().
        images_dir: Directory containing image files.

    Returns:
        List of ValidationProblem objects (empty if all valid).
    """
    problems: list[ValidationProblem] = []
    images_path = Path(images_dir)

    # Check for duplicate filenames
    seen_filenames: dict[str, int] = {}
    for idx, rec in enumerate(records):
        if rec.filename in seen_filenames:
            problems.append(ValidationProblem(
                severity="error",
                message=f"Duplicate filename '{rec.filename}' (first seen at row {seen_filenames[rec.filename] + 1})",
                row_number=idx + 1,
                filename=rec.filename,
            ))
        else:
            seen_filenames[rec.filename] = idx

    # Validate each record
    for idx, rec in enumerate(records):
        row_num = idx + 1

        # Check file exists
        img_path = images_path / rec.filename
        if not img_path.is_file():
            problems.append(ValidationProblem(
                severity="error",
                message=f"Image file not found: {img_path}",
                row_number=row_num,
                filename=rec.filename,
            ))

        # Check gauge profile exists
        try:
            profile = get_profile(rec.gauge_profile)
        except UnknownProfileError:
            problems.append(ValidationProblem(
                severity="error",
                message=f"Unknown profile '{rec.gauge_profile}'. Available: {', '.join(sorted(get_profile.__globals__['PROFILES'].keys()))}",
                row_number=row_num,
                filename=rec.filename,
            ))
            profile = None

        if profile is not None:
            # Check true_value presence for "read" outcome
            if rec.expected_outcome == "read":
                if rec.true_value is None:
                    problems.append(ValidationProblem(
                        severity="error",
                        message="true_value is required when expected_outcome == 'read'",
                        row_number=row_num,
                        filename=rec.filename,
                    ))
                else:
                    if not (profile.minimum_value <= rec.true_value <= profile.maximum_value):
                        problems.append(ValidationProblem(
                            severity="error",
                            message=f"true_value {rec.true_value} outside profile range [{profile.minimum_value}, {profile.maximum_value}]",
                            row_number=row_num,
                            filename=rec.filename,
                        ))
            elif rec.expected_outcome == "reject_input":
                if rec.true_value is not None:
                    problems.append(ValidationProblem(
                        severity="error",
                        message="true_value must be empty when expected_outcome == 'reject_input'",
                        row_number=row_num,
                        filename=rec.filename,
                    ))

            # Check unit matches profile
            if rec.unit != profile.unit:
                problems.append(ValidationProblem(
                    severity="warning",
                    message=f"Unit '{rec.unit}' does not match profile unit '{profile.unit}'",
                    row_number=row_num,
                    filename=rec.filename,
                ))

        # Check condition is valid
        if rec.condition not in VALID_CONDITIONS:
            problems.append(ValidationProblem(
                severity="error",
                message=f"Unknown condition '{rec.condition}'. Valid: {sorted(VALID_CONDITIONS)}",
                row_number=row_num,
                filename=rec.filename,
            ))

        # Check expected_outcome is valid
        if rec.expected_outcome not in VALID_EXPECTED_OUTCOMES:
            problems.append(ValidationProblem(
                severity="error",
                message=f"Unknown expected_outcome '{rec.expected_outcome}'. Valid: {sorted(VALID_EXPECTED_OUTCOMES)}",
                row_number=row_num,
                filename=rec.filename,
            ))

        # Check label_source is not empty
        if not rec.label_source.strip():
            problems.append(ValidationProblem(
                severity="error",
                message="label_source is required and cannot be empty",
                row_number=row_num,
                filename=rec.filename,
            ))

    # Check for unlabeled images in images_dir if it exists
    if images_path.is_dir():
        labeled_files = {rec.filename for rec in records}
        for img_file in images_path.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}:
                if img_file.name not in labeled_files:
                    problems.append(ValidationProblem(
                        severity="warning",
                        message=f"Image file has no label row in labels.csv: {img_file.name}",
                        row_number=None,
                        filename=img_file.name,
                    ))

    return problems


def validate_labels_cli(csv_path: str | Path, images_dir: str | Path) -> int:
    """CLI entry point for label validation."""
    try:
        records = load_labels(csv_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 1

    if not records:
        print("Validation result: 0 labelled images (header-only CSV)")
        return 0

    problems = validate_labels(records, images_dir)
    errors = [p for p in problems if p.severity == "error"]
    warnings = [p for p in problems if p.severity == "warning"]

    if errors:
        print(f"Validation FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        for prob in errors:
            print(f"  {prob}")
        for prob in warnings:
            print(f"  {prob}")
        return 1

    if warnings:
        print(f"Validation PASSED with {len(warnings)} warning(s):")
        for prob in warnings:
            print(f"  {prob}")
    else:
        print(f"Validation PASSED: {len(records)} labelled image(s), no issues")

    return 0


# === Pipeline Adapter and Runner ===


def run_pipeline(
    image_bgr: Any,
    profile: GaugeProfile | None = None,
) -> dict[str, Any]:
    """Adapter coupling evaluation harness to the real agent.py interface.

    Adapts agent.inspect_upload to return a normalized dict:
    {status, reading, retried, attempts, first_pass_reading, action_log, reason}
    """
    import cv2
    from gaugeguard.agent import inspect_upload
    from gaugeguard.calibration import angle_to_value, get_profile

    if profile is None:
        profile = get_profile("EXAMPLE_270")

    if isinstance(image_bgr, bytes):
        image_bytes = image_bgr
    else:
        success, encoded = cv2.imencode(".png", image_bgr)
        if not success:
            raise ValueError("Failed to encode image to PNG bytes for agent pipeline")
        image_bytes = encoded.tobytes()

    inspection = inspect_upload(image_bytes, profile=profile)

    # Extract first-pass reading
    first_pass_reading = None
    if inspection.attempts:
        first_attempt = inspection.attempts[0]
        angle_deg = first_attempt.get("angle_deg")
        if angle_deg is not None:
            cal = angle_to_value(angle_deg, profile)
            first_pass_reading = cal.value
    elif inspection.action_log:
        for step in inspection.action_log:
            if getattr(step, "step_name", None) == "initial_detection":
                out = getattr(step, "output_summary", {})
                ang = out.get("angle_degrees")
                if ang is not None:
                    cal = angle_to_value(ang, profile)
                    first_pass_reading = cal.value
                break

    if first_pass_reading is None and len(inspection.attempts) <= 1:
        first_pass_reading = inspection.reading.value

    # Format action log to list of dicts
    action_log_dicts: list[dict[str, Any]] = []
    for step in inspection.action_log:
        if hasattr(step, "to_dict"):
            action_log_dicts.append(step.to_dict())
        elif hasattr(step, "__dataclass_fields__"):
            action_log_dicts.append(asdict(step))
        elif isinstance(step, dict):
            action_log_dicts.append(step)
        else:
            action_log_dicts.append({"step_name": str(step)})

    return {
        "status": inspection.reading.status,
        "reading": inspection.reading.value,
        "retried": len(inspection.attempts) > 1,
        "attempts": max(1, len(inspection.attempts)),
        "first_pass_reading": first_pass_reading,
        "action_log": action_log_dicts,
        "reason": inspection.reading.reason or "",
    }


class StubPipeline:
    """Stub pipeline used strictly for unit testing the evaluation harness.

    Never used for reported results.
    """

    def __init__(
        self,
        status: str = "accepted",
        reading: float | None = 50.0,
        retried: bool = False,
        attempts: int = 1,
        first_pass_reading: float | None = 50.0,
        action_log: list[dict[str, Any]] | None = None,
        reason: str = "stub",
        raise_exception: Exception | None = None,
    ) -> None:
        self.status = status
        self.reading = reading
        self.retried = retried
        self.attempts = attempts
        self.first_pass_reading = first_pass_reading
        self.action_log = action_log or []
        self.reason = reason
        self.raise_exception = raise_exception

    def __call__(
        self,
        image_bgr: Any,
        profile: GaugeProfile | None = None,
    ) -> dict[str, Any]:
        if self.raise_exception is not None:
            raise self.raise_exception
        return {
            "status": self.status,
            "reading": self.reading,
            "retried": self.retried,
            "attempts": self.attempts,
            "first_pass_reading": self.first_pass_reading,
            "action_log": list(self.action_log),
            "reason": self.reason,
        }


def capture_environment() -> dict[str, Any]:
    """Capture runtime environment metadata.

    Includes python version, platform, cv2.__version__, numpy version,
    and optional env vars GAUGEGUARD_RUN_LABEL / EC2_INSTANCE_TYPE (names/values only; no credentials).
    If cv2 is 4.x, includes warning 'OpenCV 5 condition not met'.
    """
    import numpy as np

    try:
        import cv2
        cv2_ver = getattr(cv2, "__version__", "unknown")
    except ImportError:
        cv2_ver = "not_installed"

    env_info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cv2_version": cv2_ver,
        "numpy_version": np.__version__,
    }

    run_label = os.environ.get("GAUGEGUARD_RUN_LABEL")
    if run_label is not None:
        env_info["GAUGEGUARD_RUN_LABEL"] = run_label

    ec2_type = os.environ.get("EC2_INSTANCE_TYPE")
    if ec2_type is not None:
        env_info["EC2_INSTANCE_TYPE"] = ec2_type

    warnings: list[str] = []
    if cv2_ver not in ("not_installed", "unknown"):
        major_ver = cv2_ver.split(".")[0]
        if major_ver == "4":
            warnings.append("OpenCV 5 condition not met")

    if warnings:
        env_info["warnings"] = warnings

    return env_info


# === Pure Metric Functions ===


@dataclass(frozen=True)
class EvalRecord:
    """Per-evaluation record for pure metric computation."""
    filename: str
    condition: str
    expected_outcome: str  # "read" or "reject_input"
    true_value: float | None
    tolerance: float  # absolute, in value units — FIXED at load time
    status: str  # "accepted" | "needs_human" | "rejected_input" | "error"
    reading: float | None
    retried: bool
    attempts: int
    latency_ms: float | None
    first_pass_reading: float | None
    error_message: str = ""
    source: str = "agent"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Rate = numerator / denominator; None if denominator == 0."""
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Compute pure metrics from EvalRecord list."""
    recs = list(records)

    n_total = len(recs)
    n_evaluable = sum(1 for r in recs if r.status != "error")
    n_accepted = sum(1 for r in recs if r.status == "accepted")
    n_needs_human = sum(1 for r in recs if r.status in ("needs_human", "needs_retake"))
    n_errors = sum(1 for r in recs if r.status == "error")
    n_rejected_input = sum(1 for r in recs if r.status == "rejected_input")

    coverage = _safe_rate(n_accepted, n_evaluable)

    accepted_with_truth = [
        r for r in recs
        if r.status == "accepted" and r.true_value is not None and r.reading is not None
    ]
    mae = None
    if accepted_with_truth:
        mae = statistics.mean(
            abs(r.reading - r.true_value) for r in accepted_with_truth
        )

    n_within_tolerance = sum(
        1 for r in accepted_with_truth
        if abs(r.reading - r.true_value) <= r.tolerance
    )
    within_tolerance_rate = _safe_rate(n_within_tolerance, n_accepted)

    n_incorrect = sum(
        1 for r in accepted_with_truth
        if abs(r.reading - r.true_value) > r.tolerance
    )
    incorrect_acceptance_rate = _safe_rate(n_incorrect, n_accepted)

    n_retried = sum(1 for r in recs if r.retried)
    retry_rate = _safe_rate(n_retried, n_evaluable)

    escalation_rate = _safe_rate(n_needs_human, n_evaluable)

    expected_reject = [r for r in recs if r.expected_outcome == "reject_input"]
    correct_rejection_rate: float | None = None
    if expected_reject:
        correct_rejection_rate = _safe_rate(
            sum(1 for r in expected_reject if r.status == "rejected_input"),
            len(expected_reject)
        )
    false_rejection_count = sum(
        1 for r in recs
        if r.expected_outcome == "read" and r.status == "rejected_input"
    )

    latencies = [r.latency_ms for r in recs if r.latency_ms is not None]
    latency_summary_dict: dict[str, Any] = {"n": len(latencies)}
    if latencies:
        latency_summary_dict["mean"] = statistics.mean(latencies)
        latency_summary_dict["median"] = statistics.median(latencies)
        sorted_lat = sorted(latencies)
        idx_95 = math.ceil(0.95 * len(sorted_lat)) - 1
        latency_summary_dict["p95"] = sorted_lat[max(0, idx_95)]
    if len(latencies) < 20:
        latency_summary_dict["p95_unreliable"] = True

    notes: list[str] = []
    if n_accepted == 0:
        notes.append("n_accepted=0: within_tolerance_rate and incorrect_acceptance_rate undefined")
    if n_evaluable == 0:
        notes.append("n_evaluable=0: coverage, retry_rate, escalation_rate undefined")

    return {
        "n_total": n_total,
        "n_evaluable": n_evaluable,
        "n_accepted": n_accepted,
        "n_needs_human": n_needs_human,
        "n_errors": n_errors,
        "n_rejected_input": n_rejected_input,
        "coverage": (coverage, n_accepted, n_evaluable) if coverage is not None else (None, n_accepted, n_evaluable),
        "mae": (mae, len(accepted_with_truth)) if mae is not None else (None, 0),
        "within_tolerance_rate": (within_tolerance_rate, n_within_tolerance, n_accepted) if within_tolerance_rate is not None else (None, n_within_tolerance, n_accepted),
        "incorrect_acceptance_rate": (incorrect_acceptance_rate, n_incorrect, n_accepted) if incorrect_acceptance_rate is not None else (None, n_incorrect, n_accepted),
        "retry_rate": (retry_rate, n_retried, n_evaluable) if retry_rate is not None else (None, n_retried, n_evaluable),
        "escalation_rate": (escalation_rate, n_needs_human, n_evaluable) if escalation_rate is not None else (None, n_needs_human, n_evaluable),
        "correct_rejection_rate": (correct_rejection_rate, sum(1 for r in expected_reject if r.status == "rejected_input"), len(expected_reject)) if correct_rejection_rate is not None else (None, 0, len(expected_reject)),
        "false_rejection_count": false_rejection_count,
        "latency_summary": latency_summary_dict,
        "note": "; ".join(notes) if notes else "all denominators > 0",
    }


def compute_baseline_metrics(records: Sequence[EvalRecord]) -> dict[str, Any] | str:
    """Compute first-pass-only baseline metrics using first_pass_reading.

    Computed as if every first-pass reading were accepted (no quality gate).
    If first_pass_reading is unavailable across all records, returns 'not available'.
    """
    has_any_first_pass = any(r.first_pass_reading is not None for r in records)
    if not has_any_first_pass:
        return "not available"

    baseline_records: list[EvalRecord] = []
    for r in records:
        if r.first_pass_reading is not None:
            baseline_records.append(
                EvalRecord(
                    filename=r.filename,
                    condition=r.condition,
                    expected_outcome=r.expected_outcome,
                    true_value=r.true_value,
                    tolerance=r.tolerance,
                    status="accepted",
                    reading=r.first_pass_reading,
                    retried=False,
                    attempts=1,
                    latency_ms=r.latency_ms,
                    first_pass_reading=r.first_pass_reading,
                    error_message="",
                    source=r.source,
                )
            )
        else:
            baseline_records.append(
                EvalRecord(
                    filename=r.filename,
                    condition=r.condition,
                    expected_outcome=r.expected_outcome,
                    true_value=r.true_value,
                    tolerance=r.tolerance,
                    status="rejected_input" if r.status != "error" else "error",
                    reading=None,
                    retried=False,
                    attempts=1,
                    latency_ms=r.latency_ms,
                    first_pass_reading=None,
                    error_message=r.error_message,
                    source=r.source,
                )
            )

    return compute_metrics(baseline_records)


def run_evaluation(
    labels_csv: str | Path,
    images_dir: str | Path,
    pipeline: Any = run_pipeline,
) -> tuple[list[EvalRecord], dict[str, Any]]:
    """Run evaluation on a labelled dataset.

    Args:
        labels_csv: Path to labels.csv.
        images_dir: Directory containing images.
        pipeline: Pipeline callable (default: run_pipeline).

    Returns:
        (records, details) tuple where records is a list of EvalRecord and details
        contains per-image metadata, action logs, and reasons.
    """
    labels_path = Path(labels_csv)
    images_path = Path(images_dir)

    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    label_records = load_labels(labels_path)
    if not label_records:
        return [], {}

    records: list[EvalRecord] = []
    details: dict[str, Any] = {}

    for rec in label_records:
        try:
            profile = get_profile(rec.gauge_profile)
            full_scale = profile.maximum_value - profile.minimum_value
            tolerance = profile.eval_tolerance_fraction * full_scale
        except Exception:
            profile = None
            tolerance = 5.0

        img_path = images_path / rec.filename
        t0 = time.perf_counter()
        img_bgr = load_image_bgr(img_path)

        if img_bgr is None:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error_msg = f"Image '{rec.filename}' unreadable, corrupt, or missing"
            eval_rec = EvalRecord(
                filename=rec.filename,
                condition=rec.condition,
                expected_outcome=rec.expected_outcome,
                true_value=rec.true_value,
                tolerance=tolerance,
                status="rejected_input",
                reading=None,
                retried=False,
                attempts=0,
                latency_ms=round(latency_ms, 2),
                first_pass_reading=None,
                error_message=error_msg,
                source="harness",
            )
            records.append(eval_rec)
            details[rec.filename] = {
                "status": "rejected_input",
                "source": "harness",
                "reading": None,
                "first_pass_reading": None,
                "action_log": [],
                "reason": error_msg,
            }
            continue

        try:
            call_start = time.perf_counter()
            out = pipeline(img_bgr, profile=profile)
            latency_ms = (time.perf_counter() - call_start) * 1000.0

            status = out.get("status", "error")
            reading = out.get("reading")
            retried = bool(out.get("retried", False))
            attempts = int(out.get("attempts", 1))
            first_pass_reading = out.get("first_pass_reading")
            reason = str(out.get("reason", ""))
            action_log = out.get("action_log", [])

            eval_rec = EvalRecord(
                filename=rec.filename,
                condition=rec.condition,
                expected_outcome=rec.expected_outcome,
                true_value=rec.true_value,
                tolerance=tolerance,
                status=status,
                reading=reading,
                retried=retried,
                attempts=attempts,
                latency_ms=round(latency_ms, 2),
                first_pass_reading=first_pass_reading,
                error_message="",
                source="agent",
            )
            records.append(eval_rec)
            details[rec.filename] = {
                "status": status,
                "source": "agent",
                "reading": reading,
                "first_pass_reading": first_pass_reading,
                "action_log": action_log,
                "reason": reason,
            }
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error_msg = str(exc)
            eval_rec = EvalRecord(
                filename=rec.filename,
                condition=rec.condition,
                expected_outcome=rec.expected_outcome,
                true_value=rec.true_value,
                tolerance=tolerance,
                status="error",
                reading=None,
                retried=False,
                attempts=0,
                latency_ms=round(latency_ms, 2),
                first_pass_reading=None,
                error_message=error_msg,
                source="harness",
            )
            records.append(eval_rec)
            details[rec.filename] = {
                "status": "error",
                "source": "harness",
                "reading": None,
                "first_pass_reading": None,
                "action_log": [],
                "reason": error_msg,
            }

    return records, details


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half_width = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    low = max(0.0, centre - half_width)
    high = min(1.0, centre + half_width)
    return (low, high)


def metrics_by_condition(records: Sequence[EvalRecord]) -> dict[str, dict[str, Any]]:
    """Group compute_metrics by condition."""
    groups: dict[str, list[EvalRecord]] = {}
    for r in records:
        groups.setdefault(r.condition, []).append(r)
    return {cond: compute_metrics(group) for cond, group in groups.items()}


def latency_summary(values_ms: Sequence[float | None]) -> dict[str, Any]:
    """Latency statistics with reliability flag for small samples."""
    clean = [v for v in values_ms if v is not None]
    result: dict[str, Any] = {"n": len(clean)}
    if clean:
        result["mean"] = statistics.mean(clean)
        result["median"] = statistics.median(clean)
        sorted_v = sorted(clean)
        idx_95 = math.ceil(0.95 * len(sorted_v)) - 1
        result["p95"] = sorted_v[max(0, idx_95)]
    if len(clean) < 20:
        result["p95_unreliable"] = True
    return result


def compute_mae(
    predictions: Sequence[float],
    ground_truths: Sequence[float],
) -> float | None:
    """Calculate Mean Absolute Error between predicted and ground-truth values."""
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return None
    errors = [abs(float(p) - float(g)) for p, g in zip(predictions, ground_truths)]
    return sum(errors) / len(errors)


def compute_within_tolerance(
    predictions: Sequence[float],
    ground_truths: Sequence[float],
    tolerance: float,
) -> tuple[int, float]:
    """Calculate count and percentage of predictions within an absolute tolerance."""
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return 0, 0.0
    if tolerance < 0:
        raise ValueError(f"Tolerance must be non-negative, got {tolerance}")
    count = sum(1 for p, g in zip(predictions, ground_truths) if abs(float(p) - float(g)) <= tolerance)
    percentage = (count / len(predictions)) * 100.0
    return count, percentage


def parse_boolean(val: Any) -> bool | None:
    """Parse common string representations of booleans."""
    if val is None or val == "":
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


# === Report Generation & Run Comparison ===


def _format_rate_with_ci(num: int, denom: int, z: float = 1.96) -> str:
    """Format rate with count (num/denom) and Wilson confidence interval."""
    if denom == 0:
        return "N/A (0/0)"
    pct = (num / denom) * 100.0
    ci = wilson_interval(num, denom, z=z)
    if ci is None:
        return f"{pct:.1f}% ({num}/{denom})"
    low_pct = ci[0] * 100.0
    high_pct = ci[1] * 100.0
    return f"{pct:.1f}% ({num}/{denom}) [{low_pct:.1f}%, {high_pct:.1f}%]"


def make_report(results_dir: str | Path) -> Path:
    """Generate report_tables.md from an evaluation run directory.

    Includes:
    1) Overall metrics table with counts and Wilson intervals, agent vs baseline
    2) Per-condition table
    3) Escalation table with reasons and flags from action logs
    4) Failure cases table (wrong accepted, errors, false rejections, missed escalations)
    5) Latency summary with n and environment metadata

    Each table ends with a note stating sample size and that results do not
    generalize beyond this gauge and these images.
    """
    res_path = Path(results_dir)
    records_file = res_path / "records.csv"
    metrics_file = res_path / "metrics.json"
    env_file = res_path / "environment.json"
    action_logs_file = res_path / "action_logs.json"

    if not records_file.is_file():
        raise FileNotFoundError(f"records.csv not found in {res_path}")

    # Load records
    records: list[dict[str, Any]] = []
    with open(records_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "filename": row["filename"],
                "condition": row.get("condition", "unspecified"),
                "expected_outcome": row.get("expected_outcome", "read"),
                "true_value": float(row["true_value"]) if row.get("true_value") not in ("", None) else None,
                "tolerance": float(row["tolerance"]) if row.get("tolerance") not in ("", None) else 5.0,
                "status": row.get("status", "error"),
                "reading": float(row["reading"]) if row.get("reading") not in ("", None) else None,
                "retried": parse_boolean(row.get("retried")) is True,
                "attempts": int(row["attempts"]) if row.get("attempts") not in ("", None) else 1,
                "latency_ms": float(row["latency_ms"]) if row.get("latency_ms") not in ("", None) else None,
                "first_pass_reading": float(row["first_pass_reading"]) if row.get("first_pass_reading") not in ("", None) else None,
                "error_message": row.get("error_message", ""),
                "source": row.get("source", "agent"),
            })

    # Load metrics if present
    metrics_data: dict[str, Any] = {}
    if metrics_file.is_file():
        try:
            metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            metrics_data = {}

    baseline_metrics = metrics_data.get("baseline", {})

    # Load environment if present
    env_data: dict[str, Any] = {}
    if env_file.is_file():
        try:
            env_data = json.loads(env_file.read_text(encoding="utf-8"))
        except Exception:
            env_data = {}

    # Load action logs if present
    action_logs_map: dict[str, Any] = {}
    if action_logs_file.is_file():
        try:
            action_logs_map = json.loads(action_logs_file.read_text(encoding="utf-8"))
        except Exception:
            action_logs_map = {}

    n_total = len(records)
    n_evaluable = sum(1 for r in records if r["status"] != "error")
    n_accepted = sum(1 for r in records if r["status"] == "accepted")
    n_escalated = sum(1 for r in records if r["status"] in ("needs_human", "needs_retake"))
    n_errors = sum(1 for r in records if r["status"] == "error")
    n_rejected = sum(1 for r in records if r["status"] == "rejected_input")
    n_retried = sum(1 for r in records if r["retried"])

    sample_size_note = (
        f"*Note: Sample size n = {n_total}. Results do not generalize beyond this gauge and these images.*"
    )

    lines: list[str] = [
        f"# GaugeGuard Evaluation Report: `{res_path.name}`",
        "",
        f"Generated from run output in `{res_path}`.",
        "",
    ]

    # --- 1) Overall Metrics Table ---
    lines.append("## 1. Overall Metrics: Agent vs Baseline")
    lines.append("")
    lines.append("| Metric | Agent Result | Baseline (First-Pass Only) | Definition |")
    lines.append("|:---|:---|:---|:---|")

    agent_cov = _format_rate_with_ci(n_accepted, n_evaluable)
    agent_accepted_with_truth = [
        r for r in records
        if r["status"] == "accepted" and r["true_value"] is not None and r["reading"] is not None
    ]
    agent_mae = (
        f"{statistics.mean(abs(r['reading'] - r['true_value']) for r in agent_accepted_with_truth):.2f} (n={len(agent_accepted_with_truth)})"
        if agent_accepted_with_truth else "N/A (n=0)"
    )
    n_agent_within = sum(1 for r in agent_accepted_with_truth if abs(r["reading"] - r["true_value"]) <= r["tolerance"])
    agent_within = _format_rate_with_ci(n_agent_within, n_accepted)

    n_agent_incorrect = sum(1 for r in agent_accepted_with_truth if abs(r["reading"] - r["true_value"]) > r["tolerance"])
    agent_incorrect = _format_rate_with_ci(n_agent_incorrect, n_accepted)

    agent_retry = _format_rate_with_ci(n_retried, n_evaluable)
    agent_escalation = _format_rate_with_ci(n_escalated, n_evaluable)

    # Baseline numbers
    if isinstance(baseline_metrics, dict) and baseline_metrics:
        base_n_acc = baseline_metrics.get("n_accepted", 0)
        base_n_eval = baseline_metrics.get("n_evaluable", n_total)
        base_cov = _format_rate_with_ci(base_n_acc, base_n_eval)
        base_mae_val = baseline_metrics.get("mae", (None, 0))[0]
        base_mae_count = baseline_metrics.get("mae", (None, 0))[1]
        base_mae = f"{base_mae_val:.2f} (n={base_mae_count})" if base_mae_val is not None else "N/A (n=0)"
        base_within_num = baseline_metrics.get("within_tolerance_rate", (None, 0, base_n_acc))[1]
        base_within = _format_rate_with_ci(base_within_num, base_n_acc)
        base_inc_num = baseline_metrics.get("incorrect_acceptance_rate", (None, 0, base_n_acc))[1]
        base_incorrect = _format_rate_with_ci(base_inc_num, base_n_acc)
        base_retry = f"0.0% (0/{base_n_eval})"
        base_escalation = f"0.0% (0/{base_n_eval})"
    else:
        base_cov = "not available"
        base_mae = "not available"
        base_within = "not available"
        base_incorrect = "not available"
        base_retry = "not available"
        base_escalation = "not available"

    lines.append(f"| **Coverage** | {agent_cov} | {base_cov} | Accepted readings / Evaluable samples |")
    lines.append(f"| **Mean Absolute Error (MAE)** | {agent_mae} | {base_mae} | Mean error on accepted readings with ground truth |")
    lines.append(f"| **Within-Tolerance Rate** | {agent_within} | {base_within} | Accepted readings with error ≤ tolerance / Accepted |")
    lines.append(f"| **Incorrect Acceptance Rate** | {agent_incorrect} | {base_incorrect} | Accepted readings with error > tolerance / Accepted |")
    lines.append(f"| **Retry Rate** | {agent_retry} | {base_retry} | Inspections requiring retry pass / Evaluable |")
    lines.append(f"| **Human Escalation Rate** | {agent_escalation} | {base_escalation} | Inspections routed to human review / Evaluable |")
    lines.append(f"| **Harness / Agent Errors** | {n_errors} ({n_errors/n_total*100:.1f}%) | N/A | Processing failures / Total |")
    lines.append(f"| **Rejected Inputs** | {n_rejected} ({n_rejected/n_total*100:.1f}%) | N/A | Corrupt/unreadable inputs rejected / Total |")
    lines.append("")
    lines.append(sample_size_note)
    lines.append("")

    # --- 2) Per-Condition Table ---
    lines.append("## 2. Per-Condition Performance Breakdown")
    lines.append("")
    lines.append("| Condition | Total (n) | Accepted | Coverage (95% CI) | Escalated | MAE | Within Tol (95% CI) | Errors | Rejected |")
    lines.append("|:---|:---|:---|:---|:---|:---|:---|:---|:---|")

    cond_groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        cond_groups.setdefault(r["condition"], []).append(r)

    for cond, group in sorted(cond_groups.items()):
        c_total = len(group)
        c_eval = sum(1 for r in group if r["status"] != "error")
        c_acc = sum(1 for r in group if r["status"] == "accepted")
        c_esc = sum(1 for r in group if r["status"] in ("needs_human", "needs_retake"))
        c_err = sum(1 for r in group if r["status"] == "error")
        c_rej = sum(1 for r in group if r["status"] == "rejected_input")

        c_cov_str = _format_rate_with_ci(c_acc, c_eval)
        c_acc_truth = [r for r in group if r["status"] == "accepted" and r["true_value"] is not None and r["reading"] is not None]
        c_mae_str = f"{statistics.mean(abs(r['reading'] - r['true_value']) for r in c_acc_truth):.2f}" if c_acc_truth else "N/A"
        c_within_num = sum(1 for r in c_acc_truth if abs(r["reading"] - r["true_value"]) <= r["tolerance"])
        c_within_str = _format_rate_with_ci(c_within_num, c_acc)

        lines.append(
            f"| `{cond}` | {c_total} | {c_acc} | {c_cov_str} | {c_esc} | {c_mae_str} | {c_within_str} | {c_err} | {c_rej} |"
        )

    lines.append("")
    lines.append(sample_size_note)
    lines.append("")

    # --- 3) Escalation Table ---
    lines.append("## 3. Human Escalation Table")
    lines.append("")
    lines.append("Images routed to human review (`needs_human` / `needs_retake`), with triggering flags and rationale:")
    lines.append("")
    lines.append("| Filename | Condition | Attempts | Reading | Triggering Flags / Rationale |")
    lines.append("|:---|:---|:---|:---|:---|")

    escalated_recs = [r for r in records if r["status"] in ("needs_human", "needs_retake")]
    if escalated_recs:
        for r in escalated_recs:
            fn = r["filename"]
            read_str = f"{r['reading']:.2f}" if r["reading"] is not None else "None"
            flags_or_reasons: list[str] = []
            if fn in action_logs_map:
                for step in action_logs_map[fn]:
                    out_sum = step.get("output_summary", {})
                    if "flags" in out_sum and out_sum["flags"]:
                        flags_or_reasons.extend(out_sum["flags"])
                    if "reason" in out_sum and out_sum["reason"]:
                        flags_or_reasons.append(out_sum["reason"])
                    if "decision" in out_sum and out_sum.get("needs_human"):
                        flags_or_reasons.append(f"decision: {out_sum['decision']}")
            if r["error_message"]:
                flags_or_reasons.append(r["error_message"])

            dedup_reasons: list[str] = []
            for item in flags_or_reasons:
                item_clean = str(item).strip()
                if item_clean and item_clean not in dedup_reasons:
                    dedup_reasons.append(item_clean)

            reason_str = "; ".join(dedup_reasons) if dedup_reasons else r["status"]
            lines.append(f"| `{fn}` | `{r['condition']}` | {r['attempts']} | {read_str} | {reason_str} |")
    else:
        lines.append("| *None* | - | - | - | No images routed to human review (0 escalations) |")

    lines.append("")
    lines.append(sample_size_note)
    lines.append("")

    # --- 4) Failure Cases Table ---
    lines.append("## 4. Failure Cases")
    lines.append("")
    lines.append("Detailed analysis of failure events: wrong accepted readings, errors, false rejections, and missed escalations.")
    lines.append("")
    lines.append("| Filename | Failure Category | Predicted | True Value | Absolute Error | Tolerance | Reason / Description |")
    lines.append("|:---|:---|:---|:---|:---|:---|:---|")

    failures: list[dict[str, Any]] = []

    for r in records:
        # a) Accepted reading that was wrong (|err| > tol)
        if r["status"] == "accepted" and r["true_value"] is not None and r["reading"] is not None:
            err = abs(r["reading"] - r["true_value"])
            if err > r["tolerance"]:
                failures.append({
                    "filename": r["filename"],
                    "category": "Wrong Accepted Reading",
                    "predicted": f"{r['reading']:.2f}",
                    "true_value": f"{r['true_value']:.2f}",
                    "error": f"{err:.2f}",
                    "tolerance": f"±{r['tolerance']:.2f}",
                    "reason": f"Reading accepted but error {err:.2f} exceeds tolerance {r['tolerance']:.2f}",
                })

        # b) Harness or agent error
        if r["status"] == "error":
            failures.append({
                "filename": r["filename"],
                "category": f"Pipeline Error ({r['source']})",
                "predicted": "None",
                "true_value": f"{r['true_value']:.2f}" if r["true_value"] is not None else "N/A",
                "error": "N/A",
                "tolerance": f"±{r['tolerance']:.2f}",
                "reason": r["error_message"] or "Internal pipeline exception",
            })

        # c) False rejection (expected read but rejected_input)
        if r["expected_outcome"] == "read" and r["status"] == "rejected_input":
            failures.append({
                "filename": r["filename"],
                "category": f"False Rejection ({r['source']})",
                "predicted": "None",
                "true_value": f"{r['true_value']:.2f}" if r["true_value"] is not None else "N/A",
                "error": "N/A",
                "tolerance": f"±{r['tolerance']:.2f}",
                "reason": r["error_message"] or "Legitimate readable image was rejected",
            })

        # d) Missed escalation on obstructed image
        if r["condition"] == "obstructed" and r["status"] == "accepted":
            err_val = abs(r["reading"] - r["true_value"]) if (r["reading"] is not None and r["true_value"] is not None) else None
            failures.append({
                "filename": r["filename"],
                "category": "Missed Escalation",
                "predicted": f"{r['reading']:.2f}" if r["reading"] is not None else "None",
                "true_value": f"{r['true_value']:.2f}" if r["true_value"] is not None else "N/A",
                "error": f"{err_val:.2f}" if err_val is not None else "N/A",
                "tolerance": f"±{r['tolerance']:.2f}",
                "reason": "Obstructed image was accepted without human review",
            })

    if failures:
        for f_item in failures:
            lines.append(
                f"| `{f_item['filename']}` | **{f_item['category']}** | {f_item['predicted']} | {f_item['true_value']} | {f_item['error']} | {f_item['tolerance']} | {f_item['reason']} |"
            )
    else:
        lines.append("| *None* | - | - | - | - | - | No failure cases observed (0 failures) |")

    lines.append("")
    lines.append(sample_size_note)
    lines.append("")

    # --- 5) Latency & Environment Summary ---
    lines.append("## 5. Latency & Environment Summary")
    lines.append("")
    latencies = [r["latency_ms"] for r in records if r["latency_ms"] is not None]

    lines.append("### Latency Profile")
    lines.append("")
    lines.append("| Metric | Value | Note |")
    lines.append("|:---|:---|:---|")
    lines.append(f"| **Sample Count (n)** | {len(latencies)} | Total timed samples |")
    if latencies:
        mean_l = statistics.mean(latencies)
        med_l = statistics.median(latencies)
        sorted_l = sorted(latencies)
        p95_idx = math.ceil(0.95 * len(sorted_l)) - 1
        p95_l = sorted_l[max(0, p95_idx)]
        min_l = min(latencies)
        max_l = max(latencies)
        p95_unreliable = len(latencies) < 20

        lines.append(f"| **Mean Latency** | {mean_l:.1f} ms | Arithmetic average per sample |")
        lines.append(f"| **Median Latency** | {med_l:.1f} ms | 50th percentile |")
        lines.append(f"| **p95 Latency** | {p95_l:.1f} ms | 95th percentile {'*(unreliable: n < 20)*' if p95_unreliable else ''} |")
        lines.append(f"| **Min Latency** | {min_l:.1f} ms | Fastest execution |")
        lines.append(f"| **Max Latency** | {max_l:.1f} ms | Slowest execution |")
    else:
        lines.append("| **Latency** | N/A | No latency measurements recorded |")

    lines.append("")
    lines.append("### Execution Environment")
    lines.append("")
    lines.append("| Property | Recorded Value |")
    lines.append("|:---|:---|")
    lines.append(f"| **Compute Platform / OS** | `{env_data.get('platform', 'unknown')}` |")
    python_ver_short = env_data.get("python_version", sys.version).splitlines()[0]
    lines.append(f"| **Python Version** | `{python_ver_short}` |")
    lines.append(f"| **OpenCV (`cv2`) Version** | `{env_data.get('cv2_version', 'unknown')}` |")
    lines.append(f"| **NumPy Version** | `{env_data.get('numpy_version', 'unknown')}` |")
    instance_type = env_data.get("EC2_INSTANCE_TYPE") or ("Local Workstation" if not env_data.get("GAUGEGUARD_RUN_LABEL") else env_data.get("GAUGEGUARD_RUN_LABEL"))
    lines.append(f"| **Compute Instance Type** | `{instance_type}` |")
    if env_data.get("GAUGEGUARD_RUN_LABEL"):
        lines.append(f"| **Run Label** | `{env_data.get('GAUGEGUARD_RUN_LABEL')}` |")
    warnings_list = env_data.get("warnings", [])
    lines.append(f"| **Environment Warnings** | `{', '.join(warnings_list) if warnings_list else 'None'}` |")

    lines.append("")
    lines.append(sample_size_note)
    lines.append("")

    report_content = "\n".join(lines)
    out_file = res_path / "report_tables.md"
    out_file.write_text(report_content, encoding="utf-8")
    return out_file


def compare_runs(dir_a: str | Path, dir_b: str | Path) -> str:
    """Generate a markdown comparison between two evaluation run directories (e.g. local vs EC2).

    Warns if labelled sets differ or n < 20.
    Ends each table with the sample size note.
    """
    path_a = Path(dir_a)
    path_b = Path(dir_b)

    rec_a_file = path_a / "records.csv"
    rec_b_file = path_b / "records.csv"
    env_a_file = path_a / "environment.json"
    env_b_file = path_b / "environment.json"

    if not rec_a_file.is_file():
        raise FileNotFoundError(f"records.csv not found in {path_a}")
    if not rec_b_file.is_file():
        raise FileNotFoundError(f"records.csv not found in {path_b}")

    def _read_records(fpath: Path) -> list[dict[str, Any]]:
        recs = []
        with open(fpath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                recs.append({
                    "filename": row["filename"],
                    "status": row.get("status", "error"),
                    "true_value": float(row["true_value"]) if row.get("true_value") not in ("", None) else None,
                    "reading": float(row["reading"]) if row.get("reading") not in ("", None) else None,
                    "tolerance": float(row["tolerance"]) if row.get("tolerance") not in ("", None) else 5.0,
                    "latency_ms": float(row["latency_ms"]) if row.get("latency_ms") not in ("", None) else None,
                })
        return recs

    recs_a = _read_records(rec_a_file)
    recs_b = _read_records(rec_b_file)

    env_a: dict[str, Any] = json.loads(env_a_file.read_text(encoding="utf-8")) if env_a_file.is_file() else {}
    env_b: dict[str, Any] = json.loads(env_b_file.read_text(encoding="utf-8")) if env_b_file.is_file() else {}

    n_a = len(recs_a)
    n_b = len(recs_b)

    files_a = {r["filename"] for r in recs_a}
    files_b = {r["filename"] for r in recs_b}

    lines: list[str] = [
        f"# Comparison of Evaluation Runs: `{path_a.name}` vs `{path_b.name}`",
        "",
    ]

    warnings: list[str] = []
    if files_a != files_b:
        warnings.append(
            f"**Labelled sets differ**: Run A has {len(files_a)} images; Run B has {len(files_b)} images. "
            f"({len(files_a ^ files_b)} discordant filenames). Comparisons may reflect different image sets."
        )
    if n_a < 20 or n_b < 20:
        warnings.append(
            f"**Small sample size**: n_A={n_a}, n_B={n_b} (n < 20). Latency percentiles and proportions are statistically noisy."
        )

    if warnings:
        lines.append("> [!WARNING]")
        for w in warnings:
            lines.append(f"> - {w}")
        lines.append("")

    note_both = f"*Note: Sample size n_A = {n_a}, n_B = {n_b}. Results do not generalize beyond this gauge and these images.*"

    # Table 1: Environment Comparison
    lines.append("## 1. Environment Comparison")
    lines.append("")
    lines.append(f"| Property | Run A (`{path_a.name}`) | Run B (`{path_b.name}`) | Match? |")
    lines.append("|:---|:---|:---|:---|")
    inst_a = env_a.get("EC2_INSTANCE_TYPE") or env_a.get("GAUGEGUARD_RUN_LABEL") or "Local Workstation"
    inst_b = env_b.get("EC2_INSTANCE_TYPE") or env_b.get("GAUGEGUARD_RUN_LABEL") or "Local Workstation"
    lines.append(f"| **Compute Instance** | `{inst_a}` | `{inst_b}` | {'Yes' if inst_a == inst_b else 'No'} |")
    lines.append(f"| **Platform** | `{env_a.get('platform', 'unknown')}` | `{env_b.get('platform', 'unknown')}` | {'Yes' if env_a.get('platform') == env_b.get('platform') else 'No'} |")
    lines.append(f"| **OpenCV (`cv2`)** | `{env_a.get('cv2_version', 'unknown')}` | `{env_b.get('cv2_version', 'unknown')}` | {'Yes' if env_a.get('cv2_version') == env_b.get('cv2_version') else 'No'} |")
    lines.append(f"| **NumPy** | `{env_a.get('numpy_version', 'unknown')}` | `{env_b.get('numpy_version', 'unknown')}` | {'Yes' if env_a.get('numpy_version') == env_b.get('numpy_version') else 'No'} |")
    lines.append("")
    lines.append(note_both)
    lines.append("")

    # Table 2: Latency Comparison
    lat_a = [r["latency_ms"] for r in recs_a if r["latency_ms"] is not None]
    lat_b = [r["latency_ms"] for r in recs_b if r["latency_ms"] is not None]

    lines.append("## 2. Latency Comparison")
    lines.append("")
    lines.append(f"| Latency Metric | Run A (`{path_a.name}`) | Run B (`{path_b.name}`) | Delta (B - A) | Ratio (B / A) |")
    lines.append("|:---|:---|:---|:---|:---|")

    if lat_a and lat_b:
        mean_a, mean_b = statistics.mean(lat_a), statistics.mean(lat_b)
        med_a, med_b = statistics.median(lat_a), statistics.median(lat_b)
        sorted_a = sorted(lat_a)
        sorted_b = sorted(lat_b)
        p95_a = sorted_a[max(0, math.ceil(0.95 * len(sorted_a)) - 1)]
        p95_b = sorted_b[max(0, math.ceil(0.95 * len(sorted_b)) - 1)]

        delta_mean = mean_b - mean_a
        ratio_mean = mean_b / mean_a if mean_a > 0 else 1.0

        delta_med = med_b - med_a
        ratio_med = med_b / med_a if med_a > 0 else 1.0

        delta_p95 = p95_b - p95_a
        ratio_p95 = p95_b / p95_a if p95_a > 0 else 1.0

        lines.append(f"| **Mean** | {mean_a:.1f} ms | {mean_b:.1f} ms | {delta_mean:+.1f} ms | {ratio_mean:.2f}x |")
        lines.append(f"| **Median** | {med_a:.1f} ms | {med_b:.1f} ms | {delta_med:+.1f} ms | {ratio_med:.2f}x |")
        lines.append(f"| **p95** | {p95_a:.1f} ms | {p95_b:.1f} ms | {delta_p95:+.1f} ms | {ratio_p95:.2f}x |")
    else:
        lines.append("| Latency | Insufficient data | Insufficient data | - | - |")

    lines.append("")
    lines.append(note_both)
    lines.append("")

    # Table 3: Accuracy & Coverage Comparison
    lines.append("## 3. Accuracy & Coverage Comparison")
    lines.append("")
    lines.append(f"| Metric | Run A (`{path_a.name}`) | Run B (`{path_b.name}`) | Delta (B - A) |")
    lines.append("|:---|:---|:---|:---|")

    def _get_acc_stats(recs: list[dict[str, Any]]):
        tot = len(recs)
        ev = sum(1 for r in recs if r["status"] != "error")
        acc = sum(1 for r in recs if r["status"] == "accepted")
        truth = [r for r in recs if r["status"] == "accepted" and r["true_value"] is not None and r["reading"] is not None]
        mae = statistics.mean(abs(r["reading"] - r["true_value"]) for r in truth) if truth else None
        n_within = sum(1 for r in truth if abs(r["reading"] - r["true_value"]) <= r["tolerance"])
        pct_within = (n_within / acc * 100.0) if acc > 0 else None
        cov = (acc / ev * 100.0) if ev > 0 else None
        return tot, ev, acc, cov, mae, pct_within

    tot_a, ev_a, acc_a, cov_a, mae_a, within_a = _get_acc_stats(recs_a)
    tot_b, ev_b, acc_b, cov_b, mae_b, within_b = _get_acc_stats(recs_b)

    cov_str_a = f"{cov_a:.1f}% ({acc_a}/{ev_a})" if cov_a is not None else "N/A"
    cov_str_b = f"{cov_b:.1f}% ({acc_b}/{ev_b})" if cov_b is not None else "N/A"
    cov_delta = f"{(cov_b - cov_a):+.1f}%" if (cov_a is not None and cov_b is not None) else "N/A"

    mae_str_a = f"{mae_a:.2f}" if mae_a is not None else "N/A"
    mae_str_b = f"{mae_b:.2f}" if mae_b is not None else "N/A"
    mae_delta = f"{(mae_b - mae_a):+.2f}" if (mae_a is not None and mae_b is not None) else "N/A"

    within_str_a = f"{within_a:.1f}%" if within_a is not None else "N/A"
    within_str_b = f"{within_b:.1f}%" if within_b is not None else "N/A"
    within_delta = f"{(within_b - within_a):+.1f}%" if (within_a is not None and within_b is not None) else "N/A"

    lines.append(f"| **Coverage** | {cov_str_a} | {cov_str_b} | {cov_delta} |")
    lines.append(f"| **Mean Absolute Error (MAE)** | {mae_str_a} | {mae_str_b} | {mae_delta} |")
    lines.append(f"| **Within-Tolerance Rate** | {within_str_a} | {within_str_b} | {within_delta} |")
    lines.append("")
    lines.append(note_both)
    lines.append("")

    return "\n".join(lines)


# === Legacy Dataset Evaluation Models ===


@dataclass
class EvaluationRecord:
    """Evaluation result for an individual gauge image sample."""
    filename: str
    gauge_profile: str
    ground_truth_value: float | None
    unit: str
    predicted_value: float | None = None
    absolute_error: float | None = None
    within_tolerance: bool | None = None
    status: str = "pending"
    confidence: float = 0.0
    needs_human: bool = False
    retried: bool = False
    attempts_count: int = 1
    condition: str = "unspecified"
    human_acceptable: bool | None = None
    consent_license: str = "unspecified"
    notes: str = ""
    error_reason: str = ""
    quality_flags: list[str] = field(default_factory=list)
    action_log_summary: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics for a gauge dataset."""
    total_samples: int = 0
    valid_ground_truth_count: int = 0
    missing_files_count: int = 0
    accepted_count: int = 0
    escalated_count: int = 0
    retried_count: int = 0
    tolerance_used: float = 0.0
    tolerance_fraction_used: float | None = None
    mae_accepted: float | None = None
    mae_all: float | None = None
    within_tolerance_accepted_count: int = 0
    within_tolerance_accepted_pct: float | None = None
    within_tolerance_all_count: int = 0
    within_tolerance_all_pct: float | None = None
    accepted_rate_pct: float | None = None
    escalation_rate_pct: float | None = None
    retry_rate_pct: float | None = None
    incorrect_acceptance_count: int = 0
    incorrect_acceptance_pct_of_accepted: float | None = None
    incorrect_acceptance_pct_of_total: float | None = None
    records: list[EvaluationRecord] = field(default_factory=list)
    failure_cases: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("records", None)
        return d


def load_labels_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load labels CSV file, filtering out comment lines and whitespace."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")

    samples: list[dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        return samples

    reader = csv.DictReader(lines)
    for row_idx, raw_row in enumerate(reader, start=1):
        cleaned_row = {k.strip(): (v.strip() if v is not None else "") for k, v in raw_row.items() if k}
        if not cleaned_row.get("filename"):
            continue

        gt_raw = cleaned_row.get("ground_truth_value", "")
        gt_val: float | None = None
        if gt_raw != "":
            try:
                gt_val = float(gt_raw)
            except ValueError:
                logger.warning(
                    "Row %d: could not parse ground_truth_value '%s' as float; skipping.",
                    row_idx,
                    gt_raw,
                )
                continue

        cleaned_row["ground_truth_value"] = gt_val
        cleaned_row["human_acceptable"] = parse_boolean(cleaned_row.get("human_acceptable"))
        samples.append(cleaned_row)

    return samples


def evaluate_dataset(
    labels_path: str | Path,
    images_dir: str | Path | None = None,
    default_profile: str = "EXAMPLE_270",
    tolerance: float | None = None,
    tolerance_fraction: float | None = None,
    output_dir: str | Path | None = None,
) -> EvaluationMetrics:
    """Run legacy GaugeGuard evaluation across labeled gauge images."""
    from gaugeguard.agent import inspect_upload

    labels_path = Path(labels_path)
    images_dir = Path(images_dir) if images_dir else labels_path.parent

    metrics = EvaluationMetrics()

    if not labels_path.is_file():
        logger.warning("Labels file %s does not exist.", labels_path)
        return metrics

    samples = load_labels_csv(labels_path)
    metrics.total_samples = len(samples)

    if not samples:
        logger.info("No valid labeled samples found in %s.", labels_path)
        return metrics

    try:
        def_profile_obj = get_profile(default_profile)
    except UnknownProfileError:
        def_profile_obj = None

    records: list[EvaluationRecord] = []
    failure_cases: list[dict[str, Any]] = []

    accepted_preds: list[float] = []
    accepted_gts: list[float] = []
    all_preds: list[float] = []
    all_gts: list[float] = []

    for sample in samples:
        fname = sample.get("filename", "")
        profile_name = sample.get("gauge_profile") or default_profile
        gt_value = sample.get("ground_truth_value")
        unit = sample.get("unit", "")
        condition = sample.get("condition", "unspecified")
        human_acceptable = sample.get("human_acceptable")
        consent_license = sample.get("consent_license", "unspecified")
        notes = sample.get("notes", "")

        rec = EvaluationRecord(
            filename=fname,
            gauge_profile=profile_name,
            ground_truth_value=gt_value,
            unit=unit,
            condition=condition,
            human_acceptable=human_acceptable,
            consent_license=consent_license,
            notes=notes,
        )

        if gt_value is not None:
            metrics.valid_ground_truth_count += 1

        img_path = images_dir / fname
        if not img_path.is_file():
            rec.status = "file_not_found"
            rec.error_reason = f"Image file not found at {img_path}"
            metrics.missing_files_count += 1
            records.append(rec)
            failure_cases.append(rec.to_dict())
            continue

        try:
            profile_obj = get_profile(profile_name)
        except UnknownProfileError:
            profile_obj = def_profile_obj

        try:
            image_bytes = img_path.read_bytes()
            inspection: Any = inspect_upload(image_bytes, profile=profile_obj)
        except Exception as exc:
            rec.status = "error"
            rec.error_reason = str(exc)
            records.append(rec)
            failure_cases.append(rec.to_dict())
            continue

        rec.predicted_value = inspection.reading.value
        rec.status = inspection.reading.status
        rec.confidence = inspection.reading.confidence
        rec.needs_human = inspection.needs_human
        rec.attempts_count = len(inspection.attempts)
        rec.retried = rec.attempts_count > 1
        rec.error_reason = inspection.reading.reason or ""
        rec.quality_flags = list(inspection.quality.flags) if inspection.quality else []
        rec.action_log_summary = [
            {"step": step.step_name, "method": step.method, "reason": step.reason}
            for step in inspection.action_log
        ]

        if rec.retried:
            metrics.retried_count += 1

        if rec.needs_human:
            metrics.escalated_count += 1

        full_scale = (
            (profile_obj.maximum_value - profile_obj.minimum_value)
            if profile_obj
            else 100.0
        )
        if tolerance is not None:
            sample_tol = tolerance
        elif tolerance_fraction is not None:
            sample_tol = tolerance_fraction * full_scale
        elif profile_obj and hasattr(profile_obj, "eval_tolerance_fraction"):
            sample_tol = profile_obj.eval_tolerance_fraction * full_scale
        else:
            sample_tol = 5.0

        metrics.tolerance_used = sample_tol
        if tolerance_fraction is not None:
            metrics.tolerance_fraction_used = tolerance_fraction

        if gt_value is not None and rec.predicted_value is not None:
            abs_err = abs(rec.predicted_value - gt_value)
            rec.absolute_error = round(abs_err, 4)
            rec.within_tolerance = abs_err <= sample_tol

            all_preds.append(rec.predicted_value)
            all_gts.append(gt_value)

            if rec.status == "accepted":
                metrics.accepted_count += 1
                accepted_preds.append(rec.predicted_value)
                accepted_gts.append(gt_value)

                is_incorrect_acc = (abs_err > sample_tol) or (human_acceptable is False)
                if is_incorrect_acc:
                    metrics.incorrect_acceptance_count += 1
            else:
                failure_cases.append(rec.to_dict())
        else:
            if rec.status == "accepted":
                metrics.accepted_count += 1
            failure_cases.append(rec.to_dict())

        records.append(rec)

    metrics.records = records
    metrics.failure_cases = failure_cases

    n_valid = metrics.valid_ground_truth_count
    n_acc = metrics.accepted_count

    if n_valid > 0:
        metrics.accepted_rate_pct = round((n_acc / n_valid) * 100.0, 2)
        metrics.escalation_rate_pct = round((metrics.escalated_count / n_valid) * 100.0, 2)
        metrics.retry_rate_pct = round((metrics.retried_count / n_valid) * 100.0, 2)
        metrics.incorrect_acceptance_pct_of_total = round(
            (metrics.incorrect_acceptance_count / n_valid) * 100.0, 2
        )

    if accepted_preds:
        mae_val = compute_mae(accepted_preds, accepted_gts)
        metrics.mae_accepted = round(mae_val, 4) if mae_val is not None else None
        c_in, pct_in = compute_within_tolerance(accepted_preds, accepted_gts, metrics.tolerance_used)
        metrics.within_tolerance_accepted_count = c_in
        metrics.within_tolerance_accepted_pct = round(pct_in, 2)

    if n_acc > 0:
        metrics.incorrect_acceptance_pct_of_accepted = round(
            (metrics.incorrect_acceptance_count / n_acc) * 100.0, 2
        )

    if all_preds:
        mae_all_val = compute_mae(all_preds, all_gts)
        metrics.mae_all = round(mae_all_val, 4) if mae_all_val is not None else None
        c_all_in, pct_all_in = compute_within_tolerance(all_preds, all_gts, metrics.tolerance_used)
        metrics.within_tolerance_all_count = c_all_in
        metrics.within_tolerance_all_pct = round(pct_all_in, 2)

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        summary_file = out_path / "metrics_summary.json"
        with open(summary_file, mode="w", encoding="utf-8") as f:
            json.dump(metrics.to_summary_dict(), f, indent=2)

        failures_file = out_path / "failure_cases.json"
        with open(failures_file, mode="w", encoding="utf-8") as f:
            json.dump(metrics.failure_cases, f, indent=2)

        csv_file = out_path / "per_image_results.csv"
        if records:
            fieldnames = [
                "filename",
                "gauge_profile",
                "ground_truth_value",
                "unit",
                "predicted_value",
                "absolute_error",
                "within_tolerance",
                "status",
                "confidence",
                "needs_human",
                "retried",
                "condition",
                "human_acceptable",
                "error_reason",
            ]
            with open(csv_file, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in records:
                    writer.writerow(r.to_dict())

    return metrics


def format_evaluation_report(metrics: EvaluationMetrics) -> str:
    """Format human-readable evaluation summary report."""
    lines = [
        "============================================================",
        "             GaugeGuard Evaluation Summary Report            ",
        "============================================================",
    ]

    if metrics.total_samples == 0:
        lines.append("Status: No evaluation samples found in labels file.")
        lines.append("Note: Real gauge images with verified reference readings are not yet available.")
        lines.append("Please populate data/samples/labels.csv using labels.template.csv.")
        lines.append("============================================================")
        return "\n".join(lines)

    lines.append(f"Total labeled samples:       {metrics.total_samples}")
    lines.append(f"Valid reference readings:    {metrics.valid_ground_truth_count}")
    lines.append(f"Missing image files:         {metrics.missing_files_count}")
    lines.append(f"Accepted readings:           {metrics.accepted_count} ({metrics.accepted_rate_pct or 0.0}%)")
    lines.append(f"Human-escalated readings:    {metrics.escalated_count} ({metrics.escalation_rate_pct or 0.0}%)")
    lines.append(f"Retried inspections:         {metrics.retried_count} ({metrics.retry_rate_pct or 0.0}%)")
    lines.append("------------------------------------------------------------")
    lines.append(f"Configurable tolerance:      ±{metrics.tolerance_used:.2f} units")
    if metrics.tolerance_fraction_used is not None:
        lines.append(f"Tolerance scale fraction:    {metrics.tolerance_fraction_used * 100:.1f}% of scale")

    lines.append("------------------------------------------------------------")
    lines.append("Metrics for Accepted Readings:")
    if metrics.mae_accepted is not None:
        lines.append(f"  Mean Absolute Error (MAE): {metrics.mae_accepted:.4f}")
        lines.append(
            f"  Within Tolerance Share:    {metrics.within_tolerance_accepted_pct:.2f}% "
            f"({metrics.within_tolerance_accepted_count}/{metrics.accepted_count})"
        )
    else:
        lines.append("  Mean Absolute Error (MAE): N/A (0 accepted readings)")
        lines.append("  Within Tolerance Share:    N/A")

    lines.append("------------------------------------------------------------")
    lines.append("Incorrect Acceptance Rate:")
    lines.append(
        f"  Denominator = Accepted:    {metrics.incorrect_acceptance_pct_of_accepted or 0.0}% "
        f"({metrics.incorrect_acceptance_count}/{metrics.accepted_count})"
    )
    lines.append(
        f"  Denominator = Total Valid: {metrics.incorrect_acceptance_pct_of_total or 0.0}% "
        f"({metrics.incorrect_acceptance_count}/{metrics.valid_ground_truth_count})"
    )
    lines.append("============================================================")
    return "\n".join(lines)


# === CLI ===


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entrypoint for GaugeGuard evaluation pipeline."""
    parser = argparse.ArgumentParser(
        description="GaugeGuard Evaluation Pipeline: evaluate pipeline, report, and compare runs."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run real pipeline evaluation on labelled dataset and write results.",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        metavar="RESULTS_DIR",
        help="Generate report_tables.md from an evaluation results directory.",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        default=None,
        metavar=("DIR_A", "DIR_B"),
        help="Compare two evaluation run directories (e.g. local vs EC2).",
    )
    parser.add_argument(
        "--out",
        "-o",
        dest="out",
        type=str,
        default=None,
        help="Output directory to write results (records.csv, metrics.json, environment.json, action_logs).",
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default=None,
        help="Path to CSV containing reference labels.",
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=str,
        default=None,
        help="Path to image directory.",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=None,
        help="Configurable absolute tolerance threshold.",
    )
    parser.add_argument(
        "--tolerance-fraction",
        "-tf",
        type=float,
        default=None,
        help="Configurable relative tolerance fraction.",
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        default="EXAMPLE_270",
        help="Fallback GaugeProfile name (default: EXAMPLE_270).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output evaluation summary directly as JSON.",
    )
    parser.add_argument(
        "--validate-labels",
        action="store_true",
        help="Validate labels CSV against images directory and registered profiles (no evaluation run).",
    )

    args = parser.parse_args(argv)

    if args.report:
        report_path = make_report(args.report)
        print(f"Report written to: {report_path}")
        return 0

    if args.compare:
        dir_a, dir_b = args.compare
        cmp_md = compare_runs(dir_a, dir_b)
        print(cmp_md)
        return 0

    # Resolve default paths
    if args.labels:
        labels_path = Path(args.labels)
    elif Path("sample_images/labels.csv").is_file():
        labels_path = Path("sample_images/labels.csv")
    else:
        labels_path = Path("data/samples/labels.csv")

    if args.images_dir:
        images_dir = Path(args.images_dir)
    elif Path("sample_images").is_dir():
        images_dir = Path("sample_images")
    else:
        images_dir = Path("data/samples")

    if args.validate_labels:
        return validate_labels_cli(labels_path, images_dir)

    if args.run:
        if not labels_path.is_file():
            print(f"[ERROR] Labels file not found: {labels_path}")
            return 1

        try:
            label_records = load_labels(labels_path)
        except Exception as exc:
            print(f"[ERROR] Failed to load labels file {labels_path}: {exc}")
            return 1

        if not label_records:
            print(
                f"[ERROR] Empty label set: 0 labelled images found in {labels_path}. "
                "Refusing to write metrics: no ground-truth samples available for evaluation."
            )
            return 1

        records, details = run_evaluation(labels_path, images_dir)
        agent_metrics = compute_metrics(records)
        baseline_metrics = compute_baseline_metrics(records)
        env_info = capture_environment()

        out_dir = Path(args.out) if args.out else Path("evaluation_results/local")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. records.csv
        records_csv = out_dir / "records.csv"
        fieldnames = [
            "filename",
            "condition",
            "expected_outcome",
            "true_value",
            "tolerance",
            "status",
            "reading",
            "retried",
            "attempts",
            "latency_ms",
            "first_pass_reading",
            "error_message",
            "source",
        ]
        with open(records_csv, mode="w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(r.to_dict())

        # 2. metrics.json
        metrics_json = out_dir / "metrics.json"
        metrics_payload = {
            "agent": agent_metrics,
            "baseline": baseline_metrics,
        }
        with open(metrics_json, mode="w", encoding="utf-8") as f:
            json.dump(metrics_payload, f, indent=2)

        # 3. environment.json
        env_json = out_dir / "environment.json"
        with open(env_json, mode="w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)

        # 4. per-image action logs as JSON
        action_logs_dir = out_dir / "action_logs"
        action_logs_dir.mkdir(parents=True, exist_ok=True)
        action_logs_all: dict[str, Any] = {}
        for fname, detail in details.items():
            log = detail.get("action_log", [])
            action_logs_all[fname] = log
            stem = Path(fname).stem
            with open(action_logs_dir / f"{stem}.json", mode="w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)

        with open(out_dir / "action_logs.json", mode="w", encoding="utf-8") as f:
            json.dump(action_logs_all, f, indent=2)

        print(f"Evaluation completed for {len(records)} sample(s). Results written to {out_dir}")
        print(f"- Records:     {records_csv}")
        print(f"- Metrics:     {metrics_json}")
        print(f"- Environment: {env_json}")
        print(f"- Action logs: {action_logs_dir}/ and {out_dir / 'action_logs.json'}")
        return 0

    # Fallback to evaluate_dataset if not --run
    metrics = evaluate_dataset(
        labels_path=labels_path,
        images_dir=images_dir,
        default_profile=args.profile,
        tolerance=args.tolerance,
        tolerance_fraction=args.tolerance_fraction,
        output_dir=args.out,
    )

    if args.json:
        print(json.dumps(metrics.to_summary_dict(), indent=2))
    else:
        print(format_evaluation_report(metrics))

    return 0


if __name__ == "__main__":
    sys.exit(main())
