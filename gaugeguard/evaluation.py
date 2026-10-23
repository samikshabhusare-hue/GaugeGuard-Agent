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
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from gaugeguard.agent import inspect_upload
from gaugeguard.calibration import GaugeProfile, UnknownProfileError, get_profile

logger = logging.getLogger(__name__)


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
        # Original line number = header_line_num + 1 + idx
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

        # Parse true_value (can be empty for reject_input)
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

        # Validate condition
        if condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Row {orig_line_num}: Unknown condition '{condition}'. Valid: {sorted(VALID_CONDITIONS)}"
            )

        # Validate expected_outcome
        if expected_outcome not in VALID_EXPECTED_OUTCOMES:
            raise ValueError(
                f"Row {orig_line_num}: Unknown expected_outcome '{expected_outcome}'. Valid: {sorted(VALID_EXPECTED_OUTCOMES)}"
            )

        # Validate label_source not empty
        if not label_source.strip():
            raise ValueError(
                f"Row {orig_line_num}: label_source is required and cannot be empty"
            )

        record = LabelRecord(
            filename=filename.strip(),
            gauge_profile=gauge_profile.strip(),
            true_value=true_value,
            label_uncertainty=label_uncertainty,
            unit=unit.strip(),
            condition=condition,
            expected_outcome=expected_outcome,
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
        row_num = idx + 1  # Data row number (1-indexed after header)

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
                message=f"Unknown gauge profile '{rec.gauge_profile}'. Available: {', '.join(sorted(get_profile.__globals__['PROFILES'].keys()))}",
                row_number=row_num,
                filename=rec.filename,
            ))
            continue  # Skip further checks for this record if profile unknown

        profile = get_profile(rec.gauge_profile)

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
                # Check value within profile range
                if not (profile.minimum_value <= rec.true_value <= profile.maximum_value):
                    problems.append(ValidationProblem(
                        severity="error",
                        message=f"true_value {rec.true_value} outside profile range [{profile.minimum_value}, {profile.maximum_value}]",
                        row_number=row_num,
                        filename=rec.filename,
                    ))
        elif rec.expected_outcome == "reject_input":
            # true_value must be None/empty for reject_input
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

        # Check condition is valid (already validated in load_labels, but double-check)
        if rec.condition not in VALID_CONDITIONS:
            problems.append(ValidationProblem(
                severity="error",
                message=f"Invalid condition '{rec.condition}'",
                row_number=row_num,
                filename=rec.filename,
            ))

    # Check for unlabeled images in images_dir
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
    """CLI entry point for label validation.

    Returns:
        Exit code: 0 if valid (no errors), 1 if errors found.
    """
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


# === Pure Metric Functions (no fabrication — return counts always) ===


@dataclass(frozen=True)
class EvalRecord:
    """Per-evaluation record for pure metric computation.

    Tolerance per record is computed at load time from profile.eval_tolerance_fraction
    * profile scale and FIXED before running; tolerances must not be tuned after
    seeing results.
    """
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Helper for safe division with counts — never NaN, never 0 when denominator missing

def _safe_rate(numerator: int, denominator: int) -> float | None:
    """Rate = numerator / denominator; None if denominator == 0.
    Always carries counts (numerator, denominator) in the returned dict.
    """
    if denominator == 0:
        return None
    return numerator / denominator


def compute_metrics(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Compute pure metrics from EvalRecord list.

    Denominator for incorrect_acceptance_rate = accepted (documented explicitly).
    Zero denominator -> None (never 0 and never NaN). Every rate carries counts.
    Sample sizes are small; always return counts.
    """
    recs = list(records)

    n_total = len(recs)
    n_evaluable = sum(1 for r in recs if r.status != "error")
    n_accepted = sum(1 for r in recs if r.status == "accepted")
    n_needs_human = sum(1 for r in recs if r.status == "needs_human")
    n_errors = sum(1 for r in recs if r.status == "error")
    n_rejected_input = sum(1 for r in recs if r.status == "rejected_input")

    # Coverage = accepted / evaluable
    coverage = _safe_rate(n_accepted, n_evaluable)

    # MAE over accepted with true_value
    accepted_with_truth = [
        r for r in recs
        if r.status == "accepted" and r.true_value is not None and r.reading is not None
    ]
    mae = None
    if accepted_with_truth:
        mae = statistics.mean(
            abs(r.reading - r.true_value) for r in accepted_with_truth
        )

    # Within-tolerance rate = accepted & |err|<=tol / n_accepted
    n_within_tolerance = sum(
        1 for r in accepted_with_truth
        if abs(r.reading - r.true_value) <= r.tolerance
    )
    within_tolerance_rate = _safe_rate(n_within_tolerance, n_accepted)

    # Incorrect acceptance rate: denominator = accepted (documented)
    n_incorrect = sum(
        1 for r in accepted_with_truth
        if abs(r.reading - r.true_value) > r.tolerance
    )
    incorrect_acceptance_rate = _safe_rate(n_incorrect, n_accepted)

    # Retry / escalation rates over evaluable
    n_retried = sum(1 for r in recs if r.retried)
    retry_rate = _safe_rate(n_retried, n_evaluable)

    escalation_rate = _safe_rate(n_needs_human, n_evaluable)

    # Reject_input expectations reported separately
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

    # Latency summary
    latencies = [r.latency_ms for r in recs if r.latency_ms is not None]
    latency_summary: dict[str, Any] = {"n": len(latencies)}
    if latencies:
        latency_summary["mean"] = statistics.mean(latencies)
        latency_summary["median"] = statistics.median(latencies)
        # p95 via sorted list (small n OK)
        sorted_lat = sorted(latencies)
        idx_95 = math.ceil(0.95 * len(sorted_lat)) - 1
        latency_summary["p95"] = sorted_lat[max(0, idx_95)]
    if len(latencies) < 20:
        latency_summary["p95_unreliable"] = True

    # Note field for zero-denominator situations
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
        "latency_summary": latency_summary,
        "note": "; ".join(notes) if notes else "all denominators > 0",
    }


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson confidence interval for a proportion.

    Args:
        successes: Number of successes.
        n: Total trials.
        z: Z-score (default 1.96 ≈ 95%).

    Returns:
        (low, high) tuple, or None if n == 0.
    """
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
    """Calculate Mean Absolute Error between predicted and ground-truth values.

    Args:
        predictions: Sequence of estimated numeric values.
        ground_truths: Sequence of verified reference values.

    Returns:
        MAE as a float, or None if input sequences are empty or lengths mismatch.
    """
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return None

    errors = [abs(float(p) - float(g)) for p, g in zip(predictions, ground_truths)]
    return sum(errors) / len(errors)


def compute_within_tolerance(
    predictions: Sequence[float],
    ground_truths: Sequence[float],
    tolerance: float,
) -> tuple[int, float]:
    """Calculate count and percentage of predictions within an absolute tolerance.

    Args:
        predictions: Sequence of estimated numeric values.
        ground_truths: Sequence of verified reference values.
        tolerance: Non-negative tolerance threshold.

    Returns:
        Tuple of (count_within_tolerance, percentage_within_tolerance).
    """
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
        """Return high-level summary metrics dictionary without row-level records."""
        d = asdict(self)
        d.pop("records", None)
        return d


def load_labels_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """Load labels CSV file, filtering out comment lines and whitespace.

    Args:
        csv_path: Path to labels.csv file.

    Returns:
        List of parsed sample dictionaries.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Labels file not found: {path}")

    samples: list[dict[str, Any]] = []
    with open(path, mode="r", encoding="utf-8-sig") as f:
        # Filter comments and empty lines
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        return samples

    reader = csv.DictReader(lines)
    for row_idx, raw_row in enumerate(reader, start=1):
        cleaned_row = {k.strip(): (v.strip() if v is not None else "") for k, v in raw_row.items() if k}
        if not cleaned_row.get("filename"):
            continue

        # Parse ground truth value
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
    """Run GaugeGuard evaluation across labeled gauge images.

    Args:
        labels_path: Path to CSV containing labeled reference values.
        images_dir: Directory containing image files. Defaults to labels_path parent dir.
        default_profile: Fallback GaugeProfile identifier if not in row.
        tolerance: Configurable absolute tolerance (e.g. 2.0). If None, calculated from profile.
        tolerance_fraction: Configurable relative tolerance fraction (e.g. 0.05 for 5% of scale).
        output_dir: Optional directory to save metrics_summary.json, per_image_results.csv, failure_cases.json.

    Returns:
        EvaluationMetrics instance containing summary stats and per-image records.
    """
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

    # Resolve default profile for scale reference
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

        # Load profile
        try:
            profile_obj = get_profile(profile_name)
        except UnknownProfileError:
            profile_obj = def_profile_obj

        # Read image bytes and run inspection
        try:
            image_bytes = img_path.read_bytes()
            inspection: Any = inspect_upload(image_bytes, profile=profile_obj)
        except Exception as exc:
            rec.status = "error"
            rec.error_reason = str(exc)
            records.append(rec)
            failure_cases.append(rec.to_dict())
            continue

        # Extract inspection details
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

        # Determine effective tolerance for this sample
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

        # Evaluate error if ground truth and prediction exist
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

                # Check for incorrect acceptance:
                # 1. Prediction error exceeds tolerance
                # 2. Sample was flagged human_acceptable=False but system accepted it
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

    # Aggregate rates and metrics
    n_valid = metrics.valid_ground_truth_count
    n_acc = metrics.accepted_count

    if n_valid > 0:
        metrics.accepted_rate_pct = round((n_acc / n_valid) * 100.0, 2)
        metrics.escalation_rate_pct = round((metrics.escalated_count / n_valid) * 100.0, 2)
        metrics.retry_rate_pct = round((metrics.retried_count / n_valid) * 100.0, 2)
        metrics.incorrect_acceptance_pct_of_total = round(
            (metrics.incorrect_acceptance_count / n_valid) * 100.0, 2
        )

    # MAE and Within-Tolerance for strictly ACCEPTED readings
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

    # MAE and Within-Tolerance across ALL non-None readings
    if all_preds:
        mae_all_val = compute_mae(all_preds, all_gts)
        metrics.mae_all = round(mae_all_val, 4) if mae_all_val is not None else None
        c_all_in, pct_all_in = compute_within_tolerance(all_preds, all_gts, metrics.tolerance_used)
        metrics.within_tolerance_all_count = c_all_in
        metrics.within_tolerance_all_pct = round(pct_all_in, 2)

    # Save output artifacts if requested
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


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entrypoint for GaugeGuard evaluation script."""
    parser = argparse.ArgumentParser(
        description="GaugeGuard Evaluation Pipeline: compute MAE and tolerance metrics from labeled gauge data."
    )
    parser.add_argument(
        "--labels",
        "-l",
        type=str,
        default="data/samples/labels.csv",
        help="Path to CSV containing reference labels (default: data/samples/labels.csv).",
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=str,
        default="data/samples",
        help="Path to image directory (default: data/samples).",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=None,
        help="Configurable absolute tolerance threshold in engineering units (e.g. 2.0).",
    )
    parser.add_argument(
        "--tolerance-fraction",
        "-tf",
        type=float,
        default=None,
        help="Configurable relative tolerance as fraction of gauge scale (e.g. 0.05 for 5%%).",
    )
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        default="EXAMPLE_270",
        help="Fallback GaugeProfile name (default: EXAMPLE_270).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Optional directory to save evaluation artifacts (summary JSON, per-image CSV, failures JSON).",
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

    if args.validate_labels:
        return validate_labels_cli(args.labels, args.images_dir)

    metrics = evaluate_dataset(
        labels_path=args.labels,
        images_dir=args.images_dir,
        default_profile=args.profile,
        tolerance=args.tolerance,
        tolerance_fraction=args.tolerance_fraction,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(metrics.to_summary_dict(), indent=2))
    else:
        print(format_evaluation_report(metrics))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
