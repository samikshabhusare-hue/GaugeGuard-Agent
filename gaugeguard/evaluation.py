"""Evaluation pipeline for GaugeGuard analog gauge readings.

Calculates Mean Absolute Error (MAE), percentage of readings within configurable tolerance,
retry rate, escalation rate, and incorrect-acceptance rate against verified ground-truth reference labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from gaugeguard.agent import inspect_upload
from gaugeguard.calibration import GaugeProfile, UnknownProfileError, get_profile

logger = logging.getLogger(__name__)


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

    args = parser.parse_args(argv)

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
