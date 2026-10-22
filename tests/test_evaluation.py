"""Tests for GaugeGuard evaluation pipeline, MAE, tolerance metrics, and CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gaugeguard.evaluation import (
    EvaluationMetrics,
    EvaluationRecord,
    compute_mae,
    compute_within_tolerance,
    evaluate_dataset,
    format_evaluation_report,
    load_labels_csv,
    main,
    parse_boolean,
)
from gaugeguard.models import AgentStep, ImageQuality, InspectionResult, ReadingEstimate


def test_compute_mae_known_values():
    """Verify MAE calculation on known scalar values."""
    preds = [10.0, 20.0, 30.0]
    gts = [12.0, 18.0, 31.0]
    # Errors: |10-12|=2.0, |20-18|=2.0, |30-31|=1.0 -> Sum=5.0 -> MAE=5/3
    mae = compute_mae(preds, gts)
    assert mae is not None
    assert abs(mae - (5.0 / 3.0)) < 1e-6

    # Zero error
    assert compute_mae([42.5], [42.5]) == 0.0


def test_compute_mae_edge_cases():
    """Verify MAE handles empty lists and mismatched inputs cleanly."""
    assert compute_mae([], []) is None
    assert compute_mae([1.0], []) is None
    assert compute_mae([], [1.0]) is None
    assert compute_mae([1.0, 2.0], [1.0]) is None


def test_compute_within_tolerance_boundaries():
    """Verify tolerance count and percentage calculation across thresholds."""
    preds = [10.0, 20.0, 30.0]
    gts = [10.0, 21.0, 33.0]
    # Errors: 0.0, 1.0, 3.0

    # Tolerance 1.0: 0.0 and 1.0 pass (2 / 3 -> 66.67%)
    count, pct = compute_within_tolerance(preds, gts, 1.0)
    assert count == 2
    assert abs(pct - 66.6666666) < 1e-4

    # Tolerance 0.0: only exact match 10.0 passes (1 / 3 -> 33.33%)
    count, pct = compute_within_tolerance(preds, gts, 0.0)
    assert count == 1
    assert abs(pct - 33.3333333) < 1e-4

    # Tolerance 5.0: all pass (3 / 3 -> 100.0%)
    count, pct = compute_within_tolerance(preds, gts, 5.0)
    assert count == 3
    assert pct == 100.0

    # Negative tolerance raises ValueError
    with pytest.raises(ValueError):
        compute_within_tolerance(preds, gts, -0.5)

    # Empty inputs return 0, 0.0
    assert compute_within_tolerance([], [], 1.0) == (0, 0.0)


def test_parse_boolean():
    """Verify boolean parser parses string values flexibly."""
    assert parse_boolean(True) is True
    assert parse_boolean("True") is True
    assert parse_boolean("yes") is True
    assert parse_boolean("1") is True
    assert parse_boolean(False) is False
    assert parse_boolean("FALSE") is False
    assert parse_boolean("0") is False
    assert parse_boolean("no") is False
    assert parse_boolean("") is None
    assert parse_boolean(None) is None


def test_load_labels_csv_with_comments_and_whitespace(tmp_path: Path):
    """Verify CSV loading filters comment lines and trims whitespace."""
    csv_file = tmp_path / "test_labels.csv"
    csv_file.write_text(
        "# Header comment\n"
        "# Another comment line\n"
        "filename, gauge_profile, ground_truth_value, unit, condition, human_acceptable, consent_license, notes\n"
        "sample1.jpg, EXAMPLE_270, 45.2, PSI, clear, true, CC0, good lighting\n"
        "# Mid-file comment\n"
        "sample2.jpg, EXAMPLE_270, 80.0, PSI, blurry, false, owner_consent, out of focus\n"
        "sample3.jpg, EXAMPLE_270, invalid_num, PSI, clear, true, CC0, bad number\n",
        encoding="utf-8",
    )

    samples = load_labels_csv(csv_file)
    assert len(samples) == 2  # sample3 has invalid float, should be skipped
    assert samples[0]["filename"] == "sample1.jpg"
    assert samples[0]["ground_truth_value"] == 45.2
    assert samples[0]["human_acceptable"] is True
    assert samples[0]["consent_license"] == "CC0"
    assert samples[1]["filename"] == "sample2.jpg"
    assert samples[1]["ground_truth_value"] == 80.0
    assert samples[1]["human_acceptable"] is False


def test_load_labels_csv_missing_file():
    """Verify FileNotFoundError when labels file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_labels_csv("nonexistent_path/labels.csv")


def test_evaluate_dataset_empty_template(tmp_path: Path):
    """Verify evaluation on an empty or template-only labels file exits gracefully."""
    empty_csv = tmp_path / "empty_labels.csv"
    empty_csv.write_text(
        "filename,gauge_profile,ground_truth_value,unit,condition,human_acceptable,consent_license,notes\n",
        encoding="utf-8",
    )

    metrics = evaluate_dataset(labels_path=empty_csv, images_dir=tmp_path)
    assert metrics.total_samples == 0
    assert metrics.mae_accepted is None
    assert metrics.within_tolerance_accepted_pct is None

    report = format_evaluation_report(metrics)
    assert "No evaluation samples found" in report


def test_evaluate_dataset_missing_image_files(tmp_path: Path):
    """Verify evaluation handles referenced images that do not exist on disk."""
    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,ground_truth_value,unit\n"
        "missing_image.png,EXAMPLE_270,50.0,units\n",
        encoding="utf-8",
    )

    metrics = evaluate_dataset(labels_path=csv_file, images_dir=tmp_path)
    assert metrics.total_samples == 1
    assert metrics.missing_files_count == 1
    assert metrics.accepted_count == 0
    assert metrics.records[0].status == "file_not_found"


def test_evaluate_dataset_with_mocked_inspection(tmp_path: Path):
    """Verify full evaluation calculation and metric aggregation with mocked agent."""
    # Create dummy image files
    img1 = tmp_path / "gauge1.png"
    img1.write_bytes(b"dummy_bytes_1")
    img2 = tmp_path / "gauge2.png"
    img2.write_bytes(b"dummy_bytes_2")
    img3 = tmp_path / "gauge3.png"
    img3.write_bytes(b"dummy_bytes_3")

    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,ground_truth_value,unit,human_acceptable\n"
        "gauge1.png,EXAMPLE_270,50.0,units,true\n"
        "gauge2.png,EXAMPLE_270,75.0,units,true\n"
        "gauge3.png,EXAMPLE_270,20.0,units,false\n",
        encoding="utf-8",
    )

    # Mock inspect_upload responses:
    # 1. gauge1: accepted, pred=51.5 (error=1.5, within tol=2.0)
    # 2. gauge2: accepted, pred=78.5 (error=3.5, outside tol=2.0 -> incorrect acceptance!)
    # 3. gauge3: needs_human, pred=20.5 (escalated, retry=2)
    def fake_inspect(image_bytes: bytes, profile=None):
        if image_bytes == b"dummy_bytes_1":
            return InspectionResult(
                reading=ReadingEstimate(
                    value=51.5, unit="units", angle_degrees=100.0, confidence=0.85, status="accepted", reason="ok"
                ),
                quality=ImageQuality(width=100, height=100, blur_score=150.0, contrast_score=60.0),
                action_log=[AgentStep(step_name="detect", method="default", reason="pass")],
                attempts=[{"attempt": 1, "success": True}],
                needs_human=False,
            )
        elif image_bytes == b"dummy_bytes_2":
            return InspectionResult(
                reading=ReadingEstimate(
                    value=78.5, unit="units", angle_degrees=180.0, confidence=0.75, status="accepted", reason="ok"
                ),
                quality=ImageQuality(width=100, height=100, blur_score=120.0, contrast_score=50.0),
                action_log=[AgentStep(step_name="detect", method="default", reason="pass")],
                attempts=[{"attempt": 1, "success": True}],
                needs_human=False,
            )
        else:
            return InspectionResult(
                reading=ReadingEstimate(
                    value=20.5, unit="units", angle_degrees=230.0, confidence=0.45, status="needs_human", reason="low confidence"
                ),
                quality=ImageQuality(width=100, height=100, blur_score=40.0, contrast_score=20.0),
                action_log=[
                    AgentStep(step_name="detect_1", method="default", reason="fail"),
                    AgentStep(step_name="retry_clahe", method="clahe", reason="retry"),
                ],
                attempts=[{"attempt": 1, "success": False}, {"attempt": 2, "success": True}],
                needs_human=True,
            )

    out_dir = tmp_path / "results"
    with patch("gaugeguard.evaluation.inspect_upload", side_effect=fake_inspect):
        metrics = evaluate_dataset(
            labels_path=csv_file,
            images_dir=tmp_path,
            tolerance=2.0,
            output_dir=out_dir,
        )

    assert metrics.total_samples == 3
    assert metrics.valid_ground_truth_count == 3
    assert metrics.accepted_count == 2
    assert metrics.escalated_count == 1
    assert metrics.retried_count == 1

    # Accepted readings: [51.5, 78.5] vs GT [50.0, 75.0] -> errors: 1.5, 3.5 -> MAE = 2.5
    assert metrics.mae_accepted == 2.5
    # Within tolerance (tol=2.0): gauge1 is 1.5 (<=2.0), gauge2 is 3.5 (>2.0) -> 1 of 2 (50.0%)
    assert metrics.within_tolerance_accepted_count == 1
    assert metrics.within_tolerance_accepted_pct == 50.0

    # Incorrect acceptance: gauge2 exceeded tolerance (3.5 > 2.0) -> 1 incorrect acceptance
    assert metrics.incorrect_acceptance_count == 1
    assert metrics.incorrect_acceptance_pct_of_accepted == 50.0

    # Verify generated output files
    assert (out_dir / "metrics_summary.json").is_file()
    assert (out_dir / "failure_cases.json").is_file()
    assert (out_dir / "per_image_results.csv").is_file()

    summary_json = json.loads((out_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    assert summary_json["total_samples"] == 3
    assert summary_json["mae_accepted"] == 2.5

    report = format_evaluation_report(metrics)
    assert "Mean Absolute Error (MAE): 2.5000" in report
    assert "Within Tolerance Share:    50.00%" in report


def test_cli_execution_template_file():
    """Verify CLI executes cleanly against repo template labels file."""
    labels_file = Path(__file__).parent.parent / "data" / "samples" / "labels.csv"
    assert labels_file.is_file()

    # Run CLI with template file
    exit_code = main(["--labels", str(labels_file), "--json"])
    assert exit_code == 0
