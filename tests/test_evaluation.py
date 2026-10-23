"""Tests for GaugeGuard evaluation pipeline, MAE, tolerance metrics, and CLI."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gaugeguard.evaluation import (
    EvalRecord,
    EvaluationMetrics,
    EvaluationRecord,
    LabelRecord,
    StubPipeline,
    ValidationProblem,
    capture_environment,
    compute_baseline_metrics,
    compute_mae,
    compute_within_tolerance,
    evaluate_dataset,
    format_evaluation_report,
    load_image_bgr,
    load_labels,
    main,
    parse_boolean,
    run_evaluation,
    run_pipeline,
    validate_labels,
    validate_labels_cli,
)
from gaugeguard.models import AgentStep, ImageQuality, InspectionResult, ReadingEstimate


# === load_labels tests ===

def test_load_labels_csv_with_comments_and_whitespace(tmp_path: Path):
    """Verify CSV loading filters comment lines and trims whitespace."""
    csv_file = tmp_path / "test_labels.csv"
    csv_file.write_text(
        "# Header comment\n"
        "# Another comment line\n"
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "sample1.jpg,EXAMPLE_270,45.2,0.5,PSI,clear,read,known setpoint,good lighting\n"
        "# Mid-file comment\n"
        "sample2.jpg,EXAMPLE_270,80.0,1.0,PSI,blurry,reject_input,two independent human reads,A: 78.0 B: 80.0 PSI\n"
        "sample3.jpg,EXAMPLE_270,,0.5,PSI,clear,read,known setpoint,bad number\n",
        encoding="utf-8",
    )

    samples = load_labels(csv_file)
    assert len(samples) == 3
    assert samples[0].filename == "sample1.jpg"
    assert samples[0].true_value == 45.2
    assert samples[0].label_uncertainty == 0.5
    assert samples[0].unit == "PSI"
    assert samples[0].condition == "clear"
    assert samples[0].expected_outcome == "read"


def test_load_labels_csv_missing_file():
    """Verify FileNotFoundError when labels file does not exist."""
    with pytest.raises(FileNotFoundError):
        load_labels("nonexistent_path/labels.csv")


# === validation tests ===

def test_validate_labels_header_only(tmp_path: Path):
    """Verify header-only CSV reports '0 labelled images' clearly."""
    csv_file = tmp_path / "empty_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n",
        encoding="utf-8",
    )

    exit_code = validate_labels_cli(csv_file, tmp_path)
    assert exit_code == 0


def test_validate_labels_duplicate_filenames(tmp_path: Path):
    """Verify duplicate filenames are caught."""
    (tmp_path / "gauge1.png").write_bytes(b"dummy")

    csv_file = tmp_path / "dup_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,50.0,0.5,units,clear,read,known setpoint,\n"
        "gauge1.png,EXAMPLE_270,75.0,1.0,units,blurry,reject_input,two independent reads,\n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    dup_problems = [p for p in problems if "Duplicate" in p.message]
    assert len(dup_problems) >= 1


def test_validate_labels_missing_file(tmp_path: Path):
    """Verify missing image files are caught."""
    csv_file = tmp_path / "missing_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "missing_image.png,EXAMPLE_270,50.0,0.5,PSI,clear,read,known setpoint,, \n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    missing_problems = [p for p in problems if "not found" in p.message.lower()]
    assert len(missing_problems) >= 1


def test_validate_labels_unknown_profile(tmp_path: Path):
    """Verify unknown gauge profile is caught."""
    csv_file = tmp_path / "unknown_profile.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,UNKNOWN_PROFILE,50.0,0.5,PSI,clear,read,known setpoint,, \n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    unknown_problems = [p for p in problems if "Unknown profile" in p.message]
    assert len(unknown_problems) >= 1


def test_validate_labels_true_value_missing_for_read(tmp_path: Path):
    """Verify true_value missing for 'read' cases."""
    csv_file = tmp_path / "missing_gt.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,,0.5,PSI,clear,read,known setpoint,, \n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    missing_gt_problems = [p for p in problems if "true_value is required" in p.message]
    assert len(missing_gt_problems) >= 1


def test_validate_labels_value_outside_profile_range(tmp_path: Path):
    """Verify true_value outside profile range is caught."""
    csv_file = tmp_path / "out_of_range.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,200.0,0.5,PSI,clear,read,known setpoint,, \n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    range_problems = [p for p in problems if "outside profile range" in p.message]
    assert len(range_problems) >= 1


def test_validate_labels_missing_label_source(tmp_path: Path):
    """Verify missing label_source is caught."""
    csv_file = tmp_path / "missing_source.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,50.0,0.5,PSI,clear,read,,\n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    missing_source_problems = [p for p in problems if "label_source is required" in p.message]
    assert len(missing_source_problems) >= 1


def test_validate_labels_unknown_condition(tmp_path: Path):
    """Verify unknown condition is caught."""
    csv_file = tmp_path / "unknown_condition.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,50.0,0.5,PSI,unknown_condition,read,known setpoint,, \n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    unknown_cond_problems = [p for p in problems if "Unknown condition" in p.message]
    assert len(unknown_cond_problems) >= 1


def test_validate_labels_reject_input_empty_true_value(tmp_path: Path):
    """Verify reject_input has empty true_value."""
    csv_file = tmp_path / "reject_no_gt.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270, ,0.5,PSI,blurry,reject_input,two independent reads,unreadable\n",
        encoding="utf-8",
    )

    records = load_labels(csv_file)
    problems = validate_labels(records, tmp_path)
    errors = [p for p in problems if p.severity == "error"]
    reject_errors = [p for p in errors if "true_value must be empty" in p.message]
    assert len(reject_errors) == 0


def test_validate_labels_cli_valid(tmp_path: Path):
    """Verify CLI validates valid labels cleanly."""
    csv_file = tmp_path / "valid_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,50.0,0.5,units,clear,read,known setpoint,Druck calibrator,\n",
        encoding="utf-8",
    )
    (tmp_path / "gauge1.png").write_bytes(b"dummy")

    exit_code = validate_labels_cli(csv_file, tmp_path)
    assert exit_code == 0


def test_validate_labels_cli_errors(tmp_path: Path):
    """Verify CLI reports errors."""
    csv_file = tmp_path / "error_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,UNKNOWN,200.0,0.5,PSI,clear,read,known setpoint,, \n",
        encoding="utf-8",
    )

    exit_code = validate_labels_cli(csv_file, tmp_path)
    assert exit_code == 1


def test_evaluate_dataset_template(tmp_path: Path):
    """Verify evaluation on empty template exits gracefully."""
    csv_file = tmp_path / "empty_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n",
        encoding="utf-8",
    )

    metrics = evaluate_dataset(labels_path=csv_file, images_dir=tmp_path)
    assert metrics.total_samples == 0
    assert metrics.mae_accepted is None

    report = format_evaluation_report(metrics)
    assert "No evaluation samples found" in report


def test_evaluate_dataset_missing_images(tmp_path: Path):
    """Verify evaluation handles missing images."""
    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "missing.png,EXAMPLE_270,50.0,0.5,PSI,clear,read,known setpoint,, \n",
        encoding="utf-8",
    )

    (tmp_path / "other.png").write_bytes(b"dummy")

    metrics = evaluate_dataset(labels_path=csv_file, images_dir=tmp_path)
    assert metrics.missing_files_count >= 1


def test_cli_execution_template_file():
    """Verify CLI executes cleanly against repo template labels file."""
    labels_file = Path(__file__).parent.parent / "data" / "samples" / "labels.csv"
    assert labels_file.is_file()

    exit_code = main(["--labels", str(labels_file), "--json"])
    assert exit_code == 0


def test_cli_execution_validate_labels(tmp_path: Path):
    """Verify CLI --validate-labels flag works."""
    csv_file = tmp_path / "valid_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "test.png,EXAMPLE_270,50.0,0.5,units,clear,read,known setpoint,Druck calibrator,\n",
        encoding="utf-8",
    )
    (tmp_path / "test.png").write_bytes(b"dummy")

    exit_code = main(["--labels", str(csv_file), "--images-dir", str(tmp_path), "--validate-labels"])
    assert exit_code == 0


# === LabelRecord tests ===

def test_label_record_creation():
    """Verify LabelRecord can be created with all fields."""
    rec = LabelRecord(
        filename="test.jpg",
        gauge_profile="EXAMPLE_270",
        true_value=50.0,
        label_uncertainty=0.5,
        unit="PSI",
        condition="clear",
        expected_outcome="read",
        label_source="known setpoint",
        notes="test note",
    )
    assert rec.filename == "test.jpg"
    assert rec.true_value == 50.0
    assert rec.label_uncertainty == 0.5
    assert rec.unit == "PSI"
    assert rec.condition == "clear"
    assert rec.expected_outcome == "read"
    assert rec.label_source == "known setpoint"
    assert rec.notes == "test note"

    d = rec.to_dict()
    assert d["filename"] == "test.jpg"
    assert d["true_value"] == 50.0


def test_label_record_empty_true_value_reject():
    """Verify LabelRecord with empty true_value for reject_input."""
    rec = LabelRecord(
        filename="test.jpg",
        gauge_profile="EXAMPLE_270",
        true_value=None,
        label_uncertainty=0.5,
        unit="PSI",
        condition="blurry",
        expected_outcome="reject_input",
        label_source="two independent human reads",
        notes="both operators could not read needle",
    )
    assert rec.true_value is None
    assert rec.expected_outcome == "reject_input"


# === Runner and Pipeline Adapter Tests ===


def _create_sample_image(path: Path) -> None:
    """Create a minimal valid PNG image on disk."""
    import cv2
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, (255, 255, 255), -1)
    cv2.imwrite(str(path), img)


def test_load_image_bgr_corrupt_returns_none(tmp_path: Path):
    """Verify load_image_bgr returns None on corrupt files without crashing."""
    corrupt_file = tmp_path / "corrupt.jpg"
    corrupt_file.write_bytes(b"garbage byte sequence not an image")

    img = load_image_bgr(corrupt_file)
    assert img is None

    nonexistent_file = tmp_path / "does_not_exist.png"
    assert load_image_bgr(nonexistent_file) is None


def test_run_evaluation_with_stub_pipeline(tmp_path: Path):
    """Verify run_evaluation with fake injected StubPipeline."""
    img_path = tmp_path / "gauge.png"
    _create_sample_image(img_path)

    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge.png,EXAMPLE_270,50.0,0.5,units,clear,read,calibrator,clean\n",
        encoding="utf-8",
    )

    stub = StubPipeline(
        status="accepted",
        reading=49.5,
        retried=True,
        attempts=2,
        first_pass_reading=48.0,
        action_log=[{"step_name": "step1"}],
        reason="ok",
    )

    records, details = run_evaluation(csv_file, tmp_path, pipeline=stub)
    assert len(records) == 1
    rec = records[0]
    assert rec.filename == "gauge.png"
    assert rec.status == "accepted"
    assert rec.reading == 49.5
    assert rec.retried is True
    assert rec.attempts == 2
    assert rec.first_pass_reading == 48.0
    assert rec.source == "agent"
    assert rec.latency_ms is not None
    assert rec.latency_ms >= 0

    assert "gauge.png" in details
    assert details["gauge.png"]["status"] == "accepted"
    assert len(details["gauge.png"]["action_log"]) == 1


def test_run_evaluation_pipeline_exception_becomes_error(tmp_path: Path):
    """Verify exceptions from pipeline become status 'error' flagged source='harness'."""
    img_path = tmp_path / "gauge.png"
    _create_sample_image(img_path)

    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge.png,EXAMPLE_270,50.0,0.5,units,clear,read,calibrator,clean\n",
        encoding="utf-8",
    )

    crashing_pipeline = StubPipeline(raise_exception=RuntimeError("Agent pipeline crash"))

    records, details = run_evaluation(csv_file, tmp_path, pipeline=crashing_pipeline)
    assert len(records) == 1
    rec = records[0]
    assert rec.status == "error"
    assert rec.source == "harness"
    assert "Agent pipeline crash" in rec.error_message
    assert details["gauge.png"]["status"] == "error"


def test_run_evaluation_corrupted_fixture_becomes_rejected_input(tmp_path: Path):
    """Verify corrupted fixture file becomes harness 'rejected_input' with source='harness'."""
    corrupt_img = tmp_path / "bad.jpg"
    corrupt_img.write_bytes(b"\x00\x01\x02\x03garbage\xff\xfe")

    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "bad.jpg,EXAMPLE_270,50.0,0.5,units,blurry,reject_input,human,unreadable\n",
        encoding="utf-8",
    )

    stub = StubPipeline()
    records, details = run_evaluation(csv_file, tmp_path, pipeline=stub)

    assert len(records) == 1
    rec = records[0]
    assert rec.status == "rejected_input"
    assert rec.source == "harness"
    assert rec.reading is None
    assert rec.first_pass_reading is None
    assert details["bad.jpg"]["status"] == "rejected_input"
    assert details["bad.jpg"]["source"] == "harness"


def test_capture_environment():
    """Verify capture_environment captures python, platform, cv2, and numpy versions."""
    env = capture_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "cv2_version" in env
    assert "numpy_version" in env

    # Check warning if cv2 is 4.x
    import cv2
    cv2_ver = getattr(cv2, "__version__", "")
    if cv2_ver.startswith("4."):
        assert "warnings" in env
        assert "OpenCV 5 condition not met" in env["warnings"]

    # Test env vars capture
    with patch.dict(os.environ, {"GAUGEGUARD_RUN_LABEL": "test-run", "EC2_INSTANCE_TYPE": "t3.medium"}):
        env_with_vars = capture_environment()
        assert env_with_vars["GAUGEGUARD_RUN_LABEL"] == "test-run"
        assert env_with_vars["EC2_INSTANCE_TYPE"] == "t3.medium"


def test_baseline_metrics_computation():
    """Verify baseline first-pass-only metrics computation and 'not available' fallback."""
    records_with_first_pass = [
        EvalRecord(
            filename="g1.png",
            condition="clear",
            expected_outcome="read",
            true_value=50.0,
            tolerance=2.5,
            status="needs_human",
            reading=53.0,
            retried=True,
            attempts=2,
            latency_ms=100.0,
            first_pass_reading=51.0,
            source="agent",
        )
    ]
    base = compute_baseline_metrics(records_with_first_pass)
    assert isinstance(base, dict)
    assert base["n_accepted"] == 1  # accepted without quality gate
    assert base["mae"][0] == 1.0  # |51.0 - 50.0| = 1.0

    records_without_first_pass = [
        EvalRecord(
            filename="g2.png",
            condition="clear",
            expected_outcome="read",
            true_value=50.0,
            tolerance=2.5,
            status="error",
            reading=None,
            retried=False,
            attempts=0,
            latency_ms=10.0,
            first_pass_reading=None,
            source="harness",
        )
    ]
    base_na = compute_baseline_metrics(records_without_first_pass)
    assert base_na == "not available"


def test_cli_run_writes_all_files(tmp_path: Path):
    """Verify CLI --run writes records.csv, metrics.json, environment.json, and action logs."""
    img_path = tmp_path / "img1.png"
    _create_sample_image(img_path)

    csv_file = tmp_path / "labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "img1.png,EXAMPLE_270,50.0,0.5,units,clear,read,setpoint,notes\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "results"

    exit_code = main([
        "--run",
        "--labels", str(csv_file),
        "--images-dir", str(tmp_path),
        "--out", str(out_dir),
    ])
    assert exit_code == 0

    # Verify created files
    records_file = out_dir / "records.csv"
    metrics_file = out_dir / "metrics.json"
    env_file = out_dir / "environment.json"
    action_logs_file = out_dir / "action_logs.json"
    action_log_img = out_dir / "action_logs" / "img1.json"

    assert records_file.is_file()
    assert metrics_file.is_file()
    assert env_file.is_file()
    assert action_logs_file.is_file()
    assert action_log_img.is_file()

    # Validate JSON formats
    metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert "agent" in metrics_data
    assert "baseline" in metrics_data

    env_data = json.loads(env_file.read_text(encoding="utf-8"))
    assert "python_version" in env_data
    assert "cv2_version" in env_data


def test_cli_run_missing_labels_and_empty_label_set(tmp_path: Path, capsys):
    """Verify CLI handles missing labels and empty label set gracefully."""
    # Missing labels file
    exit_code_missing = main(["--run", "--labels", str(tmp_path / "nonexistent.csv")])
    assert exit_code_missing == 1
    captured_missing = capsys.readouterr()
    assert "Labels file not found" in captured_missing.out

    # Empty label set
    empty_csv = tmp_path / "empty_labels.csv"
    empty_csv.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n",
        encoding="utf-8",
    )
    exit_code_empty = main(["--run", "--labels", str(empty_csv), "--images-dir", str(tmp_path)])
    assert exit_code_empty == 1
    captured_empty = capsys.readouterr()
    assert "Empty label set" in captured_empty.out
    assert "Refusing to write metrics" in captured_empty.out
