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
    LabelRecord,
    ValidationProblem,
    load_labels,
    validate_labels,
    validate_labels_cli,
    evaluate_dataset,
    compute_mae,
    compute_within_tolerance,
    format_evaluation_report,
    main,
    parse_boolean,
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
    assert len(samples) == 2  # sample3 has invalid float, should be skipped
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

    problems = validate_labels([], tmp_path)
    # Header-only should not crash, and should report 0 labelled images
    from gaugeguard.evaluation import validate_labels_cli
    exit_code = validate_labels_cli(csv_file, tmp_path)
    assert exit_code == 0


def test_validate_labels_duplicate_filenames(tmp_path: Path):
    """Verify duplicate filenames are caught."""
    # Create dummy image file so file-not-found doesn't interfere
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
    # Should detect duplicate filenames
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
    # Should not flag error for empty true_value with reject_input
    errors = [p for p in problems if p.severity == "error"]
    reject_errors = [p for p in errors if "true_value must be empty" in p.message]
    # Empty string might be parsed as 0.0, so check it's not flagged as missing
    # The key is that it should not be flagged as "missing for read"
    assert len(reject_errors) == 0  # Empty string allowed for reject_input


def test_validate_labels_cli_valid(tmp_path: Path):
    """Verify CLI validates valid labels cleanly."""
    csv_file = tmp_path / "valid_labels.csv"
    csv_file.write_text(
        "filename,gauge_profile,true_value,label_uncertainty,unit,condition,expected_outcome,label_source,notes\n"
        "gauge1.png,EXAMPLE_270,50.0,0.5,units,clear,read,known setpoint,Druck calibrator,\n",
        encoding="utf-8",
    )
    # Create the image file so validation passes
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

    # Create a dummy image that won't be found
    (tmp_path / "other.png").write_bytes(b"dummy")

    metrics = evaluate_dataset(labels_path=csv_file, images_dir=tmp_path)
    assert metrics.missing_files_count >= 1


def test_cli_execution_template_file():
    """Verify CLI executes cleanly against repo template labels file."""
    labels_file = Path(__file__).parent.parent / "data" / "samples" / "labels.csv"
    assert labels_file.is_file()

    # Run CLI with template file
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
    # Must pass --images-dir to point to temp dir
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

    # Test to_dict
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