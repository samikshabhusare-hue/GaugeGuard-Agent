from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from gaugeguard.calibration import GaugeProfile
from gaugeguard.history import (
    HISTORY_COLUMNS,
    append_history_record,
    build_history_record,
    clear_history_cache,
    count_retries,
    get_history_csv_text,
    load_history,
    log_inspection,
    make_dedup_key,
)
from gaugeguard.models import AgentStep, ImageQuality, InspectionResult, ReadingEstimate


@pytest.fixture(autouse=True)
def _reset_cache():
    """Ensure in-memory history cache is cleared before and after each test."""
    clear_history_cache()
    yield
    clear_history_cache()


def _make_dummy_result(
    value: float | None = 66.73,
    unit: str | None = "psi",
    angle_degrees: float | None = 45.2,
    status: str = "accepted",
    reason: str = "Passed all checks",
    needs_human: bool = False,
    retries: int = 0,
) -> InspectionResult:
    steps: list[AgentStep] = [
        AgentStep(step_name="quality_check", method="laplacian", reason="Check blur"),
        AgentStep(step_name="initial_detection", method="hough", reason="Detect dial"),
    ]
    for i in range(1, retries + 1):
        steps.append(
            AgentStep(
                step_name=f"retry_{i}_clahe",
                method="clahe",
                reason=f"Retry pass {i}",
            )
        )
    steps.append(
        AgentStep(
            step_name="final_decision",
            method="quality_gated",
            reason="Final acceptance",
        )
    )

    reading = ReadingEstimate(
        value=value,
        unit=unit,
        angle_degrees=angle_degrees,
        confidence=0.85,
        status=status,
        reason=reason,
    )
    quality = ImageQuality(
        width=400,
        height=400,
        blur_score=120.0,
        contrast_score=45.0,
        flags=(),
    )
    return InspectionResult(
        reading=reading,
        quality=quality,
        action_log=steps,
        needs_human=needs_human,
    )


def test_count_retries():
    assert count_retries([]) == 0

    no_retry_steps = [
        AgentStep(step_name="quality_check", method="m", reason="r"),
        AgentStep(step_name="initial_detection", method="m", reason="r"),
        AgentStep(step_name="reading_comparison", method="m", reason="r"),
        AgentStep(step_name="calibration_mapping", method="m", reason="r"),
        AgentStep(step_name="final_decision", method="m", reason="r"),
    ]
    assert count_retries(no_retry_steps) == 0

    retry_steps = [
        AgentStep(step_name="quality_check", method="m", reason="r"),
        AgentStep(step_name="initial_detection", method="m", reason="r"),
        AgentStep(step_name="retry_1_clahe", method="m", reason="r"),
        AgentStep(step_name="retry_2_adaptive_threshold", method="m", reason="r"),
        AgentStep(step_name="final_decision", method="m", reason="r"),
    ]
    assert count_retries(retry_steps) == 2


def test_build_history_record_populated():
    result = _make_dummy_result(
        value=66.728,
        unit="psi",
        angle_degrees=45.234,
        status="accepted",
        reason="Reading verified",
        needs_human=False,
        retries=1,
    )
    record = build_history_record(
        result=result,
        filename="gauge_01.png",
        calibration_profile="VERIFIED_270",
        timestamp="2026-10-21T12:00:00Z",
    )

    assert set(record.keys()) == set(HISTORY_COLUMNS)
    assert record["timestamp"] == "2026-10-21T12:00:00Z"
    assert record["filename"] == "gauge_01.png"
    assert record["calibration_profile"] == "VERIFIED_270"
    assert record["estimated_reading"] == "66.73"
    assert record["unit"] == "psi"
    assert record["angle_degrees"] == "45.23"
    assert record["inspection_status"] == "accepted"
    assert record["needs_human_review"] == "False"
    assert record["retry_count"] == 1
    assert record["reason"] == "Reading verified"


def test_build_history_record_profile_object():
    profile = GaugeProfile(
        name="PROFILE_OBJ",
        unit="bar",
        minimum_value=0.0,
        maximum_value=10.0,
        minimum_angle_degrees=225.0,
        maximum_angle_degrees=135.0,
    )
    result = _make_dummy_result(value=5.0, unit="bar", angle_degrees=0.0)
    record = build_history_record(
        result=result,
        filename="test.png",
        calibration_profile=profile,
    )
    assert record["calibration_profile"] == "PROFILE_OBJ"


def test_build_history_record_missing_values_safe():
    # Reading value and angle are None, action_log is empty
    result = InspectionResult(
        reading=ReadingEstimate(
            value=None,
            unit=None,
            angle_degrees=None,
            confidence=0.0,
            status="needs_retake",
            reason="Vision detection failed",
        ),
        quality=None,
        action_log=[],
        needs_human=True,
    )

    record = build_history_record(
        result=result,
        filename="corrupt.png",
        calibration_profile=None,
    )

    assert record["filename"] == "corrupt.png"
    assert record["calibration_profile"] == ""
    assert record["estimated_reading"] == ""
    assert record["unit"] == ""
    assert record["angle_degrees"] == ""
    assert record["inspection_status"] == "needs_retake"
    assert record["needs_human_review"] == "True"
    assert record["retry_count"] == 0
    assert record["reason"] == "Vision detection failed"


def test_build_history_record_nan_and_inf_safe():
    result = InspectionResult(
        reading=ReadingEstimate(
            value=float("nan"),
            unit="units",
            angle_degrees=float("inf"),
            confidence=0.0,
            status="needs_human",
            reason="Invalid calculation",
        ),
        quality=None,
        action_log=[],
        needs_human=True,
    )
    record = build_history_record(result=result, filename="nan.png")
    assert record["estimated_reading"] == ""
    assert record["angle_degrees"] == ""


def test_append_creates_new_file_with_header(tmp_path: Path):
    csv_file = tmp_path / "inspection_history.csv"
    assert not csv_file.exists()

    result = _make_dummy_result(value=42.0)
    record = build_history_record(result, "test.jpg", "DEMO")
    append_history_record(record, csv_path=csv_file)

    assert csv_file.exists()
    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    header = lines[0].split(",")
    assert header == list(HISTORY_COLUMNS)

    data_row = lines[1].split(",")
    assert data_row[1] == "test.jpg"
    assert data_row[2] == "DEMO"
    assert data_row[3] == "42.0"


def test_append_to_existing_file_does_not_repeat_header(tmp_path: Path):
    csv_file = tmp_path / "inspection_history.csv"

    record1 = build_history_record(_make_dummy_result(value=10.0), "f1.png", "P1")
    record2 = build_history_record(_make_dummy_result(value=20.0), "f2.png", "P1")

    append_history_record(record1, csv_path=csv_file)
    append_history_record(record2, csv_path=csv_file)

    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # Header + 2 rows

    # Verify header line occurs exactly once
    header_str = ",".join(HISTORY_COLUMNS)
    assert lines[0] == header_str
    assert header_str not in lines[1]
    assert header_str not in lines[2]


def test_append_writes_header_on_empty_existing_file(tmp_path: Path):
    csv_file = tmp_path / "empty_history.csv"
    csv_file.touch()
    assert csv_file.stat().st_size == 0

    record = build_history_record(_make_dummy_result(value=15.0), "empty.png", "P")
    append_history_record(record, csv_path=csv_file)

    lines = csv_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert lines[0] == ",".join(HISTORY_COLUMNS)


def test_log_inspection_prevents_duplicate_on_rerun(tmp_path: Path):
    csv_file = tmp_path / "inspection_history.csv"
    result = _make_dummy_result(value=50.0)

    # First call simulates the initial inspection completion
    logged_first = log_inspection(
        result=result,
        filename="gauge.png",
        calibration_profile="EXAMPLE_270",
        csv_path=csv_file,
    )
    assert logged_first is True

    # Second call with the same result object simulates a Streamlit rerun
    logged_second = log_inspection(
        result=result,
        filename="gauge.png",
        calibration_profile="EXAMPLE_270",
        csv_path=csv_file,
    )
    assert logged_second is False

    # Verify only 1 row was appended
    records = load_history(csv_path=csv_file)
    assert len(records) == 1
    assert records[0]["filename"] == "gauge.png"
    assert records[0]["estimated_reading"] == "50.0"


def test_log_inspection_with_explicit_dedup_key(tmp_path: Path):
    csv_file = tmp_path / "inspection_history.csv"
    result = _make_dummy_result(value=75.0)

    logged1 = log_inspection(
        result=result,
        filename="g.png",
        calibration_profile="P",
        csv_path=csv_file,
        dedup_key="token-123",
    )
    assert logged1 is True

    logged2 = log_inspection(
        result=result,
        filename="g.png",
        calibration_profile="P",
        csv_path=csv_file,
        dedup_key="token-123",
    )
    assert logged2 is False

    # Force bypasses deduplication
    logged3 = log_inspection(
        result=result,
        filename="g.png",
        calibration_profile="P",
        csv_path=csv_file,
        dedup_key="token-123",
        force=True,
    )
    assert logged3 is True

    records = load_history(csv_path=csv_file)
    assert len(records) == 2


def test_load_history_and_get_csv_text(tmp_path: Path):
    csv_file = tmp_path / "nonexistent.csv"
    assert load_history(csv_path=csv_file) == []
    text_empty = get_history_csv_text(csv_path=csv_file)
    assert text_empty.strip() == ",".join(HISTORY_COLUMNS)

    result = _make_dummy_result(value=33.3)
    log_inspection(result, "sample.png", "TEST_PROFILE", csv_path=csv_file)

    records = load_history(csv_path=csv_file)
    assert len(records) == 1
    assert records[0]["filename"] == "sample.png"
    assert records[0]["estimated_reading"] == "33.3"

    text = get_history_csv_text(csv_path=csv_file)
    assert "sample.png" in text
    assert "33.3" in text


def test_default_history_file_without_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GAUGEGUARD_HISTORY_DIR", raising=False)
    from gaugeguard.history import (
        DEFAULT_HISTORY_FILE,
        PROJECT_ROOT_HISTORY_FILE,
        get_default_history_file,
    )
    assert get_default_history_file() == PROJECT_ROOT_HISTORY_FILE
    assert DEFAULT_HISTORY_FILE == PROJECT_ROOT_HISTORY_FILE
    assert DEFAULT_HISTORY_FILE.name == "inspection_history.csv"
    assert DEFAULT_HISTORY_FILE.parent.name == "GaugeGuard-Agent"
    assert DEFAULT_HISTORY_FILE.is_absolute()


def test_history_file_with_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    custom_dir = tmp_path / "custom_data_dir"
    monkeypatch.setenv("GAUGEGUARD_HISTORY_DIR", str(custom_dir))

    from gaugeguard.history import (
        DEFAULT_HISTORY_FILE,
        get_default_history_file,
    )
    expected_csv = custom_dir / "inspection_history.csv"
    assert get_default_history_file() == expected_csv
    assert DEFAULT_HISTORY_FILE == expected_csv

    # Verify log_inspection, load_history, and get_history_csv_text write to/read from custom_dir
    result = _make_dummy_result(value=88.5)
    logged = log_inspection(result, "custom.png", "ENV_PROFILE")
    assert logged is True
    assert expected_csv.exists()

    records = load_history()
    assert len(records) == 1
    assert records[0]["filename"] == "custom.png"
    assert records[0]["estimated_reading"] == "88.5"

    text = get_history_csv_text()
    assert "custom.png" in text
    assert "88.5" in text


def test_history_file_with_empty_or_whitespace_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GAUGEGUARD_HISTORY_DIR", "   \t\n")
    from gaugeguard.history import (
        DEFAULT_HISTORY_FILE,
        PROJECT_ROOT_HISTORY_FILE,
        get_default_history_file,
    )
    assert get_default_history_file() == PROJECT_ROOT_HISTORY_FILE
    assert DEFAULT_HISTORY_FILE == PROJECT_ROOT_HISTORY_FILE


def test_append_history_record_error_handling(tmp_path: Path):
    # Attempting to write into a directory path instead of a file should raise OSError/IOError
    dir_path = tmp_path / "a_directory"
    dir_path.mkdir()

    record = build_history_record(_make_dummy_result(), "test.png", "P")
    with pytest.raises((OSError, IOError, IsADirectoryError, PermissionError)):
        append_history_record(record, csv_path=dir_path)


def test_load_history_handles_unreadable_file_safely(tmp_path: Path):
    dir_as_file = tmp_path / "dir_pretending_file"
    dir_as_file.mkdir()
    # Reading a directory with load_history should safely return [] without unhandled crash
    assert load_history(csv_path=dir_as_file) == []
    # get_history_csv_text should safely return fallback header
    assert get_history_csv_text(csv_path=dir_as_file) == ",".join(HISTORY_COLUMNS) + "\n"

