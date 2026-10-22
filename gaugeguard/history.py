from __future__ import annotations

import csv
from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any

from gaugeguard.calibration import GaugeProfile
from gaugeguard.models import InspectionResult

HISTORY_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "filename",
    "calibration_profile",
    "estimated_reading",
    "unit",
    "angle_degrees",
    "inspection_status",
    "needs_human_review",
    "retry_count",
    "reason",
)

PROJECT_ROOT_HISTORY_FILE = Path(__file__).resolve().parent.parent / "inspection_history.csv"


def get_default_history_file() -> Path:
    """Return the active history file path.

    If GAUGEGUARD_HISTORY_DIR is set and non-empty, stores inspection_history.csv in that directory.
    Otherwise, defaults to the project root inspection_history.csv.
    """
    env_dir = os.environ.get("GAUGEGUARD_HISTORY_DIR", "").strip()
    if env_dir:
        return Path(env_dir) / "inspection_history.csv"
    return PROJECT_ROOT_HISTORY_FILE


def __getattr__(name: str) -> Any:
    """Provide dynamic backwards-compatible DEFAULT_HISTORY_FILE attribute access."""
    if name == "DEFAULT_HISTORY_FILE":
        return get_default_history_file()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


_LOGGED_KEYS: set[str] = set()


def clear_history_cache() -> None:
    """Clear the in-memory deduplication cache. Useful for testing."""
    _LOGGED_KEYS.clear()


def is_already_logged(dedup_key: str) -> bool:
    """Return True if dedup_key has already been recorded in memory."""
    return dedup_key in _LOGGED_KEYS


def count_retries(action_log: list[Any]) -> int:
    """Count retry steps in action_log using existing retry step naming prefix 'retry_'."""
    count = 0
    for step in action_log:
        step_name = getattr(step, "step_name", None)
        if step_name is None and isinstance(step, dict):
            step_name = step.get("step_name")
        if isinstance(step_name, str) and step_name.startswith("retry_"):
            count += 1
    return count


def make_dedup_key(
    result: InspectionResult,
    filename: str,
    calibration_profile: str | GaugeProfile | None = None,
    dedup_key: str | None = None,
) -> str:
    """Generate a unique key to prevent duplicate writes during Streamlit reruns."""
    if dedup_key is not None:
        return str(dedup_key)

    profile_name = getattr(calibration_profile, "name", str(calibration_profile or ""))
    reading = getattr(result, "reading", None)
    val = getattr(reading, "value", None)
    ang = getattr(reading, "angle_degrees", None)
    stat = getattr(reading, "status", None)
    return f"{id(result)}:{filename}:{profile_name}:{val}:{ang}:{stat}"


def build_history_record(
    result: InspectionResult,
    filename: str,
    calibration_profile: str | GaugeProfile | None = None,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Convert an InspectionResult and metadata into a dictionary matching HISTORY_COLUMNS.

    Handles missing values and edge cases safely without crashing.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    if hasattr(calibration_profile, "name"):
        profile_name = str(calibration_profile.name)
    elif calibration_profile is not None:
        profile_name = str(calibration_profile)
    else:
        profile_name = ""

    reading = getattr(result, "reading", None)

    # Safely extract estimated_reading
    val = getattr(reading, "value", None)
    if val is not None:
        try:
            f_val = float(val)
            reading_str = str(round(f_val, 2)) if math.isfinite(f_val) else ""
        except (ValueError, TypeError):
            reading_str = str(val)
    else:
        reading_str = ""

    # Safely extract angle_degrees
    angle = getattr(reading, "angle_degrees", None)
    if angle is not None:
        try:
            f_angle = float(angle)
            angle_str = str(round(f_angle, 2)) if math.isfinite(f_angle) else ""
        except (ValueError, TypeError):
            angle_str = str(angle)
    else:
        angle_str = ""

    unit = getattr(reading, "unit", "") or ""
    status = getattr(reading, "status", "") or ""
    reason = getattr(reading, "reason", "") or ""
    needs_human = getattr(result, "needs_human", False)
    action_log = getattr(result, "action_log", [])
    retry_count = count_retries(action_log)

    return {
        "timestamp": timestamp,
        "filename": filename or "",
        "calibration_profile": profile_name,
        "estimated_reading": reading_str,
        "unit": unit,
        "angle_degrees": angle_str,
        "inspection_status": status,
        "needs_human_review": str(bool(needs_human)),
        "retry_count": retry_count,
        "reason": reason,
    }


def append_history_record(
    record: dict[str, Any],
    csv_path: str | Path | None = None,
) -> Path:
    """Append a single record row to the CSV file.

    Writes the header only when creating a new or empty file.
    """
    path = Path(csv_path) if csv_path is not None else get_default_history_file()
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    is_empty = file_exists and path.stat().st_size == 0
    write_header = (not file_exists) or is_empty

    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(HISTORY_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(record)

    return path


def log_inspection(
    result: InspectionResult,
    filename: str,
    calibration_profile: str | GaugeProfile | None = None,
    timestamp: str | None = None,
    csv_path: str | Path | None = None,
    dedup_key: str | None = None,
    force: bool = False,
) -> bool:
    """Log an inspection to the persistent CSV history file.

    Prevents duplicate entries caused by Streamlit page reruns.
    Returns True if logged, or False if skipped as a duplicate.
    """
    key = make_dedup_key(result, filename, calibration_profile, dedup_key)
    if not force and key in _LOGGED_KEYS:
        return False

    record = build_history_record(
        result=result,
        filename=filename,
        calibration_profile=calibration_profile,
        timestamp=timestamp,
    )
    append_history_record(record, csv_path=csv_path)
    _LOGGED_KEYS.add(key)
    return True


def load_history(csv_path: str | Path | None = None) -> list[dict[str, str]]:
    """Load all inspection records from the CSV file.

    Returns a list of dictionaries matching HISTORY_COLUMNS.
    """
    path = Path(csv_path) if csv_path is not None else get_default_history_file()
    if not path.exists() or path.stat().st_size == 0:
        return []

    try:
        with open(path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception:
        return []


def get_history_csv_text(csv_path: str | Path | None = None) -> str:
    """Retrieve full CSV content as string for download."""
    path = Path(csv_path) if csv_path is not None else get_default_history_file()
    if not path.exists():
        return ",".join(HISTORY_COLUMNS) + "\n"
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ",".join(HISTORY_COLUMNS) + "\n"
