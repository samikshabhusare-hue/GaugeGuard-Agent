"""Run the GaugeGuard agent controller without starting the Streamlit UI.

Usage (from the repository root):
    python run_agent.py path/to/gauge.png
    python run_agent.py path/to/gauge.png --profile EXAMPLE_270 --json trace.json

This command exercises the agent-controller loop and writes a readable audit trace.
The EXAMPLE_270 profile is a placeholder; its reading must not be treated as verified.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2

from gaugeguard.agent import inspect_upload
from gaugeguard.calibration import PROFILES, get_profile


def _json_safe(value: Any) -> Any:
    """Convert nested dataclasses and NumPy values to JSON-compatible values."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GaugeGuard's agent controller on one gauge image.")
    parser.add_argument("image", type=Path, help="Path to a PNG or JPEG gauge image")
    parser.add_argument("--profile", default="EXAMPLE_270", choices=sorted(PROFILES), help="Registered gauge calibration profile")
    parser.add_argument("--json", dest="json_path", type=Path, help="Optional path to save the full action trace as JSON")
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"ERROR: image file not found: {args.image}", file=sys.stderr)
        return 2

    image_bytes = args.image.read_bytes()
    try:
        result = inspect_upload(image_bytes, profile=get_profile(args.profile))
    except (ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    trace = {
        "image": str(args.image),
        "profile": args.profile,
        "opencv_version": cv2.__version__,
        "reading": asdict(result.reading),
        "quality": asdict(result.quality) if result.quality else None,
        "needs_human": result.needs_human,
        "attempts": result.attempts,
        "action_log": [asdict(step) for step in result.action_log],
    }
    trace = _json_safe(trace)

    print(f"OpenCV runtime : {cv2.__version__}")
    if not cv2.__version__.startswith("5."):
        print("WARNING: this run is NOT verified against the required OpenCV 5 runtime.")
    print(f"Decision       : {result.reading.status}")
    print(f"Reading        : {result.reading.value if result.reading.value is not None else 'UNAVAILABLE'} {result.reading.unit or ''}".rstrip())
    print(f"Need human     : {result.needs_human}")
    print(f"Reason         : {result.reading.reason}")
    print(f"Attempts       : {len(result.attempts)}")
    print("\nACTION TRACE")
    for i, step in enumerate(result.action_log, start=1):
        print(f"{i}. {step.step_name} [{step.method}]")
        print(f"   reason: {step.reason}")
        print(f"   output: {json.dumps(_json_safe(step.output_summary), ensure_ascii=False)}")

    if args.json_path:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull JSON trace saved to: {args.json_path}")

    # A non-accepted decision is an inspection result, not a program crash.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
