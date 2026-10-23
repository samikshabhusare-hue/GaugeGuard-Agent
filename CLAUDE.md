# CLAUDE.md — GaugeGuard Agent

> **Auto-updated on each commit.** Last sync: 2026-10-23 (commit 445daba)

---

## 1. Project Snapshot

| Item | Detail |
|------|--------|
| **Name** | GaugeGuard Agent |
| **Purpose** | Self-verifying computer vision for analog industrial gauge inspection |
| **Stack** | Python 3.11, OpenCV 5 (`opencv-python-headless==5.0.0.93`), NumPy 2.x, Streamlit 1.66+, pytest 8.x |
| **Entry point** | `streamlit run app.py` (port 8501) |
| **Test command** | `pytest -v` (116 tests passing) |
| **Key modules** | `gaugeguard/calibration.py`, `gaugeguard/vision.py`, `gaugeguard/agent.py`, `gaugeguard/models.py` |

---

## 2. Directory Structure (what matters)

```
GaugeGuard-Agent/
├── app.py                    # Streamlit dashboard
├── requirements.txt          # Production deps (pinned)
├── requirements-dev.txt      # Dev deps (pytest)
├── Dockerfile                # Container spec
├── CLAUDE.md                 # This file
├── CONTRACTS_DRAFT.md        # Interface contracts (draft)
├── gaugeguard/
│   ├── __init__.py
│   ├── calibration.py        # Angle→value, profiles, agreement, helpers
│   ├── vision.py             # OpenCV 5 dial/needle detection
│   ├── agent.py              # Orchestration, retry loop, escalation
│   └── models.py             # Shared frozen dataclasses
├── tests/
│   ├── test_calibration.py   # 84 tests (math, profiles, limits, helpers)
│   ├── test_agent.py         # 19 tests (orchestration, safety)
│   ├── test_vision.py        # 13 tests (detection, fixture)
│   └── fixtures/
│       └── industrial_gauge.png
├── docs/
│   ├── architecture.mmd      # Mermaid diagram
│   ├── evaluation-plan.md
│   └── report-outline.md
└── data/samples/README.md    # Image licensing guidelines
```

---

## 3. Core Conventions

### Angle Convention (CRITICAL — team-agreed)
- **0° = 12 o'clock** (straight up, negative Y in image coords)
- **Clockwise positive** (matches physical dials)
- Formula: `angle = degrees(atan2(dx, -dy)) % 360`
- Range: `[0, 360)`

### Profile Registry
- Defined in `calibration.py:PROFILES` (dict[str, GaugeProfile])
- `EXAMPLE_270` = synthetic demo profile — **never** accepted, always escalates to human
- Real profiles added via `profile_from_tick_points()` helper

### Error Handling
| Layer | Rule |
|-------|------|
| **calibration.py** | `CalibrationError` (ValueError) for invalid *profiles* (programmer error); bad measured data → flagged `CalibrationResult(value=None, ...)` |
| **vision.py** | `ValueError` for invalid image bytes; detection failure → `NeedleDetectionResult(success=False, ...)` — never fabricate angles |
| **agent.py** | Return `InspectionResult` with `status ∈ {accepted, needs_human, needs_retake, error}`; never hide failures |

### Type Safety
- All public dataclasses are `@dataclass(frozen=True)` (immutable)
- Full type annotations required
- JSON-serializable via `.to_dict()` methods (no numpy types)

---

## 4. Key Functions (cheat sheet)

### `gaugeguard.calibration`
| Function | Purpose |
|----------|---------|
| `get_profile(name)` | Retrieve registered `GaugeProfile`; raises `UnknownProfileError` |
| `validate_profile(p)` | Validate constraints; raises `CalibrationError` |
| `normalize_angle_deg(a)` | `[0, 360)` normalization; raises `TypeError`/`ValueError` |
| `pixel_to_angle_deg(center, point)` | **New** — pixel coords → angle per convention; raises `ValueError` if `point==center` |
| `profile_from_tick_points(...)` | **New** — build `GaugeProfile` from measured tick pixels; auto-calculates angles |
| `describe_profile(p)` | **New** — report-ready string summary |
| `angle_to_value(angle, profile, angle_uncertainty_deg)` | Angle → `CalibrationResult` (value, bounds, operating limits, uncertainty) |
| `readings_agree(a, b, profile)` | Agreement check → `AgreementResult(agree, diff, tolerance)` |

### `gaugeguard.vision`
| Function | Purpose |
|----------|---------|
| `decode_image(bytes)` | Bytes → BGR `np.ndarray` |
| `inspect_image_quality(bytes)` | Returns `ImageQuality(blur, contrast, flags)` |
| `preprocess_for_retry(img, method)` | `"clahe"` or `"adaptive_threshold"` |
| `detect_dial_and_needle(img)` | Returns `NeedleDetectionResult` with angle, center, confidence, flags, annotated image |

### `gaugeguard.agent`
| Function | Purpose |
|----------|---------|
| `inspect_upload(bytes, profile)` | Full agentic loop → `InspectionResult` (reading, quality, action_log, annotated_image, attempts, needs_human) |

---

## 5. Testing Rules

- Run `pytest -v` before committing
- All 116 tests must pass
- Test files mirror module names: `test_<module>.py`
- Fixtures live in `tests/fixtures/`
- New helpers need tests covering:
  - Cardinal directions (0°, 90°, 180°, 270°)
  - Degenerate cases (center==point, non-finite)
  - Synthetic hand-computed examples

---

## 6. Commit Workflow

1. Make changes
2. Run tests: `pytest -q`
3. Stage & commit:
   ```bash
   git add -A
   git commit -m "<type>(<scope>): <message>

   Co-Authored-By: Claude Code <noreply@anthropic.com>"
   ```
4. **Update CLAUDE.md** if:
   - New public functions/classes added
   - Conventions changed
   - Test count changed
   - Profile registry modified
   - Architecture decision made

---

## 7. Recent Changes (chronological)

| Commit | Date | Summary |
|--------|------|---------|
| 6a4a60e | 2026-10-22 | feat(calibration): pixel_to_angle_deg, profile_from_tick_points, describe_profile, EXAMPLE_270 untouched; 94 tests pass |
| e1bf425 | 2026-10-21 | feat(calibration): angle_to_value with bounds and dead-zone handling |
| 41ee925 | 2026-10-21 | feat(calibration): GaugeProfile, validation, angle normalization |
| a147647 | 2026-10-21 | Initialize GaugeGuard project scaffold |

---

## 8. Open Items / Next Steps

- [ ] **Fill real gauge profile** — awaiting measured data (center, min_tick, max_tick, min/max values, unit, operating limits, tolerances, image filename)
- [ ] Register real profile in `PROFILES` dict using `profile_from_tick_points()`
- [ ] Run acceptance tests: `pytest tests/test_calibration.py -q` + `python -c "import gaugeguard.calibration as c; print(c.describe_profile(c.get_profile('<name>')))"`
- [ ] Document tick-measurement error sources in report (click accuracy, image distortion, pixel quantization, center-finding error, dial printing tolerance)

---

## 9. Useful One-Liners

```bash
# Run all tests
pytest -v

# Run calibration tests only
pytest tests/test_calibration.py -q

# Quick profile check (after registration)
python -c "import gaugeguard.calibration as c; print(c.describe_profile(c.get_profile('<NAME>')))"

# Check OpenCV version
python -c "import cv2; print(cv2.__version__)"

# Start dashboard
streamlit run app.py
```

---

## 10. Memory / Reference Files

- `CONTRACTS_DRAFT.md` — Interface contracts between modules (draft, needs team sign-off)
- `docs/architecture.mmd` — Mermaid architecture diagram
- `docs/evaluation-plan.md` — Metrics & methodology
- `docs/report-outline.md` — Technical report structure
- `data/samples/README.md` — Image licensing rules