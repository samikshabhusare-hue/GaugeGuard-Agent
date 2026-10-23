# Scenario Test Report

## Evidence: pytest output — `pytest tests/test_scenarios.py -q -rs`

```
Fssss FF......                                                           [100%]
3 failed, 6 passed, 4 skipped in 1.86s
```

---

## Passes (6)

| Group | Test | Result |
|-------|------|--------|
| group7 | `TestUncertainReading::test_uncertainty_range_is_positive` | PASSED |
| group8 | `TestDeadZone::test_dead_zone_angle_returns_none_and_in_bounds_false` | PASSED |
| group8 | `TestDeadZone::test_agent_never_accepts_dead_zone` | PASSED |
| group9 | `TestDisagreement::test_readings_agree_false_for_hand_built_numbers` | PASSED |
| group9 | `TestDisagreement::test_agent_does_not_silently_accept_when_readings_disagree` | PASSED |
| group10 | `TestHumanVerification::test_needs_human_true_with_non_empty_reason_on_ambiguous_needle` | PASSED |

---

## Failures (3) — all owned by `agent.py / Kshitj`

Assertions are **not weakened**. Tests are kept strictly failing with clear owner attribution.

### group1 — Scenario 1: Clear Image — `accepted` status

**Test:** `TestClearImage::test_clear_image_reading_exists_and_near_ground_truth`

**Root cause:** `agent.py::inspect_upload` detects the EXAMPLE_270 profile name and appends
`unverified_demo_profile` to `all_flags`. The presence of **any flag** causes the agent to
escalate to `needs_human` instead of `accepted` — even when the reading is numeric and within
label tolerance.

**Owner:** `agent.py / Kshitj` — agent must not escalate a valid clear reading to `needs_human`
solely because the profile name starts with "EXAMPLE". Either remove the demo-profile flag from
the escalation path or allow the label-verified scenario to pass through.

**Image:** `sample_images/industrial_gauge.png` (EXAMPLE_270, true_value=66.73, label_uncertainty=0.74, condition=clear)

---

### group6a — Scenario 6: Corrupted file causes crash

**Test:** `TestCorruptedAndUnsupported::test_corrupted_bytes_return_rejected_without_crash`

**Root cause:** `agent.py::inspect_upload` calls `inspect_image_quality(image_bytes)` which
calls `decode_image(image_bytes)`, which raises `ValueError` for corrupt bytes. The exception
propagates uncaught out of `inspect_upload`, crashing instead of returning `rejected_input`.

**Owner:** `agent.py / Kshitj` — `inspect_upload` must wrap the quality/decode step in a
try/except and return `InspectionResult(reading=ReadingEstimate(status="rejected_input", ...))`.

---

### group6b — Scenario 6: Unsupported file (.txt renamed) causes crash

**Test:** `TestCorruptedAndUnsupported::test_unsupported_text_as_png_return_rejected_without_crash`

**Root cause:** Same propagation path as group6a — text content decoded as bytes is rejected
by `cv2.imdecode`, `ValueError` bubbles up from `vision.decode_image` through `agent.inspect_upload`.

**Owner:** `agent.py / Kshitj` — same fix as group6a.

---

## Skips (4) — missing data

| Group | Test | Missing data |
|-------|------|-------------|
| group2 | `TestBlurryImage::test_blurry_image_label_exists` | MISSING DATA: blurry condition image file and corresponding `sample_images/labels.csv` entry (condition=blurry, expected_outcome=needs_retake or needs_human) |
| group3 | `TestTiltedImage::test_tilted_image_label_exists` | MISSING DATA: tilted condition image file and corresponding `sample_images/labels.csv` entry (condition=tilted, expected_outcome=read/reject_input) |
| group4 | `TestDarkImage::test_dark_image_label_exists` | MISSING DATA: dark/poorly-illuminated condition image file and corresponding `sample_images/labels.csv` entry (condition=dark, expected_outcome=read/reject_input) |
| group5 | `TestObstructedImage::test_obstructed_image_has_documented_retry_or_needs_human` | MISSING DATA: obstructed image file and corresponding `sample_images/labels.csv` entry specifying expected_outcome (needs_human/retry/reject_input) |

---

## Changes from Previous Run

| Before | After | Change |
|--------|-------|--------|
| 4 failed | 3 failed | group10 now passes (fixed test logic) |
| 5 passed | 6 passed | group10 passes with ambiguous-needle image |
| 3 skipped | 4 skipped | Added missing group2 (blurry) skip test |
| 10 warnings | 0 warnings | Added `pytest.ini` with registered marks |

### group10 fix explanation

The previous test drew a single needle at 225° using a tiny 320×320 image; the vision module
could not detect the dial circle and returned `needs_retake` (not `needs_human`).

The fixed test builds a larger 400×400 image with a clear circular dial bezel and **two
competing needle lines at 90°** apart. The vision module detects the dial but finds two equally
strong candidates that are not collinear (~180°), triggering `quality_flags=("ambiguous_needle",)`.
`agent.py` maps `ambiguous_needle` → `status="needs_human"`, satisfying the scenario contract.

No assertions were weakened. The 3 remaining failures stay strict with `agent.py / Kshitj`
as the documented owner.
