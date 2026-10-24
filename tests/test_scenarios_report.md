# Scenario Test Report

## Evidence: pytest output — `pytest tests/test_scenarios.py -q -rs`

```
Fssss........                                                            [100%]
================================== FAILURES ===================================
____ TestClearImage.test_clear_image_reading_exists_and_near_ground_truth _____

self = <tests.test_scenarios.TestClearImage object at 0x000001E47F7CCA50>

    def test_clear_image_reading_exists_and_near_ground_truth(self):
        label = next((r for r in SAMPLE_LABELS if r.get("condition") == "clear"), None)
        if label is None:
            pytest.skip("MISSING DATA: clear image label in sample_images/labels.csv")
        fixture_path = os.path.join(SAMPLE_IMAGES_DIR, label["filename"])
        with open(fixture_path, "rb") as f:
            image_bytes = f.read()
        profile = get_profile(label.get("gauge_profile", "EXAMPLE_270"))
        result = inspect_upload(image_bytes, profile=profile)
        assert result.reading.value is not None, (
            "FAILURE (agent.py / Kshitj): expected numeric reading for clear image"
        )
        true_val = float(label["true_value"])
        uncert = float(label["label_uncertainty"])
        assert abs(result.reading.value - true_val) <= uncert, (
            f"FAILURE: value {result.reading.value} vs truth {true_val} "
            f"(tolerance ±{uncert})"
        )
>       assert result.reading.status == "accepted", (
            "FAILURE (agent.py / Kshitj): expected accepted, got "
            + str(result.reading.status)
            + ". Demo profile (EXAMPLE_270) sets unverified_demo_profile flag which "
            "forces needs_human; agent must not escalate a clean demo reading."
        )
E       AssertionError: FAILURE (agent.py / Kshitj): expected accepted, got needs_human. Demo profile (EXAMPLE_270) sets unverified_demo_profile flag which forces needs_human; agent must not escalate a clean demo reading.
E       assert 'needs_human' == 'accepted'
E         
E         - accepted
E         + needs_human

tests\test_scenarios.py:72: AssertionError
=========================== short test summary info ===========================
SKIPPED [1] tests\test_scenarios.py:92: MISSING DATA: blurry condition image file and corresponding sample_images/labels.csv entry (condition=blurry, expected_outcome=needs_retake or needs_human)
SKIPPED [1] tests\test_scenarios.py:108: MISSING DATA: tilted condition image file and corresponding sample_images/labels.csv entry (condition=tilted, expected_outcome=read/reject_input)
SKIPPED [1] tests\test_scenarios.py:124: MISSING DATA: dark/poorly-illuminated condition image file and corresponding sample_images/labels.csv entry (condition=dark, expected_outcome=read/reject_input)
SKIPPED [1] tests\test_scenarios.py:140: MISSING DATA: obstructed image file and corresponding sample_images/labels.csv entry specifying expected_outcome (needs_human/retry/reject_input)
1 failed, 8 passed, 4 skipped in 3.03s
```

---

## Passes (8)

| Group | Test | Result |
|-------|------|--------|
| group6 | `TestCorruptedAndUnsupported::test_corrupted_bytes_return_rejected_without_crash` | PASSED |
| group6 | `TestCorruptedAndUnsupported::test_unsupported_text_as_png_return_rejected_without_crash` | PASSED |
| group7 | `TestUncertainReading::test_uncertainty_range_is_positive` | PASSED |
| group8 | `TestDeadZone::test_dead_zone_angle_returns_none_and_in_bounds_false` | PASSED |
| group8 | `TestDeadZone::test_agent_never_accepts_dead_zone` | PASSED |
| group9 | `TestDisagreement::test_readings_agree_false_for_hand_built_numbers` | PASSED |
| group9 | `TestDisagreement::test_agent_does_not_silently_accept_when_readings_disagree` | PASSED |
| group10 | `TestHumanVerification::test_needs_human_true_with_non_empty_reason_on_ambiguous_needle` | PASSED |

---

## Failures (1) — owned by `agent.py / Kshitj`

Assertion is **strictly maintained and not weakened**.

### group1 — Scenario 1: Clear Image — `accepted` status

**Test:** `TestClearImage::test_clear_image_reading_exists_and_near_ground_truth`

**Contract & Label Invariant:**
- Input: `sample_images/industrial_gauge.png`
- Ground Truth: Profile `EXAMPLE_270`, `true_value=66.73`, `label_uncertainty=0.74`, `expected_outcome=read` (accepted reading).
- Expected: `reading.status == "accepted"`, `needs_human == False`, `reading.value` within tolerance `±0.74`.

**Root Cause:**
`agent.py::inspect_upload` checks `_is_unverified_demo_profile(profile)`. Because `EXAMPLE_270` starts with `EXAMPLE`, it appends the flag `unverified_demo_profile` to `all_flags`. In the decision gate, any flag in `all_flags` forces status to `needs_human`. Consequently, `agent.py` returns `status="needs_human"` and `needs_human=True` even when vision successfully detects the needle and the reading is within tolerance.

**Owner:** `agent.py / Kshitj` — The agent must not escalate a clean reading from a verified sample image to `needs_human` solely because the profile name starts with `EXAMPLE`. Either whitelist profile verification or handle unverified demo flags outside the acceptance gate.

---

## Skips (4) — missing data (no ground truth invented)

Per instructions: *"Do not invent ground truth. SCENARIOS (one test group each; if no image/label exists for a scenario, create a test that is skipped with a message 'MISSING DATA: <what to collect>' and list it in your report)"*.

| Group | Test | Missing Data Description |
|-------|------|--------------------------|
| group2 | `TestBlurryImage::test_blurry_image_label_exists` | `MISSING DATA: blurry condition image file and corresponding sample_images/labels.csv entry (condition=blurry, expected_outcome=needs_retake or needs_human)` |
| group3 | `TestTiltedImage::test_tilted_image_label_exists` | `MISSING DATA: tilted condition image file and corresponding sample_images/labels.csv entry (condition=tilted, expected_outcome=read/reject_input)` |
| group4 | `TestDarkImage::test_dark_image_label_exists` | `MISSING DATA: dark/poorly-illuminated condition image file and corresponding sample_images/labels.csv entry (condition=dark, expected_outcome=read/reject_input)` |
| group5 | `TestObstructedImage::test_obstructed_image_has_documented_retry_or_needs_human` | `MISSING DATA: obstructed image file and corresponding sample_images/labels.csv entry specifying expected_outcome (needs_human/retry/reject_input)` |

---

## Summary of Fixes

1. **Restored Scenario 1 Contract Assertion:** Removed the weakened assertion `assert result.reading.status == "needs_human"` introduced in a prior commit. Restored strict assertion `assert result.reading.status == "accepted"` and `assert not result.needs_human`. The test is kept failing with owner marked as `agent.py / Kshitj`.
2. **Removed Invented Ground Truth for Scenarios 2–5:** Replaced synthesized OpenCV transformations (`cv2.GaussianBlur`, `cv2.warpAffine`, etc.) with standard data-presence checks that cleanly `pytest.skip` with `"MISSING DATA: <what to collect>"`, adhering to the rule *do not invent ground truth*.
3. **Verified Scenario 6 Handling:** Corrupted bytes and unsupported text uploads now return `rejected_input` without crashing, successfully passing both tests under `TestCorruptedAndUnsupported`.
4. **Enhanced Scenario 8 & 9 Public Function Contracts:** Tested calibration logic and public `agent.inspect_upload` interface without modifying `agent.py`:
   - Scenario 8: Verified calibration dead-zone bounds checking and agent rejection.
   - Scenario 9: Verified `readings_agree` disagreement detection and agent refusal to accept silently on retry disagreement.
5. **Scenario 10 End-to-End Test:** Uses ambiguous needle geometry to verify `status="needs_human"`, `needs_human=True`, and non-empty reason.
