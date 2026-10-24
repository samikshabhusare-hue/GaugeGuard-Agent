# Scenario Test Report

## Command: `pytest tests/test_scenarios.py -q -rs`

## Evidence: Real pytest output

```
Fssss........                                                            [100%]
================================== FAILURES ===================================
____ TestClearImage.test_clear_image_reading_exists_and_near_ground_truth _____

self = <tests.test_scenarios.TestClearImage object at 0x000001B9DA6A4A50>

    def test_clear_image_reading_exists_and_near_ground_truth(self):
        """Clear image with expected_outcome=read must be accepted within tolerance.
    
        Label: industrial_gauge.png, EXAMPLE_270, true_value=66.73,
               label_uncertainty=0.74, expected_outcome=read.
        Contract: status must be 'accepted', needs_human must be False.
        """
        label = next((r for r in SAMPLE_LABELS if r.get("condition") == "clear"), None)
        if label is None:
            pytest.skip("MISSING DATA: clear image label in sample_images/labels.csv")
        fixture_path = os.path.join(SAMPLE_IMAGES_DIR, label["filename"])
        with open(fixture_path, "rb") as f:
            image_bytes = f.read()
        profile = get_profile(label.get("gauge_profile", "EXAMPLE_270"))
        result = inspect_upload(image_bytes, profile=profile)
        assert result.reading.value is not None, (
            "FAILURE (agent.py / Kshitij): expected numeric reading for clear image"
        )
        true_val = float(label["true_value"])
        uncert = float(label["label_uncertainty"])
        assert abs(result.reading.value - true_val) <= uncert, (
            f"FAILURE: value {result.reading.value} vs truth {true_val} "
            f"(tolerance ±{uncert})"
        )
        # Contract + label: expected_outcome=read means status must be 'accepted'.
        # Do NOT weaken this assertion. If agent.py escalates a clean demo reading
        # to needs_human via the unverified_demo_profile flag, that is a bug in
        # agent.py that must be fixed by its owner, not masked in tests.
>       assert result.reading.status == "accepted", (
            "FAILURE (agent.py / Kshitij): expected accepted, got "
            + str(result.reading.status)
            + ". Demo profile (EXAMPLE_270) sets unverified_demo_profile flag which "
            "forces needs_human; agent must not escalate a clean demo reading."
        )
E       AssertionError: FAILURE (agent.py / Kshitij): expected accepted, got needs_human. Demo profile (EXAMPLE_270) sets unverified_demo_profile flag which forces needs_human; agent must not escalate a clean demo reading.
E       assert 'needs_human' == 'accepted'
E         
E         - accepted
E         + needs_human

tests\test_scenarios.py:82: AssertionError
=========================== short test summary info ===========================
SKIPPED [1] tests\test_scenarios.py:103: MISSING DATA: blurry condition image file and corresponding sample_images/labels.csv entry (condition=blurry, expected_outcome=needs_retake or needs_human)
SKIPPED [1] tests\test_scenarios.py:120: MISSING DATA: tilted condition image file and corresponding sample_images/labels.csv entry (condition=tilted, expected_outcome=read/reject_input)
SKIPPED [1] tests\test_scenarios.py:140: MISSING DATA: dark/poorly-illuminated condition image file and corresponding sample_images/labels.csv entry (condition=dark, expected_outcome=read/reject_input)
SKIPPED [1] tests\test_scenarios.py:160: MISSING DATA: obstructed image file and corresponding sample_images/labels.csv entry specifying expected_outcome (needs_human/retry/reject_input)
1 failed, 8 passed, 4 skipped in 2.92s
```

---

## Summary: 1 failed, 8 passed, 4 skipped

---

## Passes (8)

| Group | Scenario | Test |
|-------|----------|------|
| group6 | 6 – Corrupted file | `TestCorruptedAndUnsupported::test_corrupted_bytes_return_rejected_without_crash` |
| group6 | 6 – Unsupported file | `TestCorruptedAndUnsupported::test_unsupported_text_as_png_return_rejected_without_crash` |
| group7 | 7 – Uncertain reading | `TestUncertainReading::test_uncertainty_range_is_positive` |
| group8 | 8 – Dead zone (calibration) | `TestDeadZone::test_dead_zone_angle_returns_none_and_in_bounds_false` |
| group8 | 8 – Dead zone (agent) | `TestDeadZone::test_agent_never_accepts_dead_zone` |
| group9 | 9 – Disagreement (calibration) | `TestDisagreement::test_readings_agree_false_for_hand_built_numbers` |
| group9 | 9 – Disagreement (agent) | `TestDisagreement::test_agent_does_not_silently_accept_when_readings_disagree` |
| group10 | 10 – Human verification | `TestHumanVerification::test_needs_human_true_with_non_empty_reason_on_ambiguous_needle` |

---

## Failures (1) — owner: `agent.py / Kshitij`

### group1 — Scenario 1: Clear image must be accepted

**Test:** `TestClearImage::test_clear_image_reading_exists_and_near_ground_truth`

**Label data:** `industrial_gauge.png`, profile `EXAMPLE_270`, `true_value=66.73`, `label_uncertainty=0.74`, `expected_outcome=read`.

**What the test asserts:** `result.reading.status == "accepted"` and `result.needs_human == False`.

**What happens:** `agent.py::_is_unverified_demo_profile()` flags `EXAMPLE_270` because its name starts with `EXAMPLE`. This sets `status = "needs_human"` in the agent decision logic — even when the reading is numerically accurate and within label tolerance.

**Owner:** `agent.py / Kshitij` — The assertion is **not weakened** per the task rules:
*"For any test that fails because of another module, mark it clearly with the owner's name and keep it failing (do not weaken the assertion)."*
The agent module must not escalate a clean, label-verified reading to `needs_human` solely because the profile name starts with `EXAMPLE`.

---

## Skips (4) — missing data

No ground truth was invented. Per the task rules (*"Do not invent ground truth. SCENARIOS (one test group each; if no image/label exists for a scenario, create a test that is skipped with a message 'MISSING DATA: <what to collect>' and list it in your report)"*):

| Group | Scenario | Missing Data |
|-------|------|--------------------------|
| group2 | 2 – Blurry image | `MISSING DATA: blurry condition image file and corresponding sample_images/labels.csv entry (condition=blurry, expected_outcome=needs_retake or needs_human)` |
| group3 | 3 – Tilted image | `MISSING DATA: tilted condition image file and corresponding sample_images/labels.csv entry (condition=tilted, expected_outcome=read/reject_input)` |
| group4 | 4 – Poorly illuminated | `MISSING DATA: dark/poorly-illuminated condition image file and corresponding sample_images/labels.csv entry (condition=dark, expected_outcome=read/reject_input)` |
| group5 | 5 – Partially obstructed | `MISSING DATA: obstructed image file and corresponding sample_images/labels.csv entry specifying expected_outcome (needs_human/retry/reject_input)` |
