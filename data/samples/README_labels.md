# Labeling Rules for GaugeGuard Reference Set

This document specifies how to create and verify ground-truth labels for the GaugeGuard evaluation dataset. **Never fabricate ground-truth values.**

---

## 1. Source of Truth

All `true_value` entries must come from **one of two verified sources**:

### A. Known Setpoint (Preferred)
- The gauge was set to a known, traceable reference value (e.g., dead-weight tester, calibrated pressure source, calibrated temperature bath).
- Record the source in `label_source` (e.g., "Druck DPI 610 pressure calibrator, certificate #12345").
- `label_uncertainty` = the expanded uncertainty (k=2) of the reference standard, or if unknown, **half a minor division** on the gauge dial.

### B. Two Independent Human Reads
- Two qualified operators independently read the gauge from the **same image** without conferring.
- Record both reads in `notes` (e.g., "Operator A: 45.2 PSI, Operator B: 45.0 PSI").
- If the two reads differ by **more than `label_uncertainty`**, **drop the image** from the dataset (do not label it).
- If they agree within `label_uncertainty`, use the **mean** as `true_value`.
- `label_uncertainty` = half the minor division on the gauge dial, unless a tighter uncertainty is justified by the reading method (document justification in `notes`).

---

## 2. Uncertainty Assignment

| Situation | `label_uncertainty` |
|-----------|---------------------|
| Known setpoint with traceable certificate | Certificate expanded uncertainty (k=2) |
| Known setpoint without certificate | Half a minor division |
| Two independent reads | Half a minor division (or tighter if justified in `notes`) |
| Single human read | **Not permitted** — requires two independent reads or known setpoint |

**Minor division** = the smallest tick mark interval printed on the gauge dial (e.g., if dial shows 0, 10, 20... with ticks at 1, 2, 3..., minor division = 1 unit).

---

## 3. Column Definitions (`labels.csv`)

| Column | Required | Type | Rules |
|--------|----------|------|-------|
| `filename` | Yes | string | Relative path to image file in `data/samples/` or subdirectory. Must exist on disk. |
| `gauge_profile` | Yes | string | Must match a registered profile name in `gaugeguard.calibration.PROFILES`. |
| `true_value` | Conditional | float | **Required when `expected_outcome = read`**. Must be within `[profile.minimum_value, profile.maximum_value]`. **Must be empty when `expected_outcome = reject_input`**. |
| `label_uncertainty` | Yes | float | Non-negative. Assigned per Section 2. |
| `unit` | Yes | string | Must match `profile.unit` (e.g., "PSI", "bar", "kPa", "°C"). |
| `condition` | Yes | string | One of: `clear`, `blurry`, `tilted`, `dark`, `obstructed`, `other`. |
| `expected_outcome` | Yes | string | `read` — image should produce a valid reading; `reject_input` — image quality too poor, should be rejected. |
| `label_source` | Yes | string | Free text. Examples: "known setpoint — Druck DPI 610", "two independent human reads (A: 45.2, B: 45.0 PSI)". |
| `notes` | No | string | Context: minor division value, operator IDs, justification for uncertainty, reason for rejection, etc. |

---

## 4. Condition Categories

| Condition | Description |
|-----------|-------------|
| `clear` | Ideal illumination, perpendicular view, no obstructions. |
| `blurry` | Motion blur or defocus — needle/ticks not sharp. |
| `tilted` | Camera not perpendicular to dial face (perspective distortion). |
| `dark` | Low illumination, underexposed, shadows on dial. |
| `obstructed` | Partial occlusion of needle or critical tick marks (finger, bezel, dirt). |
| `other` | Any other degradation (glare, reflection, damaged dial, etc.) — explain in `notes`. |

---

## 5. Expected Outcome Categories

| Value | Meaning |
|-------|---------|
| `read` | A valid gauge reading should be produced. `true_value` **must be present**. |
| `reject_input` | Image quality insufficient; system should reject/escalate. `true_value` **must be empty**. |

---

## 6. Validation Rules (Enforced by `evaluation.py --validate-labels`)

The validator checks for:
1. **Missing files**: Every `filename` must exist in the images directory.
2. **Duplicate filenames**: Each image must appear at most once.
3. **Unknown profile**: `gauge_profile` must be registered in `gaugeguard.calibration`.
4. **True value missing for `read` cases**: If `expected_outcome == "read"`, `true_value` must be present and numeric.
5. **Value outside profile range**: `true_value` must be within `[minimum_value, maximum_value]` of the profile.
6. **Unknown condition**: `condition` must be one of the six defined categories.
7. **Missing label_source**: Every row must have a `label_source`.
8. **Malformed CSV**: Reports row number for any parsing error without crashing.
9. **Header-only CSV**: Reports "0 labelled images" clearly (does not pass silently).

---

## 7. Images Without Labels

Run `python evaluation.py --validate-labels` to list all images in `data/samples/` that have no corresponding row in `labels.csv`. These must be labeled or explicitly excluded (with reason in `notes` of a placeholder row, or moved out of the samples directory).

---

## 8. Workflow for Adding a New Labeled Image

1. Place image file in `data/samples/` (or appropriate subdirectory).
2. Determine `true_value` via **known setpoint** or **two independent human reads** (Section 1).
3. Assign `label_uncertainty` per Section 2.
4. Classify `condition` per Section 4.
5. Set `expected_outcome` per Section 5.
6. Record `label_source` and any `notes`.
7. Append row to `labels.csv`.
8. Run `python evaluation.py --validate-labels` to validate.
9. Commit both image and CSV row together.

---

## 9. Example Rows

```csv
# Known setpoint example
gauge_001_clear.jpg,EXAMPLE_270,45.0,0.5,PSI,clear,read,"known setpoint — Druck DPI 610 calibrator, cert #2024-001","Minor division = 1 PSI; certificate uncertainty k=2 = 0.5 PSI"

# Two independent reads example
gauge_002_blurry.jpg,EXAMPLE_270,78.5,1.0,PSI,blurry,read,"two independent human reads (A: 78.0, B: 79.0 PSI)","Minor division = 2 PSI; half = 1 PSI; reads differ by 1.0 <= uncertainty"

# Reject example (no true_value)
gauge_003_obstructed.jpg,EXAMPLE_270,,1.0,PSI,obstructed,reject_input,"two independent human reads (A: unreadable, B: unreadable)","Needle fully obscured by bezel shadow"
```

---

## 10. Prohibited Practices

- ❌ Copying readings from the GaugeGuard pipeline output
- ❌ Single human read without a known setpoint
- ❌ Using images without verified provenance/license
- ❌ Inventing `true_value` for `reject_input` cases
- ❌ Modifying labels to "improve" metrics
- ❌ Using calibration/tuning images as evaluation samples

---

*This document is part of the GaugeGuard evaluation framework. Update it when new profiles or conditions are added.*