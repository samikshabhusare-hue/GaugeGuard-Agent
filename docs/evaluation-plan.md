# Evaluation Plan and Execution Instructions

## 1. Current Dataset Status

As verified by direct inspection of `data/samples/` and the codebase:
- **No real gauge images with verified reference readings are currently available** in the repository.
- In accordance with project integrity standards, **no synthetic readings, mock benchmark images, or fictional accuracy metrics have been invented**.
- A standardized label template (`data/samples/labels.template.csv` and `data/samples/labels.csv`) and an automated evaluation pipeline (`evaluate.py` / `gaugeguard/evaluation.py`) are provided to execute evaluation immediately upon acquisition of verified real-world samples.

---

## 2. Label Set Specification

### 2.1 Image Assembly Guidelines
- **Supported Gauge Design**: Collect images of one target gauge model matching a registered calibration profile (e.g., `EXAMPLE_270` or calibrated gauge profile).
- **Condition Diversity**: Include representative test samples across 5 core environmental conditions:
  1. `clear`: Ideal illumination, frontal perpendicular viewpoint.
  2. `rotated`: Perspective tilt or in-plane rotation (±15° to ±45°).
  3. `blurry`: Motion blur or slight defocus.
  4. `glare`: Specular reflections or harsh highlights obscuring dial markings.
  5. `partially_obstructed`: Finger, bezel, or fixture shadow partially covering needle or tick marks.
- **Data Governance & Licensing**: Record the provenance and explicit license or owner consent for every image. Keep unconsented or proprietary operational images strictly out of source control.
- **Split Separation**: Keep calibration/tuning images strictly isolated from the final evaluation set to prevent data leakage.

### 2.2 CSV Schema (`data/samples/labels.csv`)

The evaluation script expects a CSV file structured as follows (refer to `data/samples/labels.template.csv`):

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `filename` | string (required) | Path to image file relative to `images-dir` | `gauge_01_clear.jpg` |
| `gauge_profile` | string (optional) | Registered gauge profile identifier (defaults to `--profile`) | `EXAMPLE_270` |
| `ground_truth_value` | float (required) | Verified reference gauge reading in engineering units | `45.0` |
| `unit` | string | Engineering unit of measurement | `PSI` |
| `condition` | string | Image challenge condition (`clear`, `rotated`, `blurry`, `glare`, `partially_obstructed`) | `clear` |
| `human_acceptable` | boolean | Whether a human inspector considers the image readable (`true`/`false`) | `true` |
| `consent_license` | string | Provenance record (`CC0`, `CC-BY-4.0`, `owner_consent`) | `owner_consent` |
| `notes` | string | Contextual notes or rig observation | `Bench test rig at 22C` |

---

## 3. Evaluation Metrics and Formulas

The evaluation tool computes the following metrics against verified reference labels:

1. **Mean Absolute Error (MAE) for Accepted Readings**:
   Calculated strictly over predictions accepted by the agent (`status == "accepted"`):
   $$\text{MAE}_{\text{accepted}} = \frac{1}{|S_{\text{accepted}}|} \sum_{i \in S_{\text{accepted}}} |v_{\text{pred}, i} - v_{\text{ref}, i}|$$
   *(The script also reports `MAE_all` across all samples where a reading was calculated).*

2. **Share of Accepted Readings within Configurable Tolerance**:
   Percentage of accepted readings with absolute error within a configurable tolerance threshold $\tau$:
   $$\text{Share}_{\le \tau} = \frac{|\{i \in S_{\text{accepted}} : |v_{\text{pred}, i} - v_{\text{ref}, i}| \le \tau\}|}{|S_{\text{accepted}}|} \times 100\%$$
   *$\tau$ can be configured via `--tolerance <value>` (absolute units) or `--tolerance-fraction <fraction>` (percentage of full scale).*

3. **Incorrect-Acceptance Rate**:
   Tracks instances where the system accepted a reading (`status == "accepted"`) that either exceeded the allowable error tolerance ($|v_{\text{pred}} - v_{\text{ref}}| > \tau$) or was designated unreadable by a human (`human_acceptable == false`).
   Both denominators are explicitly reported:
   - **Denominator = Accepted Readings**:
     $$\text{Rate}_{\text{inc/acc}} = \frac{N_{\text{incorrect accepted}}}{N_{\text{accepted}}} \times 100\%$$
   - **Denominator = Total Valid Samples**:
     $$\text{Rate}_{\text{inc/total}} = \frac{N_{\text{incorrect accepted}}}{N_{\text{total valid}}} \times 100\%$$

4. **Human Escalation Rate**:
   Share of samples routed to human review (`needs_human == true`):
   $$\text{Rate}_{\text{escalation}} = \frac{N_{\text{escalated}}}{N_{\text{total valid}}} \times 100\%$$

5. **Retry Rate**:
   Share of samples requiring second-pass image enhancement or thresholding:
   $$\text{Rate}_{\text{retry}} = \frac{N_{\text{retried}}}{N_{\text{total valid}}} \times 100\%$$

6. **Latency Monitoring**:
   End-to-end execution latency per sample under documented runtime environments (local CPU vs. AWS deployment).

---

## 4. Execution Instructions

### Step 1: Place Images and Prepare Labels
1. Copy licensed gauge images into `data/samples/`.
2. Populate `data/samples/labels.csv` with verified values (use `data/samples/labels.template.csv` as a guide).

### Step 2: Run Evaluation CLI
Execute the evaluation script from the repository root:

```bash
# Default execution (reads data/samples/labels.csv and data/samples/)
python evaluate.py

# Specify custom absolute tolerance (e.g. ±2.5 units)
python evaluate.py --tolerance 2.5

# Specify relative tolerance fraction (e.g. 5% of full gauge scale)
python evaluate.py --tolerance-fraction 0.05

# Export evaluation artifacts to a directory
python evaluate.py --labels data/samples/labels.csv --images-dir data/samples/ --output-dir evaluation_results/

# Output machine-readable JSON summary to stdout
python evaluate.py --json
```

### Step 3: Inspect Output Artifacts
When `--output-dir` is specified, the script automatically exports:
- `metrics_summary.json`: Top-level summary of MAE, tolerance percentage, escalation rates, and counts.
- `per_image_results.csv`: Row-level log including predicted reading, reference value, absolute error, tolerance pass/fail, and quality flags.
- `failure_cases.json`: Detailed records of all samples that were escalated, retried, or failed vision detection.

---

## 5. Agentic Evidence Preservation

For each inspection sample, the evaluation pipeline captures the complete agent action log:
1. **Initial Pass**: Pre-flight Laplacian blur variance, contrast score, and initial dial/needle detection.
2. **Quality Signal**: Triggering reason for retry (e.g., low contrast score, ambiguous needle, low confidence).
3. **Selected Preprocessing Operation**: Alternate OpenCV method applied (e.g., CLAHE contrast enhancement or Gaussian adaptive thresholding).
4. **Retry Result**: Needle angle and confidence from the second pass.
5. **Agreement & Decision**: Outcome of `readings_agree()`, discrepancy check against tolerance, and final accept vs. human escalation decision.
6. **Unresolved Failures**: Every unresolvable case is verified to preserve its rationale in `InspectionResult.reading.reason` and correctly set `needs_human = True`.
