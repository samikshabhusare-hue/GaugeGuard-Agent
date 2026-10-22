# Sample Images and Evaluation Dataset

This directory is designated for evaluation images of analog gauges.

## Dataset Guidelines

1. **Licensing and Consent**: Add only images that are licensed for this use (e.g., CC0, CC-BY) or captured with the explicit consent of the equipment owner. Keep private, proprietary, or unverified images out of Git.
2. **Diversity of Test Conditions**: Assemble evaluation images covering clear, rotated, blurry, glare, and partially obstructed dials. Keep calibration/tuning images strictly separate from final evaluation images.
3. **Verified Reference Readings**: Each sample must be paired with an independently verified reference reading, documented unit, and calibration profile name.
4. **Labels Template**:
   - Use `labels.template.csv` or `labels.csv` as the schema template.
   - Column format:
     ```csv
     filename,gauge_profile,ground_truth_value,unit,condition,human_acceptable,consent_license,notes
     ```
5. **Running Evaluation**:
   Once images and labels are populated, run:
   ```bash
   python evaluate.py --labels data/samples/labels.csv --images-dir data/samples/
   ```
   For details on metrics (MAE, % within tolerance, escalation rate), see `docs/evaluation-plan.md`.
