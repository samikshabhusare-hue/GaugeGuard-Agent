# Agent Controller: local run and verification

This runbook covers the agent-controller responsibility: quality gating, selecting an alternate preprocessing retry, comparing readings, choosing accept/review/retake, and recording the action trace. It does not claim that the full Streamlit app or AWS deployment is complete.

## 1. Recommended Windows setup

Use **Python 3.12 (64-bit)** for the project's pinned OpenCV 5 target. The project's Python 3.14 environment previously failed dependency resolution, so do not keep reusing that virtual environment.

From PowerShell, in the repository folder:

```powershell
# Confirm Python 3.12 is installed
py -3.12 --version

# Create a fresh environment (do not reuse the old .venv)
py -3.12 -m venv .venv312
.\.venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Install the agent-controller runtime only
python -m pip install -r requirements-agent.txt

# Confirm the required OpenCV major version
python -c "import cv2, numpy; print('OpenCV:', cv2.__version__); print('NumPy:', numpy.__version__)"
```

Before describing this as OpenCV 5 verified, the version printed for OpenCV must begin with `5.`. If installation still fails, preserve the complete pip error; do not silently replace OpenCV 5 with OpenCV 4 for the competition build.

## 2. Run the controller on a real image

```powershell
python run_agent.py "C:\path\to\gauge.jpg" --json output\agent-trace.json
```

The console prints the final decision and every action-log step. The optional JSON file contains image quality, attempts, the reading decision, and the complete audit trace.

The only registered profile in this repository is `EXAMPLE_270`, a placeholder profile. It is intentionally treated as unverified and should escalate to human review. Do not use its numeric result as a calibrated real-world measurement.

## 3. Run controller tests

Install test tooling in the same virtual environment, then run:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests validate the controller's retry/compare/escalate paths and calibration/vision behavior. A passing test suite on OpenCV 4 is useful for development, but does **not** prove the OpenCV 5 requirement has been met.

## 4. Controller acceptance checklist

- [ ] A valid image runs through quality assessment and first-pass detection.
- [ ] Low confidence / uncertain detection triggers an alternate preprocessing pass.
- [ ] When both passes succeed, their readings are compared against profile tolerance.
- [ ] Disagreement results in `needs_human`, with no numeric final reading.
- [ ] Failed detection results in human review or image retake, not a fabricated reading.
- [ ] The action trace records the selected method, rationale, flags, and final decision.
- [ ] OpenCV version is confirmed as 5.x in the same environment used for the demo.
- [ ] A teammate supplies real gauge calibration and labelled evaluation data before any real-world accuracy claim.
