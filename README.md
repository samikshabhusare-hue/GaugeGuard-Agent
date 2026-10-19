# GaugeGuard Agent

A self-verifying computer-vision prototype for reading one supported type of analog industrial gauge. It estimates the needle value, checks image quality and calibration bounds, and requests another image or human verification when evidence is not reliable.

> **Safety:** GaugeGuard is a prototype, not a certified measuring instrument. Never use it to control equipment or make safety-critical decisions. A qualified person must verify uncertain readings.

## Project status

This repository is the initial project scaffold. The upload UI and OpenCV image-quality check are present. Dial/needle detection, the agent retry loop, AWS deployment, and evaluation are planned implementation work and are not yet complete. Do not present the scaffold as a working gauge reader.

## Planned workflow

1. Upload a gauge image.
2. OpenCV 5 checks the image and estimates the dial and needle.
3. Calibration maps the needle angle to a measurement.
4. The agent accepts a sufficiently supported result, retries with a different OpenCV method when image evidence is uncertain, or asks for human review.
5. The UI shows the reading, annotated image, confidence/quality indicators, and action trace.

The core project is intentionally limited to one gauge style and one documented calibration profile first.

## Stack

- Python
- OpenCV 5 (`opencv-python-headless`)
- NumPy
- Streamlit
- AWS EC2 for the deployed app and actual OpenCV workload
- Optional Amazon S3 for explicitly approved output storage

## Team ownership

| Teammate | Focus |
|---|---|
| Bhumika | Streamlit UI, integration, README, report and demo coordination |
| Samiksha | OpenCV vision pipeline: preprocessing, dial/needle detection, angle estimate and annotation |
| Prasoon | Gauge calibration, labelled reference images and evaluation metrics |
| Kshitj | Agent controller, retry policy, comparison, escalation and action history |

## Repository layout

```text
app.py                       Streamlit entry point

gaugeguard/
  agent.py                    quality-gated orchestration (scaffold)
  calibration.py              angle-to-reading conversion
  vision.py                   OpenCV image loading and quality analysis
  models.py                   shared result types

data/samples/                 add consented/licensed sample images; do not commit sensitive footage

docs/
  architecture.mmd            editable Mermaid architecture diagram
  evaluation-plan.md          proposed metrics and dataset notes
  report-outline.md           technical report outline
requirements.txt              pinned direct dependencies
```

## Setup

Use Python 3.11 or newer and a virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

Confirm that the printed OpenCV version starts with `5.` before claiming OpenCV 5 is running. Use the same check in the local environment and the deployed AWS environment.

### Run the scaffold

```powershell
streamlit run app.py
```

The initial app can accept an image and display an OpenCV image-quality estimate. Needle reading is not implemented yet; the UI says so explicitly.

## OpenCV 5 implementation milestones

1. Pick one gauge model and document its scale, dial center, minimum/maximum angles, units, and allowed reading tolerance.
2. Implement dial localization and needle direction estimation in `gaugeguard/vision.py` with OpenCV 5 operations.
3. Define and test the angle convention, then use `gaugeguard/calibration.py` to convert angle into a reading.
4. Add quality checks for blur, glare, obstruction, and geometry; never turn low confidence into a guessed reading.
5. Implement an evidence-triggered retry in `gaugeguard/agent.py`: select an alternate crop/contrast/threshold method based on the first-pass failure signal, compare results, then accept or escalate.
6. Record each tool/method, parameters, output, quality flags, and final decision in the action trace.

## AWS deployment plan

Deploy the Streamlit app and the actual OpenCV 5 processing code on an EC2 instance. Restrict inbound access, use an instance role for AWS access, and never put credentials in source code. S3 storage is optional; only save images with clear user consent and a defined retention period. Record the instance type, OpenCV version, deployment steps, and measured latency for the report.

## Evaluation plan

Use separate calibration and evaluation images with known readings. Include clear, tilted, blurry, glare, and partially obstructed examples. Report measured results only:

- Mean absolute reading error for accepted readings
- Percentage of accepted readings within a predefined tolerance
- Incorrect-acceptance rate (define the denominator)
- Retry and human-escalation rates
- Local and AWS processing latency
- Failure examples and supported gauge limitations

The agentic demo must show a first-pass visual result causing a different follow-up OpenCV call, and a case where unresolved uncertainty results in human review.

## Security and data

- Do not commit AWS keys, credentials, private images, or personal data.
- Use licensed or consented images in `data/samples/`.
- Keep secrets in environment variables or AWS IAM roles.
- Do not use the prototype to control hazardous equipment.

## Competition submission checklist

- [ ] Verify OpenCV 5 in the exact runtime used by the app.
- [ ] Run the real OpenCV 5 vision workload on AWS and show it in the demo.
- [ ] Show visual evidence changing a later agent action.
- [ ] Include pinned dependencies and reproducible setup/deploy instructions.
- [ ] Publish an architecture diagram, evaluation results, limitations, and failure cases.
- [ ] Record a judge-accessible video no longer than five minutes.

Check the competition's live rules before submitting: <https://opencv26.devpost.com/>.
