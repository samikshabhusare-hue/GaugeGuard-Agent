# GaugeGuard Evaluation Report: `local`

Generated from run output in `evaluation_results\local`.

## 1. Overall Metrics: Agent vs Baseline

| Metric | Agent Result | Baseline (First-Pass Only) | Definition |
|:---|:---|:---|:---|
| **Coverage** | 0.0% (0/1) [0.0%, 79.3%] | 100.0% (1/1) [20.7%, 100.0%] | Accepted readings / Evaluable samples |
| **Mean Absolute Error (MAE)** | N/A (n=0) | 0.00 (n=1) | Mean error on accepted readings with ground truth |
| **Within-Tolerance Rate** | N/A (0/0) | 100.0% (1/1) [20.7%, 100.0%] | Accepted readings with error ≤ tolerance / Accepted |
| **Incorrect Acceptance Rate** | N/A (0/0) | 0.0% (0/1) [0.0%, 79.3%] | Accepted readings with error > tolerance / Accepted |
| **Retry Rate** | 0.0% (0/1) [0.0%, 79.3%] | 0.0% (0/1) | Inspections requiring retry pass / Evaluable |
| **Human Escalation Rate** | 100.0% (1/1) [20.7%, 100.0%] | 0.0% (0/1) | Inspections routed to human review / Evaluable |
| **Harness / Agent Errors** | 0 (0.0%) | N/A | Processing failures / Total |
| **Rejected Inputs** | 0 (0.0%) | N/A | Corrupt/unreadable inputs rejected / Total |

*Note: Sample size n = 1. Results do not generalize beyond this gauge and these images.*

## 2. Per-Condition Performance Breakdown

| Condition | Total (n) | Accepted | Coverage (95% CI) | Escalated | MAE | Within Tol (95% CI) | Errors | Rejected |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| `clear` | 1 | 0 | 0.0% (0/1) [0.0%, 79.3%] | 1 | N/A | N/A (0/0) | 0 | 0 |

*Note: Sample size n = 1. Results do not generalize beyond this gauge and these images.*

## 3. Human Escalation Table

Images routed to human review (`needs_human` / `needs_retake`), with triggering flags and rationale:

| Filename | Condition | Attempts | Reading | Triggering Flags / Rationale |
|:---|:---|:---|:---|:---|
| `industrial_gauge.png` | `clear` | 1 | 66.73 | Needle detected at angle 45.2° with center (629.4, 556.2) and radius 454.3px.; linear_scale_assumed; unverified_demo_profile; Angle 45.2° mapped to value 66.73 units.; Reading calculated (66.73 units), but human review required: unverified_demo_profile; decision: needs_human |

*Note: Sample size n = 1. Results do not generalize beyond this gauge and these images.*

## 4. Failure Cases

Detailed analysis of failure events: wrong accepted readings, errors, false rejections, and missed escalations.

| Filename | Failure Category | Predicted | True Value | Absolute Error | Tolerance | Reason / Description |
|:---|:---|:---|:---|:---|:---|:---|
| *None* | - | - | - | - | - | No failure cases observed (0 failures) |

*Note: Sample size n = 1. Results do not generalize beyond this gauge and these images.*

## 5. Latency & Environment Summary

### Latency Profile

| Metric | Value | Note |
|:---|:---|:---|
| **Sample Count (n)** | 1 | Total timed samples |
| **Mean Latency** | 1787.8 ms | Arithmetic average per sample |
| **Median Latency** | 1787.8 ms | 50th percentile |
| **p95 Latency** | 1787.8 ms | 95th percentile *(unreliable: n < 20)* |
| **Min Latency** | 1787.8 ms | Fastest execution |
| **Max Latency** | 1787.8 ms | Slowest execution |

### Execution Environment

| Property | Recorded Value |
|:---|:---|
| **Compute Platform / OS** | `Windows-11-10.0.26300-SP0` |
| **Python Version** | `3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:21:58) [MSC v.1929 64 bit (AMD64)]` |
| **OpenCV (`cv2`) Version** | `5.0.0` |
| **NumPy Version** | `2.4.6` |
| **Compute Instance Type** | `Local Workstation` |
| **Environment Warnings** | `None` |

*Note: Sample size n = 1. Results do not generalize beyond this gauge and these images.*
