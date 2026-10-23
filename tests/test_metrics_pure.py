"""Hand-computed metric fixtures — arithmetic shown in comments.
These are test fixtures, not results.
"""
from __future__ import annotations

from gaugeguard.evaluation import (
    EvalRecord,
    compute_metrics,
    wilson_interval,
    metrics_by_condition,
    latency_summary,
)


# === Fixture: basic accepted with true_value ===
# 3 accepted, all with true_value present, errors = 1.0, 2.0, 3.0
# MAE = (1 + 2 + 3) / 3 = 2.0
# Within tol = 2.0: |1|<=2 yes, |2|<=2 yes, |3|>2 no -> 2/3 = 0.667
# Incorrect = 1/3 = 0.333 (only 3.0 > 2.0)

def test_compute_metrics_basic_accepted():
    records = [
        EvalRecord(filename="a.jpg", condition="clear", expected_outcome="read",
                   true_value=10.0, tolerance=2.0, status="accepted",
                   reading=11.0, retried=False, attempts=1, latency_ms=120.0,
                   first_pass_reading=11.0, error_message=""),
        EvalRecord(filename="b.jpg", condition="clear", expected_outcome="read",
                   true_value=20.0, tolerance=2.0, status="accepted",
                   reading=22.0, retried=False, attempts=1, latency_ms=150.0,
                   first_pass_reading=22.0, error_message=""),
        EvalRecord(filename="c.jpg", condition="clear", expected_outcome="read",
                   true_value=30.0, tolerance=2.0, status="accepted",
                   reading=33.0, retried=False, attempts=1, latency_ms=180.0,
                   first_pass_reading=33.0, error_message=""),
    ]
    m = compute_metrics(records)
    # Arithmetic: MAE = (|11-10|+|22-20|+|33-30|)/3 = (1+2+3)/3 = 2.0
    assert m["mae"][0] == 2.0  # (value, count)
    # Within tol: 2 of 3 within 2.0 -> 0.667
    assert round(m["within_tolerance_rate"][0], 3) == 0.667
    # Incorrect: 1 of 3 (only error 3.0 > 2.0)
    assert round(m["incorrect_acceptance_rate"][0], 3) == 0.333
    # Counts carry
    assert m["n_accepted"] == 3
    assert m["n_evaluable"] == 3
    assert m["n_errors"] == 0
    assert m["note"] == "all denominators > 0"


# === Fixture: zero accepted ===
def test_compute_metrics_zero_accepted():
    records = [
        EvalRecord(filename="e.jpg", condition="blurry", expected_outcome="read",
                   true_value=10.0, tolerance=2.0, status="error",
                   reading=None, retried=False, attempts=1, latency_ms=None,
                   first_pass_reading=None, error_message="detect_failed"),
    ]
    m = compute_metrics(records)
    # Zero accepted -> rates None, counts preserved
    assert m["mae"][0] is None
    assert m["within_tolerance_rate"][0] is None
    assert m["incorrect_acceptance_rate"][0] is None
    assert m["n_accepted"] == 0
    assert m["n_evaluable"] == 0  # error excludes
    assert "n_accepted=0" in m["note"]


# === Fixture: boundary |err| == tol ===
# Error exactly at tolerance should count as WITHIN (<=)

def test_compute_metrics_boundary_exact_tolerance():
    # 1 accepted, error = 2.0, tol = 2.0 -> within (<=)
    # 1 accepted, error = 2.01, tol = 2.0 -> outside (>)
    records = [
        EvalRecord(filename="b1.jpg", condition="clear", expected_outcome="read",
                   true_value=100.0, tolerance=2.0, status="accepted",
                   reading=102.0, retried=False, attempts=1, latency_ms=100.0,
                   first_pass_reading=102.0),
        EvalRecord(filename="b2.jpg", condition="clear", expected_outcome="read",
                   true_value=100.0, tolerance=2.0, status="accepted",
                   reading=102.01, retried=False, attempts=1, latency_ms=100.0,
                   first_pass_reading=102.01),
    ]
    m = compute_metrics(records)
    # Within: 1/2 = 0.5 (only 102.0 <= 2.0; 102.01 > 2.0)
    assert m["within_tolerance_rate"][0] == 0.5
    # Incorrect: 1/2 = 0.5 (only 102.01 > 2.0)
    assert m["incorrect_acceptance_rate"][0] == 0.5


# === Fixture: all errors ===
def test_compute_metrics_all_errors():
    records = [
        EvalRecord(filename="e1.jpg", condition="dark", expected_outcome="read",
                   true_value=10.0, tolerance=5.0, status="error",
                   reading=None, retried=False, attempts=1, latency_ms=None,
                   first_pass_reading=None, error_message="fail"),
        EvalRecord(filename="e2.jpg", condition="dark", expected_outcome="read",
                   true_value=20.0, tolerance=5.0, status="error",
                   reading=None, retried=False, attempts=1, latency_ms=None,
                   first_pass_reading=None, error_message="fail"),
    ]
    m = compute_metrics(records)
    assert m["n_errors"] == 2
    assert m["n_evaluable"] == 0
    assert m["n_accepted"] == 0
    assert m["mae"][0] is None
    assert "n_evaluable=0" in m["note"]


# === Fixture: mixed conditions ===

def test_metrics_by_condition():
    records = [
        EvalRecord(filename="a.jpg", condition="clear", expected_outcome="read",
                   true_value=10.0, tolerance=2.0, status="accepted",
                   reading=10.0, retried=False, attempts=1, latency_ms=100.0,
                   first_pass_reading=10.0),
        EvalRecord(filename="b.jpg", condition="blurry", expected_outcome="read",
                   true_value=20.0, tolerance=2.0, status="needs_human",
                   reading=20.5, retried=True, attempts=2, latency_ms=250.0,
                   first_pass_reading=19.5),
    ]
    by_cond = metrics_by_condition(records)
    assert "clear" in by_cond
    assert "blurry" in by_cond
    # Clear: 1 accepted, 0 errors -> evaluable=1, coverage=1.0
    assert by_cond["clear"]["n_accepted"] == 1
    # Blurry: needs_human, not accepted -> accepted=0, retry=1
    assert by_cond["blurry"]["n_accepted"] == 0
    assert by_cond["blurry"]["n_needs_human"] == 1


# === Fixture: retry / escalation ===

def test_retry_and_escalation_rates():
    # 4 evaluable, 2 retried, 1 needs_human
    # retry_rate = 2/4 = 0.5; escalation = 1/4 = 0.25
    records = [
        EvalRecord(filename="r1.jpg", condition="clear", expected_outcome="read",
                   true_value=10.0, tolerance=2.0, status="accepted",
                   reading=10.5, retried=True, attempts=2, latency_ms=100.0,
                   first_pass_reading=10.5),
        EvalRecord(filename="r2.jpg", condition="clear", expected_outcome="read",
                   true_value=20.0, tolerance=2.0, status="accepted",
                   reading=21.0, retried=False, attempts=1, latency_ms=110.0,
                   first_pass_reading=21.0),
        EvalRecord(filename="r3.jpg", condition="blurry", expected_outcome="read",
                   true_value=30.0, tolerance=2.0, status="needs_human",
                   reading=30.5, retried=True, attempts=2, latency_ms=300.0,
                   first_pass_reading=30.5),
        EvalRecord(filename="r4.jpg", condition="clear", expected_outcome="read",
                   true_value=40.0, tolerance=2.0, status="accepted",
                   reading=41.0, retried=False, attempts=1, latency_ms=120.0,
                   first_pass_reading=41.0),
    ]
    m = compute_metrics(records)
    # Retried: r1, r3 = 2; evaluable = 4
    assert m["retry_rate"][0] == 0.5
    # Needs_human: r3 = 1; evaluable = 4
    assert m["escalation_rate"][0] == 0.25


# === Fixture: latency n < 20 => p95_unreliable ===

def test_latency_small_n_unreliable():
    # 3 values (< 20) -> p95_unreliable = True
    result = latency_summary([120.0, 150.0, 180.0])
    assert result["n"] == 3
    assert result["mean"] == 150.0  # (120+150+180)/3
    assert result["median"] == 150.0
    assert result["p95_unreliable"] is True


def test_latency_large_n_reliable():
    # 25 values -> p95_unreliable absent
    values = [100.0 + i * 10 for i in range(25)]  # 100, 110, ..., 340
    result = latency_summary(values)
    assert result["n"] == 25
    assert "p95_unreliable" not in result
    # p95 at index ceil(0.95*25)-1 = 24-1 = 23 -> 100+23*10 = 330
    assert result["p95"] == 330.0


# === Fixture: wilson_interval ===

def test_wilson_interval_basic():
    # 10 successes / 20 trials -> ~50% with CI
    low, high = wilson_interval(10, 20)
    assert low is not None
    assert high is not None
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert low < high


def test_wilson_interval_n_zero():
    # n == 0 -> None
    assert wilson_interval(0, 0) is None


# === Fixture: reject_input expectations ===

def test_reject_input_metrics():
    # 2 expected reject, 1 correct (rejected_input), 1 false (read accepted)
    records = [
        EvalRecord(filename="rj.jpg", condition="blurry", expected_outcome="reject_input",
                   true_value=None, tolerance=2.0, status="rejected_input",
                   reading=None, retried=False, attempts=1, latency_ms=100.0,
                   first_pass_reading=None),
        EvalRecord(filename="rj2.jpg", condition="blurry", expected_outcome="reject_input",
                   true_value=None, tolerance=2.0, status="accepted",  # false negative
                   reading=10.0, retried=False, attempts=1, latency_ms=100.0,
                   first_pass_reading=10.0),
        EvalRecord(filename="rd.jpg", condition="clear", expected_outcome="read",
                   true_value=10.0, tolerance=2.0, status="accepted",
                   reading=10.0, retried=False, attempts=1, latency_ms=100.0,
                   first_pass_reading=10.0),
    ]
    m = compute_metrics(records)
    # Correct rejection = 1/2 = 0.5
    assert m["correct_rejection_rate"][0] == 0.5
    # False rejection = 0 (none of the read expected were rejected)
    assert m["false_rejection_count"] == 0
