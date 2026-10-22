from __future__ import annotations

from typing import Any

from gaugeguard.calibration import (
    AgreementResult,
    CalibrationResult,
    GaugeProfile,
    angle_to_value,
    get_profile,
    readings_agree,
)
from gaugeguard.models import AgentStep, ImageQuality, InspectionResult, ReadingEstimate
from gaugeguard.vision import (
    NeedleDetectionResult,
    decode_image,
    detect_dial_and_needle,
    inspect_image_quality,
    preprocess_for_retry,
)


def _is_unverified_demo_profile(profile: GaugeProfile | None) -> bool:
    """Return True if profile is an unverified demonstration profile or uses placeholder units."""
    if profile is None:
        return True
    if not profile.unit or profile.unit.strip().lower() in ("units", "unit", "placeholder"):
        return True
    name_upper = profile.name.upper()
    if name_upper.startswith(("EXAMPLE", "DEMO", "SAMPLE", "MOCK", "TEST_")):
        return True
    return False


def _select_retry_methods(
    quality: ImageQuality,
    initial_detection: NeedleDetectionResult,
) -> list[tuple[str, str]]:
    """Determine evidence-driven alternate preprocessing methods and rationale."""
    methods: list[tuple[str, str]] = []

    # Evidence: Low contrast or marginal confidence favors CLAHE contrast enhancement
    if (
        "low_contrast" in quality.flags
        or quality.contrast_score < 25.0
        or (initial_detection.success and initial_detection.confidence < 0.65)
    ):
        methods.append((
            "clahe",
            f"Image contrast score is {quality.contrast_score:.1f} (or low confidence); "
            "applying CLAHE local contrast enhancement.",
        ))
        methods.append((
            "adaptive_threshold",
            "Secondary fallback: applying Gaussian adaptive thresholding to segment dial geometry.",
        ))
    else:
        # Evidence: Structural detection failure favors adaptive thresholding
        flags_str = (
            ", ".join(initial_detection.quality_flags)
            if initial_detection.quality_flags
            else "unresolved needle"
        )
        methods.append((
            "adaptive_threshold",
            f"Initial pass returned flags [{flags_str}]; "
            "applying Gaussian adaptive thresholding to isolate needle contours under non-uniform illumination.",
        ))
        methods.append((
            "clahe",
            "Secondary fallback: applying CLAHE contrast enhancement.",
        ))

    return methods


def inspect_upload(
    image_bytes: bytes,
    profile: GaugeProfile | None = None,
) -> InspectionResult:
    """Orchestrate evidence-driven gauge inspection with an agentic retry loop."""
    if not image_bytes:
        raise ValueError("The uploaded image is empty.")

    if profile is None:
        profile = get_profile("EXAMPLE_270")

    # Step 1: Pre-flight image quality check
    quality: ImageQuality = inspect_image_quality(image_bytes)
    step1 = AgentStep(
        step_name="image_quality",
        method="opencv_laplacian_and_contrast",
        reason="Evaluate blur variance, contrast standard deviation, and image resolution.",
        output_summary={
            "flags": list(quality.flags),
            "blur_score": round(quality.blur_score, 1),
            "contrast_score": round(quality.contrast_score, 1),
            "width": quality.width,
            "height": quality.height,
        },
    )
    action_log = [step1]
    attempts: list[dict[str, Any]] = []

    # Decode image for vision pipeline
    image = decode_image(image_bytes)

    # Step 2: Initial detection pass (default preprocessing)
    initial_detection: NeedleDetectionResult = detect_dial_and_needle(image)
    step2 = AgentStep(
        step_name="initial_detection",
        method="detect_dial_and_needle_default",
        reason="First-pass dial localization and needle angle estimation.",
        output_summary={
            "attempt": 1,
            "method": "default",
            "success": initial_detection.success,
            "center": initial_detection.center,
            "radius": initial_detection.radius,
            "angle_degrees": initial_detection.angle_deg,
            "confidence": initial_detection.confidence,
            "flags": list(initial_detection.quality_flags),
            "reason": initial_detection.reason,
        },
    )
    action_log.append(step2)
    attempts.append({
        "attempt": 1,
        "method": "default",
        "success": initial_detection.success,
        "angle_deg": initial_detection.angle_deg,
        "confidence": initial_detection.confidence,
        "flags": list(initial_detection.quality_flags),
    })

    active_detection = initial_detection
    active_annotated = (
        initial_detection.annotated_image
        if initial_detection.annotated_image is not None
        else image
    )

    # Evaluate evidence for triggering a retry pass:
    needs_retry = (
        not initial_detection.success
        or initial_detection.angle_deg is None
        or initial_detection.confidence < 0.65
        or "ambiguous_needle" in initial_detection.quality_flags
        or "low_confidence" in initial_detection.quality_flags
    )

    retry_detection: NeedleDetectionResult | None = None

    if needs_retry:
        retry_candidates = _select_retry_methods(quality, initial_detection)
        attempt_idx = 2

        for method_name, method_reason in retry_candidates:
            preprocessed = preprocess_for_retry(image, method_name)
            curr_detection = detect_dial_and_needle(preprocessed)

            retry_step = AgentStep(
                step_name=f"retry_{method_name}",
                method=f"preprocess_for_retry_{method_name}",
                reason=method_reason,
                output_summary={
                    "attempt": attempt_idx,
                    "method": method_name,
                    "success": curr_detection.success,
                    "center": curr_detection.center,
                    "radius": curr_detection.radius,
                    "angle_degrees": curr_detection.angle_deg,
                    "confidence": curr_detection.confidence,
                    "flags": list(curr_detection.quality_flags),
                    "reason": curr_detection.reason,
                },
            )
            action_log.append(retry_step)
            attempts.append({
                "attempt": attempt_idx,
                "method": method_name,
                "success": curr_detection.success,
                "angle_deg": curr_detection.angle_deg,
                "confidence": curr_detection.confidence,
                "flags": list(curr_detection.quality_flags),
            })
            attempt_idx += 1

            if curr_detection.success and curr_detection.angle_deg is not None:
                retry_detection = curr_detection
                if curr_detection.annotated_image is not None:
                    active_annotated = curr_detection.annotated_image
                break

    # Case A: Both initial detection and retry detection succeeded -> compare using readings_agree()
    if (
        initial_detection.success
        and initial_detection.angle_deg is not None
        and retry_detection is not None
        and retry_detection.success
        and retry_detection.angle_deg is not None
    ):
        cal_init = angle_to_value(initial_detection.angle_deg, profile)
        cal_retry = angle_to_value(retry_detection.angle_deg, profile)
        agree_res: AgreementResult = readings_agree(cal_init.value, cal_retry.value, profile)

        cmp_step = AgentStep(
            step_name="reading_comparison",
            method="readings_agree",
            reason="Compare initial reading and retry reading for consistency within profile tolerance.",
            output_summary={
                "initial_value": cal_init.value,
                "retry_value": cal_retry.value,
                "difference": agree_res.difference,
                "tolerance": agree_res.tolerance,
                "agree": agree_res.agree,
                "reason": agree_res.reason,
            },
        )
        action_log.append(cmp_step)

        if not agree_res.agree:
            final_reason = f"Readings disagree: {agree_res.reason}"
            reading = ReadingEstimate(
                value=None,
                unit=profile.unit,
                angle_degrees=None,
                confidence=min(initial_detection.confidence, retry_detection.confidence),
                status="needs_human",
                reason=final_reason,
            )
            dec_step = AgentStep(
                step_name="final_decision",
                method="quality_gated_acceptance",
                reason=final_reason,
                output_summary={
                    "decision": "needs_human",
                    "reason": final_reason,
                    "needs_human": True,
                    "value": None,
                },
            )
            action_log.append(dec_step)
            return InspectionResult(
                reading=reading,
                quality=quality,
                action_log=action_log,
                annotated_image=active_annotated,
                needs_human=True,
                attempts=attempts,
            )
        else:
            # Corroborated! Select pass with higher confidence
            if retry_detection.confidence >= initial_detection.confidence:
                active_detection = retry_detection
                active_cal = cal_retry
            else:
                active_detection = initial_detection
                active_cal = cal_init

    elif retry_detection is not None and retry_detection.success and retry_detection.angle_deg is not None:
        active_detection = retry_detection
        active_cal = angle_to_value(active_detection.angle_deg, profile)
    elif initial_detection.success and initial_detection.angle_deg is not None:
        active_detection = initial_detection
        active_cal = angle_to_value(active_detection.angle_deg, profile)
    else:
        # All passes failed
        final_reason = "All vision detection passes failed to localize dial and needle."
        reading = ReadingEstimate(
            value=None,
            unit=profile.unit,
            angle_degrees=None,
            confidence=0.0,
            status="needs_human" if "ambiguous_needle" in initial_detection.quality_flags else "needs_retake",
            reason=final_reason,
        )
        dec_step = AgentStep(
            step_name="final_decision",
            method="quality_gated_acceptance",
            reason=final_reason,
            output_summary={
                "decision": reading.status,
                "reason": final_reason,
                "needs_human": True,
                "value": None,
            },
        )
        action_log.append(dec_step)
        return InspectionResult(
            reading=reading,
            quality=quality,
            action_log=action_log,
            annotated_image=active_annotated,
            needs_human=True,
            attempts=attempts,
        )

    # Step: Calibration Mapping Record
    cal_flags = list(active_cal.flags)
    if _is_unverified_demo_profile(profile):
        cal_flags.append("unverified_demo_profile")

    cal_step = AgentStep(
        step_name="calibration_mapping",
        method="linear_angle_to_value",
        reason=f"Map detected angle {active_detection.angle_deg:.1f}° to engineering value using profile '{profile.name}'.",
        output_summary={
            "value": active_cal.value,
            "unit": active_cal.unit,
            "in_bounds": active_cal.in_bounds,
            "at_scale_limit": active_cal.at_scale_limit,
            "within_operating_limits": active_cal.within_operating_limits,
            "flags": cal_flags,
            "reason": active_cal.reason,
        },
    )
    action_log.append(cal_step)

    # Determine acceptance vs human verification
    all_flags = (
        list(quality.flags)
        + list(active_detection.quality_flags)
        + [f for f in cal_flags if f != "linear_scale_assumed"]
    )

    if active_cal.value is None or not active_cal.in_bounds:
        final_reason = f"Calibration rejected: {active_cal.reason}"
        status = "needs_human"
        value = None
        needs_human = True
    elif (
        all_flags
        or "outside_operating_limits" in active_cal.flags
        or active_cal.at_scale_limit
        or active_detection.confidence < 0.60
    ):
        status = "needs_human"
        value = active_cal.value
        needs_human = True
        flag_desc = ", ".join(all_flags) if all_flags else active_cal.reason
        final_reason = (
            f"Reading calculated ({active_cal.value:.2f} {active_cal.unit}), "
            f"but human review required: {flag_desc}"
        )
    else:
        status = "accepted"
        value = active_cal.value
        needs_human = False
        final_reason = (
            f"Reading accepted and verified within profile bounds "
            f"({active_cal.value:.2f} {active_cal.unit})."
        )

    reading = ReadingEstimate(
        value=value,
        unit=active_cal.unit,
        angle_degrees=active_detection.angle_deg,
        confidence=active_detection.confidence,
        status=status,
        reason=final_reason,
    )

    dec_step = AgentStep(
        step_name="final_decision",
        method="quality_gated_acceptance",
        reason=final_reason,
        output_summary={
            "decision": status,
            "reason": final_reason,
            "needs_human": needs_human,
            "value": value,
            "unit": reading.unit,
            "confidence": reading.confidence,
            "flags": all_flags,
        },
    )
    action_log.append(dec_step)

    return InspectionResult(
        reading=reading,
        quality=quality,
        action_log=action_log,
        annotated_image=active_annotated,
        needs_human=needs_human,
        attempts=attempts,
    )


