from __future__ import annotations

import cv2
import streamlit as st

from gaugeguard.agent import inspect_upload
from gaugeguard.calibration import PROFILES
from gaugeguard.models import InspectionResult

st.set_page_config(
    page_title="GaugeGuard Agent",
    page_icon="🎛️",
    layout="wide",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-container {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .step-card {
        border-left: 4px solid #0284c7;
        background-color: rgba(2, 132, 199, 0.05);
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Section
st.title("🎛️ GaugeGuard Agent")
st.markdown(
    "**AI-powered vision agent for automated industrial gauge inspection, "
    "quality verification, and self-checking diagnostics.**"
)

# Human-Review & Safety Notice
st.subheader("🛡️ Safety & Human Review Policy")
st.warning(
    "⚠️ **PROTOTYPE USE ONLY:** GaugeGuard is an experimental AI prototype and not a certified "
    "measuring instrument. Never use it to control machinery or make safety-critical decisions. "
    "All physical readings must be independently verified by a qualified human operator.",
    icon="⚠️",
)

# Sidebar for Configuration & Metadata
with st.sidebar:
    st.header("⚙️ Inspection Config")
    profile_names = list(PROFILES.keys())
    selected_profile_name = st.selectbox(
        "Gauge Calibration Profile",
        options=profile_names,
        index=0,
        help="Select the profile for the target gauge model.",
    )
    profile = PROFILES.get(selected_profile_name)
    if profile:
        st.markdown(f"**Target Model:** `{profile.name}`")
        st.markdown(f"**Engineering Unit:** `{profile.unit}`")
        st.markdown(f"**Scale Range:** `{profile.minimum_value}` – `{profile.maximum_value} {profile.unit}`")
        st.markdown(f"**Angle Sweep:** `{profile.minimum_angle_degrees}°` → `{profile.maximum_angle_degrees}°`")

    st.divider()
    st.subheader("Runtime Environment")
    st.markdown(f"- **OpenCV Version:** `{cv2.__version__}`")
    st.markdown(f"- **Streamlit Version:** `{st.__version__}`")
    st.caption("OpenCV dial/needle detection and agentic retry loop active.")

# Image Upload
uploaded_file = st.file_uploader(
    "Upload a gauge photo for inspection",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is None:
    st.info("👆 Upload a gauge photo above and click **Analyze Gauge** to begin automated inspection.")
else:
    image_bytes = uploaded_file.getvalue()

    # Track upload changes in session state
    if st.session_state.get("current_image_name") != uploaded_file.name:
        st.session_state["current_image_name"] = uploaded_file.name
        st.session_state["analyzed"] = False
        st.session_state["inspection_result"] = None

    # Analyze Button
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("🔍 Analyze Gauge", type="primary", use_container_width=True):
            try:
                result: InspectionResult = inspect_upload(image_bytes, profile=profile)
                st.session_state["inspection_result"] = result
                st.session_state["analyzed"] = True
            except ValueError as exc:
                st.error(f"Image Error: {exc}")
                st.session_state["analyzed"] = False
                st.session_state["inspection_result"] = None
            except Exception as exc:
                st.error(f"Inspection Error: {exc}")
                st.session_state["analyzed"] = False
                st.session_state["inspection_result"] = None

    is_analyzed = st.session_state.get("analyzed", False)
    result: InspectionResult | None = st.session_state.get("inspection_result")

    # Side-by-Side Images Section
    st.subheader("📷 Image Comparison")
    col_orig, col_analyzed = st.columns(2)

    with col_orig:
        st.markdown("##### Original Image")
        st.image(image_bytes, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

    with col_analyzed:
        st.markdown("##### Analyzed Image")
        if is_analyzed and result:
            if result.annotated_image is not None:
                # Convert BGR/Gray/BGRA array to RGB for accurate Streamlit color display
                ann = result.annotated_image
                if hasattr(ann, "shape"):
                    if len(ann.shape) == 2:
                        annotated_rgb = cv2.cvtColor(ann, cv2.COLOR_GRAY2RGB)
                    elif len(ann.shape) == 3 and ann.shape[2] == 4:
                        annotated_rgb = cv2.cvtColor(ann, cv2.COLOR_BGRA2RGB)
                    else:
                        annotated_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                else:
                    annotated_rgb = ann
                st.image(
                    annotated_rgb,
                    caption="Analyzed View (Dial Circle & Needle Ray Overlay)",
                    use_container_width=True,
                )
            else:
                st.image(image_bytes, caption="Analyzed View (No Overlay Available)", use_container_width=True)
        else:
            st.info("Click **🔍 Analyze Gauge** above to process this image.")

    # Result Cards Section
    st.subheader("📊 Inspection Results")
    rc1, rc2, rc3, rc4 = st.columns(4)

    with rc1:
        if is_analyzed and result and result.reading.value is not None:
            reading_str = f"{result.reading.value:.1f}"
            delta_str = (
                f"Angle: {result.reading.angle_degrees:.1f}°"
                if result.reading.angle_degrees is not None
                else None
            )
            delta_color = "normal"
            caption_str = f"Status: `{result.reading.status}`"
        else:
            reading_str = "--"
            delta_str = "No Valid Reading" if is_analyzed else "Pending Analysis"
            delta_color = "off"
            caption_str = "🔒 *Numeric reading unavailable*"

        st.metric(
            label="Gauge Reading",
            value=reading_str,
            delta=delta_str,
            delta_color=delta_color,
            help="Calculated engineering reading from detected needle angle.",
        )
        st.caption(caption_str)

    with rc2:
        unit_display = (
            result.reading.unit
            if (is_analyzed and result and result.reading.unit)
            else (profile.unit if profile else "--")
        )
        st.metric(
            label="Engineering Unit",
            value=unit_display,
            delta=profile.name if profile else "No Profile",
            delta_color="off",
            help="Configured physical unit.",
        )
        st.caption(f"Profile: `{selected_profile_name}`")

    with rc3:
        if is_analyzed and result:
            if result.reading.status == "accepted":
                op_status = "Accepted"
                op_delta = "In Bounds"
                op_color = "normal"
            elif result.reading.status == "needs_human":
                op_status = "Review Required"
                op_delta = "Escalated"
                op_color = "inverse"
            elif result.reading.status == "needs_retake":
                op_status = "Needs Retake"
                op_delta = "Quality Failed"
                op_color = "inverse"
            else:
                op_status = result.reading.status.replace("_", " ").title()
                op_delta = "Action Needed"
                op_color = "off"
            reason_text = result.reading.reason
        else:
            op_status = "Standby"
            op_delta = "Not Analyzed"
            op_color = "off"
            reason_text = "Awaiting inspection"

        st.metric(
            label="Operating Status",
            value=op_status,
            delta=op_delta,
            delta_color=op_color,
            help="Operating condition determined by vision and calibration bounds.",
        )
        st.caption(f"Reason: {reason_text}")

    with rc4:
        if is_analyzed and result:
            conf_val = (
                f"{result.reading.confidence * 100:.0f}%"
                if result.reading.confidence > 0
                else "--"
            )
            # Collect unique quality flags across steps
            flags_collected: list[str] = []
            for step in result.action_log:
                flags_collected.extend(step.output_summary.get("flags", []))
            seen_flags: set[str] = set()
            unique_flags = [f for f in flags_collected if not (f in seen_flags or seen_flags.add(f))]
            flag_summary = ", ".join(unique_flags) if unique_flags else "Checks Passed"
            q_color = "normal" if not unique_flags else "inverse"

            st.metric(
                label="Detection Quality",
                value=f"Conf: {conf_val}",
                delta=flag_summary[:22] + ("..." if len(flag_summary) > 22 else ""),
                delta_color=q_color,
                help="Vision confidence score and active inspection flags.",
            )
            st.caption(f"Active Flags: `{flag_summary}`")
        else:
            st.metric(
                label="Detection Quality",
                value="Unchecked",
                delta="Ready to Scan",
                delta_color="off",
                help="Image quality scores and detection confidence.",
            )
            st.caption("Awaiting Image Analysis")

    # Dynamic Escalation Notice based on Analysis
    if is_analyzed and result:
        if result.needs_human or result.reading.status != "accepted":
            st.error(
                f"🚨 **HUMAN VERIFICATION REQUIRED:** {result.reading.reason} "
                "Automated reading cannot be safely accepted without independent verification by a qualified technician.",
                icon="🚨",
            )
        else:
            st.success(
                f"✅ **Inspection Passed:** Needle detected at {result.reading.angle_degrees:.1f}° "
                f"and verified within calibrated bounds ({result.reading.value:.2f} {result.reading.unit}).",
                icon="✅",
            )

    # Agent Activity & Retry History Section
    st.subheader("🤖 Agent Activity & Execution Trace")
    act_col1, act_col2 = st.columns([3, 2])

    with act_col1:
        st.markdown("##### 📋 Inspection Steps")
        if is_analyzed and result and result.action_log:
            for idx, step in enumerate(result.action_log, 1):
                st.markdown(
                    f"""
                    <div class="step-card">
                        <strong>Step {idx}: {step.step_name}</strong><br/>
                        <span style="font-size: 0.85em; color: #64748b;">Method: <code>{step.method}</code></span><br/>
                        <span>{step.reason}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander(f"Step {idx} Details & Outputs", expanded=False):
                    st.json(step.output_summary)
        else:
            st.info("No inspection steps executed yet. Click **🔍 Analyze Gauge** to trigger the agent.")

    with act_col2:
        st.markdown("##### 🔄 Retry History")
        if is_analyzed and result:
            retry_steps = [s for s in result.action_log if s.step_name.startswith("retry_")]
            cmp_step = next((s for s in result.action_log if s.step_name == "reading_comparison"), None)
            dec_step = next((s for s in result.action_log if s.step_name == "final_decision"), None)

            st.markdown(f"**Retry Passes:** `{len(retry_steps)} / 2`")

            if not retry_steps:
                st.success("First-pass detection was sufficiently confident; no retry pass was triggered.")
            else:
                for r_step in retry_steps:
                    method_used = r_step.output_summary.get("method", r_step.method)
                    r_success = r_step.output_summary.get("success", False)
                    r_status = "✅ Succeeded" if r_success else "❌ Failed"
                    st.write(f"- **{r_step.step_name}** (`{method_used}`): {r_status}")
                    st.caption(f"Trigger: {r_step.reason}")

            if cmp_step:
                is_agree = cmp_step.output_summary.get("agree", False)
                diff_val = cmp_step.output_summary.get("difference")
                tol_val = cmp_step.output_summary.get("tolerance")
                diff_str = f"{diff_val:.2f}" if diff_val is not None else "--"
                tol_str = f"{tol_val:.2f}" if tol_val is not None else "--"

                if is_agree:
                    st.success(
                        f"🤝 **Corroborated:** {cmp_step.output_summary.get('reason')} "
                        f"(Diff: {diff_str} / Tol: {tol_str})"
                    )
                else:
                    st.error(
                        f"⚠️ **Disagreement:** {cmp_step.output_summary.get('reason')} "
                        f"(Diff: {diff_str} / Tol: {tol_str})"
                    )

            if dec_step:
                st.info(f"**Final Decision:** `{dec_step.output_summary.get('decision')}` — {dec_step.reason}")

            with st.expander("Fallback Preprocessing Methods", expanded=False):
                st.markdown(
                    "- `clahe`: Contrast Limited Adaptive Histogram Equalization\n"
                    "- `adaptive_threshold`: Gaussian adaptive binary thresholding\n\n"
                    "*Available in `gaugeguard.vision.preprocess_for_retry` for the agent retry loop.*"
                )
        else:
            st.caption("Retry telemetry will appear after image analysis.")



with st.expander("ℹ️ Project Status & Diagnostics"):
    st.write("This application is the user-facing dashboard for the GaugeGuard Agent prototype.")
    st.write("Current implementation: Upload interface, OpenCV dial & needle detection, evidence-based retry loop, calibration bounds gating, and audit trace.")
    st.code(f"OpenCV runtime: {cv2.__version__}")

