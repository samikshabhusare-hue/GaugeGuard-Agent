from __future__ import annotations

import streamlit as st

from gaugeguard.vision import inspect_image_quality

st.set_page_config(page_title="GaugeGuard Agent", page_icon="🎛️", layout="wide")
st.title("GaugeGuard Agent")
st.caption("Industrial gauge-reading prototype | OpenCV 5 + Python + Streamlit")
st.warning(
    "Prototype only. It is not a certified measuring instrument and must not be used "
    "to control equipment or make safety-critical decisions."
)

uploaded = st.file_uploader("Upload a gauge photo", type=["png", "jpg", "jpeg", "webp"])
if uploaded is None:
    st.info("Upload an image to inspect its basic quality. Gauge dial and needle reading are not implemented yet.")
else:
    image_bytes = uploaded.getvalue()
    left, right = st.columns(2)
    with left:
        st.image(image_bytes, caption="Uploaded image", use_container_width=True)
    with right:
        try:
            quality = inspect_image_quality(image_bytes)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.subheader("Image quality check")
            st.metric("Blur score (Laplacian variance)", f"{quality.blur_score:.1f}")
            st.metric("Contrast score (grayscale standard deviation)", f"{quality.contrast_score:.1f}")
            st.write("Image size:", f"{quality.width} × {quality.height} px")
            st.info("Next milestone: detect this gauge's dial and needle, then apply its calibration profile.")

with st.expander("Project status"):
    st.write("This is a repository scaffold. The gauge-reading and agent retry pipeline are not complete yet.")
    st.code("OpenCV runtime: " + __import__("cv2").__version__)
