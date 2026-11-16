from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "unique_trimodal_deepfake_detector.pt"
RESULTS_PATH = ROOT / "results"


def inject_brand_styles() -> None:
    """Small brand treatments for the project hero and status chips."""
    st.markdown(
        """
        <style>
        .verity-hero {
            padding: 2.1rem 2.25rem; margin: .3rem 0 1.5rem; border-radius: 22px;
            border: 1px solid rgba(148, 163, 184, .20);
            background: radial-gradient(ellipse at 88% 0%, rgba(79,140,255,.20), transparent 34%),
                        linear-gradient(125deg, #111a2f 0%, #172442 68%, #1d2040 100%);
            box-shadow: 0 18px 50px rgba(0,0,0,.22);
        }
        .verity-eyebrow { color:#8fb5ff; font-size:.75rem; font-weight:700;
            letter-spacing:.16em; text-transform:uppercase; }
        .verity-title { color:#f8fafc; font-size:clamp(2rem,4vw,3.1rem);
            line-height:1.08; letter-spacing:-.045em; margin:.55rem 0 .7rem; }
        .verity-copy { max-width:760px; color:#aebbd1; font-size:1.02rem; line-height:1.65; }
        .verity-card { padding:1rem 1.15rem; border:1px solid rgba(148,163,184,.18);
            background:linear-gradient(145deg,rgba(21,29,50,.92),rgba(17,26,47,.92));
            border-radius:16px; min-height:112px; }
        .verity-card-label { color:#94a3b8; font-size:.75rem; letter-spacing:.08em;
            text-transform:uppercase; }
        .verity-card-value { color:#f8fafc; font-size:1.4rem; font-weight:650; margin-top:.45rem; }
        .verity-card-note { color:#8fa0ba; font-size:.82rem; margin-top:.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(eyebrow: str, title: str, body: str) -> None:
    st.markdown(
        f'<section class="verity-hero"><div class="verity-eyebrow">{eyebrow}</div>'
        f'<h1 class="verity-title">{title}</h1><div class="verity-copy">{body}</div></section>',
        unsafe_allow_html=True,
    )


def card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f'<div class="verity-card"><div class="verity-card-label">{label}</div>'
        f'<div class="verity-card-value">{value}</div><div class="verity-card-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def read_project_csv(filename: str) -> pd.DataFrame:
    path = RESULTS_PATH / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource(show_spinner=False)
def get_model() -> Any:
    # Keep the expensive framework and checkpoint load off pages that only show project details.
    from model_engine import load_deepfake_model

    return load_deepfake_model(str(MODEL_PATH))


def make_preview_jpeg(image: Any, max_size: int = 560) -> bytes:
    preview = image.copy().convert("RGB")
    preview.thumbnail((max_size, max_size))
    buffer = BytesIO()
    preview.save(buffer, format="JPEG", quality=82, optimize=True)
    return buffer.getvalue()


def result_status(prediction: str) -> tuple[str, str]:
    if prediction == "REAL":
        return "green", ":material/verified_user: Content classified as real"
    if prediction == "FAKE":
        return "red", ":material/gpp_bad: Content classified as fake"
    return "orange", ":material/visibility: Uncertain result — human review advised"


def render_result(result: dict[str, Any], show_preview: bool = True) -> None:
    color, message = result_status(result["prediction"])
    st.badge(message, color=color)
    st.markdown(f"### {result['prediction']}")
    left, right = st.columns(2)
    left.metric("Model confidence", f"{result['confidence'] * 100:.1f}%", border=True)
    right.metric("Fake-class probability", f"{result['fake_probability'] * 100:.1f}%", border=True)

    st.markdown("#### Class probability distribution")
    probabilities = pd.DataFrame(
        {"Probability": [result["real_probability"], result["fake_probability"]]},
        index=["REAL", "FAKE"],
    )
    st.bar_chart(probabilities, horizontal=True, x_label="Class", y_label="Probability", alt="Model probability for REAL and FAKE classes")
    st.caption(
        "UNCERTAIN is assigned when fake probability falls between the model's 0.30 and 0.70 thresholds. "
        "The model still returns its two class probabilities."
    )

    if show_preview and result.get("preview_jpeg"):
        st.image(result["preview_jpeg"], caption=result.get("filename", "Analyzed image"), alt="Preview of the image analyzed")

    st.caption(
        f"Analysis ID: {result['analysis_id']} · {result['timestamp']} · "
        f"{result['processing_seconds']:.2f}s · {result['modalities']}"
    )
    with st.container(border=True):
        st.markdown("#### AI model used for this run")
        st.write(f"**Checkpoint:** `{result.get('model_checkpoint', MODEL_PATH.name)}`")
        st.write(f"**AI components:** {result.get('ai_components', 'DistilBERT + ResNet-50 + text-to-vision attention + classifier; audio branch unused.')}")
        st.write(f"**Input path:** {result.get('analysis_path', result['modalities'])}")
        if result.get("aggregation"):
            st.write(f"**Score method:** {result['aggregation']}")
