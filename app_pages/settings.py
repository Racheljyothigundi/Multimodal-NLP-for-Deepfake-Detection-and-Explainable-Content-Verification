import streamlit as st

from ui_common import MODEL_PATH, hero, inject_brand_styles


inject_brand_styles()
hero(
    "Settings · privacy · responsible use",
    "Keep the review accountable.",
    "Local controls and project guardrails for responsible research use. This build has no user accounts, external verification service, or persistent analysis database.",
)

st.subheader("Inference configuration")
with st.container(border=True):
    st.write(f"**Checkpoint:** `{MODEL_PATH.name}`")
    st.write("**Supported analysis input:** text + image, text + sampled video, or text + image + sampled video")
    st.write("**Decision thresholds:** REAL at fake probability ≤ 0.30 · FAKE at ≥ 0.70 · UNCERTAIN between thresholds")
    st.write("**Threshold controls:** fixed in `model_engine.py`; this screen does not alter the trained model or thresholds")
    st.write("**Runtime:** CUDA when PyTorch reports it is available; otherwise CPU")

st.subheader("Session data")
st.write(
    "Analysis history and compressed image/frame previews are held in the active Streamlit session only. "
    "Uploaded videos are temporarily written to the local OS temp folder for OpenCV decoding and removed after frame sampling. "
    "The app does not save them to the project or a database. A hosted deployment still sends uploads to its server for inference."
)
if st.button("Clear this session’s analysis history", icon=":material/delete_sweep:", type="secondary"):
    st.session_state["analysis_history"] = []
    st.session_state["latest_analysis"] = None
    st.success("Session analysis history cleared.", icon=":material/check:")

st.subheader("Ethical and responsible use")
principles = [
    ("Consent and privacy", "Use media you have permission to analyze. Avoid uploading private or identifying content without consent."),
    ("Provenance and licensing", "Document dataset sources, licenses, labels, consent conditions, and preprocessing before sharing research results."),
    ("Bias and generalization", "Performance can vary by identity, language, source, compression, and manipulation type. The supplied files do not document subgroup evaluations."),
    ("Human review", "Use predictions as triage signals. Do not use a model score alone to accuse a person, establish identity, or make a high-impact decision."),
    ("Responsible demonstration", "Do not use this detector to create, improve, or distribute deceptive media. Report limitations and failure cases alongside the metrics."),
]
for title, detail in principles:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(detail)

st.info("A model can be confidently wrong, especially outside its training distribution. Verify consequential claims with independent, trusted sources.", icon=":material/gpp_maybe:")
