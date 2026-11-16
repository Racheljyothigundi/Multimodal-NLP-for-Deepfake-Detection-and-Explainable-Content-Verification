import streamlit as st

from ui_common import hero, inject_brand_styles


inject_brand_styles()
hero(
    "Explainability · evidence status",
    "Show what the model can support.",
    "This page distinguishes model outputs from explanations. Only evidence actually emitted by the checked-in inference code is presented as available.",
)

st.warning(
    "Faithful explainability is not implemented in the current inference path. No Grad-CAM, image heatmap, frame localization, "
    "audio visualization, calibrated modality contribution, or grounded explanation is generated.",
    icon=":material/visibility_off:",
)

left, right = st.columns(2, gap="large")
with left:
    st.subheader("Available signal")
    st.badge("Tokenizer preview only", color="orange", icon=":material/notes:")
    latest = st.session_state.get("latest_analysis")
    if latest:
        st.markdown("**Input tokens displayed by the demo**")
        tokens = latest.get("keywords", [])
        st.code(" · ".join(tokens) if tokens else "No token preview returned")
        st.caption("These are the first few tokens from the input after tokenization. They are not feature importance, a rationale, or causal evidence for the prediction.")
        with st.expander("Associated input text"):
            st.write(latest["text"])
    else:
        st.info("Run an analysis to see the limited tokenizer preview produced by this version.")

with right:
    st.subheader("Not emitted by the model")
    unavailable = [
        ("Image regions", "No Grad-CAM or suspicious-region heatmap is produced."),
        ("Attention evidence", "The inference code discards attention weights instead of returning them."),
        ("Audio / video evidence", "There is no audio feature extraction or video-frame inference path."),
        ("Modality contribution", "No per-modality score or ablation is computed for an individual uploaded sample."),
    ]
    for label, note in unavailable:
        with st.container(border=True):
            st.markdown(f"**{label}**")
            st.caption(note)

st.info(
    "The result page shows the model's REAL and FAKE probabilities and its uncertainty threshold. "
    "A probability describes the classifier output; it does not explain which content evidence caused that output.",
    icon=":material/info:",
)
