import streamlit as st

from ui_common import hero, inject_brand_styles


inject_brand_styles()
hero(
    "Research-grade content verification",
    "Trust the signal. Inspect the evidence.",
    "A multimodal NLP project for reviewing text and image samples or text with sampled video frames using the included classifier. "
    "Explore the model's real outputs, saved evaluation results, and current explainability limits.",
)
st.caption("Project: Multimodal NLP for Deepfake Detection and Explainable Content Verification")

left, right = st.columns([1.2, 0.8], gap="large")
with left:
    st.subheader("A careful second look at synthetic media")
    st.write(
        "Verity brings a claim or caption together with an image or sampled video frames and returns a REAL, FAKE, or UNCERTAIN "
        "assessment. It is a research demonstration: scores are model outputs for human review, not proof of truth."
    )
    analyze, explore = st.columns(2)
    with analyze:
        st.page_link("app_pages/analyze.py", label="Analyze content", icon=":material/arrow_forward:", width="stretch")
    with explore:
        st.page_link("app_pages/overview.py", label="Explore dashboard", icon=":material/space_dashboard:", width="stretch")

with right:
    with st.container(border=True):
        st.markdown("#### Available in this build")
        st.badge("Text + image inference", color="blue", icon=":material/image_search:")
        st.badge("Text + sampled video frames", color="violet", icon=":material/movie:")
        st.badge("REAL / FAKE / UNCERTAIN", color="green", icon=":material/rule:")
        st.badge("Saved experiment results", color="violet", icon=":material/monitoring:")
        st.badge("Local-session history", color="orange", icon=":material/history:")
        st.caption("Temporal video analysis, audio inference, and faithful feature attribution are not implemented in the checked-in model.")

st.subheader("How a sample moves through the system")
steps = st.columns(4)
for col, number, title, note in zip(
    steps,
    ["01", "02", "03", "04"],
    ["Text", "Image", "Fusion", "Review"],
    ["DistilBERT tokens", "ResNet-50 features", "Text–vision attention", "Score + human judgment"],
):
    with col:
        with st.container(border=True, height="stretch"):
            st.caption(number)
            st.markdown(f"#### {title}")
            st.caption(note)

st.info(
    "Text + video is handled by sampling frames and reusing the image encoder. The model does not analyze motion, audio, or lip-sync. "
    "Its audio projection receives zeros because no audio encoder is implemented.",
    icon=":material/info:",
)
