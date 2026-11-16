from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Verity | Deepfake intelligence",
    page_icon=":material/shield_lock:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("analysis_history", [])
st.session_state.setdefault("latest_analysis", None)

with st.sidebar:
    st.markdown("### :material/shield_lock: VERITY")
    st.caption("Multimodal content intelligence")
    st.markdown(":green-badge[MODEL ARTIFACT FOUND]" if (ROOT / "models" / "unique_trimodal_deepfake_detector.pt").exists() else ":red-badge[MODEL ARTIFACT MISSING]")

pages = {
    "Workspace": [
        st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True,
                help="Project welcome page and direct links to analysis and the overview."),
        st.Page("app_pages/overview.py", title="Overview", icon=":material/dashboard:",
                help="Current-session activity and the evaluation figures saved with this project."),
        st.Page("app_pages/analyze.py", title="New analysis", icon=":material/biotech:",
                help="Run the downloaded checkpoint on a text and image sample."),
        st.Page("app_pages/history.py", title="Analysis history", icon=":material/history:",
                help="Search and inspect analyses created during this browser session."),
    ],
    "Research & model": [
        st.Page("app_pages/experiments.py", title="Experiments & cases", icon=":material/monitoring:",
                help="Inspect saved ablations, dataset results, plots, and difficult-case records."),
        st.Page("app_pages/explainability.py", title="Explainability", icon=":material/troubleshoot:",
                help="Review what explanation signals exist and which methods are not implemented."),
        st.Page("app_pages/model_info.py", title="Model information", icon=":material/model_training:",
                help="Architecture, preprocessing, inputs, outputs, and evidence available in the repository."),
        st.Page("app_pages/settings.py", title="Settings & ethics", icon=":material/tune:",
                help="Session data controls, model thresholds, and responsible-use notes."),
    ],
}

page = st.navigation(pages, position="sidebar")
page.run()
