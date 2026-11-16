import pandas as pd
import streamlit as st

from ui_common import hero, inject_brand_styles, render_result


inject_brand_styles()
hero(
    "Analysis history · this session",
    "Your review trail, kept close.",
    "Analyses are stored only in this Streamlit browser session. They are not written to a database or shared across sessions.",
)

history = st.session_state.get("analysis_history", [])
if not history:
    st.info("This session has no analyses yet.", icon=":material/history:")
    st.page_link("app_pages/analyze.py", label="Create your first analysis", icon=":material/arrow_forward:")
else:
    controls = st.columns([1.4, 0.8])
    with controls[0]:
        query = st.text_input("Search analyses", placeholder="Search filename, text, or ID", type="search", key="history_search")
    with controls[1]:
        verdict_filter = st.selectbox("Prediction", ["All", "REAL", "FAKE", "UNCERTAIN"], key="history_filter")

    filtered = [
        item for item in history
        if (verdict_filter == "All" or item["prediction"] == verdict_filter)
        and (not query.strip() or query.casefold() in " ".join([item["analysis_id"], item["filename"], item["text"]]).casefold())
    ]
    st.caption(f"Showing {len(filtered)} of {len(history)} session analyses")
    if filtered:
        table = pd.DataFrame(
            [
                {
                    "Analysis ID": item["analysis_id"],
                    "File": item["filename"],
                    "Modality": item["modalities"],
                    "Prediction": item["prediction"],
                    "Confidence": item["confidence"],
                    "Date and time": item["timestamp"],
                }
                for item in filtered
            ]
        )
        st.dataframe(
            table,
            hide_index=True,
            alt="Filtered analysis history for this session",
            column_config={"Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent")},
        )
        selected_id = st.selectbox("View analysis details", [item["analysis_id"] for item in filtered], key="history_detail_id")
        selected = next(item for item in filtered if item["analysis_id"] == selected_id)
        st.subheader("Analysis details")
        detail, preview = st.columns([1.05, 0.95], gap="large")
        with detail:
            render_result(selected, show_preview=False)
            st.markdown("**Submitted text**")
            st.write(selected["text"])
        with preview:
            if selected.get("preview_jpeg"):
                st.image(selected["preview_jpeg"], caption=selected["filename"], alt="Image retained in session history", width="stretch")
    else:
        st.warning("No session analyses match these filters.")
