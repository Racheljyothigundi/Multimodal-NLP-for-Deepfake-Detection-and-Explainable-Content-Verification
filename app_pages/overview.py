import pandas as pd
import streamlit as st

from ui_common import card, hero, inject_brand_styles, read_project_csv


inject_brand_styles()
history = st.session_state.get("analysis_history", [])
total = len(history)
real_count = sum(item["prediction"] == "REAL" for item in history)
fake_count = sum(item["prediction"] == "FAKE" for item in history)
uncertain_count = sum(item["prediction"] == "UNCERTAIN" for item in history)
average_confidence = sum(item["confidence"] for item in history) / total if total else None

hero(
    "Workspace overview",
    "A clear view of the evidence.",
    "Session activity comes from analyses run in this browser session. Research metrics below come from the saved project CSVs.",
)

st.subheader("This session")
metric_cols = st.columns(4)
with metric_cols[0]:
    card("Samples analyzed", str(total), "Current browser session only")
with metric_cols[1]:
    card("Classified real", str(real_count), "Model output")
with metric_cols[2]:
    card("Classified fake", str(fake_count), f"{uncertain_count} uncertain" if total else "No session predictions yet")
with metric_cols[3]:
    card("Average confidence", f"{average_confidence:.1%}" if average_confidence is not None else "—", "Current session only")

st.subheader("Reported experiment snapshot")
comparison = read_project_csv("ablation_study_results.csv")
if comparison.empty:
    st.info("No saved ablation results CSV was found in results/.")
else:
    tri = comparison[comparison["Model Variant"].str.contains("Tri-Modal System", na=False)]
    if not tri.empty:
        row = tri.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Reported accuracy", f"{row['Accuracy']:.1%}", border=True)
        cols[1].metric("Reported precision", f"{row['Precision']:.1%}", border=True)
        cols[2].metric("Reported recall", f"{row['Recall']:.1%}", border=True)
        cols[3].metric("Reported F1", f"{row['F1-Score']:.1%}", border=True)
    chart_columns = [c for c in ["Accuracy", "F1-Score", "ROC-AUC"] if c in comparison.columns]
    if chart_columns:
        st.bar_chart(
            comparison.set_index("Model Variant")[chart_columns],
            horizontal=True,
            x_label="Model variant",
            y_label="Reported score",
            alt="Saved accuracy, F1, and ROC-AUC scores for model variants",
        )
    with st.expander("Read the complete saved comparison"):
        st.dataframe(comparison, hide_index=True, alt="Ablation study results saved in the project")

dataset_results = read_project_csv("dataset_evaluation_results.csv")
if not dataset_results.empty:
    st.subheader("Dataset-wise reported results")
    st.dataframe(dataset_results, hide_index=True, alt="Evaluation metrics recorded for each dataset source")

if total:
    st.subheader("Recent activity")
    activity = pd.DataFrame(history[:6])[["timestamp", "filename", "modalities", "prediction", "confidence"]]
    activity["confidence"] = activity["confidence"].map(lambda value: f"{value:.1%}")
    st.dataframe(activity, hide_index=True, alt="Most recent content analyses in this browser session")
else:
    st.subheader("Recent activity")
    st.info("No samples have been analyzed in this session yet. Start with a text + image sample.", icon=":material/hourglass_empty:")
    st.page_link("app_pages/analyze.py", label="Start an analysis", icon=":material/arrow_forward:")

st.caption("Saved benchmark CSVs are displayed as supplied. Their evaluation splits, sample counts, and provenance are not documented in this app, so the figures are not independently verified here.")
