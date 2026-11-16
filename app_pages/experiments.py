import streamlit as st

from ui_common import RESULTS_PATH, hero, inject_brand_styles, read_project_csv


inject_brand_styles()
hero(
    "Research workspace · saved artifacts",
    "Compare, ablate, and inspect failure cases.",
    "Tables and figures below are loaded from the CSV and PNG files included in this project folder. The app does not recompute these experiments.",
)

st.subheader("Experiment 1 · single-modality baselines and multimodal system")
ablation = read_project_csv("ablation_study_results.csv")
if not ablation.empty:
    st.dataframe(ablation, hide_index=True, alt="Saved baseline and modality ablation metrics")
    chart_cols = [column for column in ["Accuracy", "F1-Score", "ROC-AUC"] if column in ablation.columns]
    if chart_cols:
        st.bar_chart(
            ablation.set_index("Model Variant")[chart_cols], horizontal=True,
            x_label="Model variant", y_label="Reported score",
            alt="Reported accuracy, F1, and ROC-AUC for baselines and ablations",
        )
    st.download_button(
        "Download ablation results CSV", ablation.to_csv(index=False),
        file_name="ablation_study_results.csv", mime="text/csv", icon=":material/download:",
    )
else:
    st.info("The ablation_study_results.csv file was not found.")

st.subheader("Experiment 3 · saved dataset-wise evaluation")
datasets = read_project_csv("dataset_evaluation_results.csv")
if not datasets.empty:
    st.dataframe(datasets, hide_index=True, alt="Saved evaluation metrics by dataset source")
else:
    st.info("The dataset_evaluation_results.csv file was not found.")

st.subheader("Experiment artifacts")
plot_cols = st.columns(2)
for column, filename, label, alt in [
    (plot_cols[0], "confusion_matrices_all_models.png", "Confusion matrices", "Saved confusion matrix plot for model variants"),
    (plot_cols[1], "roc_pr_auc_curves.png", "ROC and PR curves", "Saved ROC and precision-recall curves"),
]:
    path = RESULTS_PATH / filename
    with column:
        if path.exists():
            st.markdown(f"#### {label}")
            st.image(str(path), alt=alt, width="stretch")
        else:
            st.info(f"{filename} is not present.")

st.subheader("Difficult-case analysis · saved records")
cases = read_project_csv("results_difficult_cases_analysis.csv")
if not cases.empty:
    st.caption(f"{len(cases)} records are present in the supplied CSV. The file contains text samples and short notes, not the original media or per-case model probabilities.")
    st.dataframe(cases, hide_index=True, alt="Difficult-case records provided with this project")
    st.download_button(
        "Download difficult-case CSV", cases.to_csv(index=False),
        file_name="results_difficult_cases_analysis.csv", mime="text/csv", icon=":material/download:",
    )
else:
    st.info("The results_difficult_cases_analysis.csv file was not found.")

st.subheader("Research coverage visible in this repository")
coverage = [
    ("Baseline vs multimodal", "Saved ablation CSV includes text-only, vision-only, and fused/ablated variants."),
    ("Fusion strategy comparison", "The supplied CSV does not document two separately evaluated fusion architectures."),
    ("Generalization", "Dataset-wise figures are present; split protocol and cross-dataset setup are not documented here."),
    ("Modality ablation", "Ablation rows are present in the supplied CSV."),
    ("Sanity checks", "The saved difficult-case CSV mentions semantic mismatch and misleading captions, but does not document a controlled genuine/fake/mismatched-pair protocol."),
    ("Difficult cases", "A 15-row case-analysis CSV is present; it does not include original media or detailed predictions."),
    ("Explanation quality", "No faithfulness/localization evaluation or implemented attribution method is present in inference code."),
    ("Dataset scope", "The brief recommends at least 300 labeled samples; the supplied files do not state the dataset sample count or class balance."),
]
for title, detail in coverage:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(detail)

st.caption("CSV values are presented as project artifacts, not independently verified metrics. Dataset licenses, sample counts, splits, and provenance should be documented alongside the final research report.")
