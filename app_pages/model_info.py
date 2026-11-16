from ui_common import MODEL_PATH, hero, inject_brand_styles, read_project_csv
import streamlit as st


inject_brand_styles()
hero(
    "Model card · repository-backed facts",
    "Know what is running under the hood.",
    "This page describes the inference code and artifacts present in the downloaded project. Unknown training details are called out rather than inferred.",
)

st.subheader("Active checkpoint")
if MODEL_PATH.exists():
    st.success(f"{MODEL_PATH.name} · {MODEL_PATH.stat().st_size / (1024**2):.1f} MiB", icon=":material/check_circle:")
else:
    st.error(f"Checkpoint missing: {MODEL_PATH}", icon=":material/error:")

st.markdown("#### Architecture and inference path")
architecture = [
    ("Text encoder", "DistilBERT base uncased; 768-dimensional hidden states projected to 256."),
    ("Vision encoder", "ResNet-50 feature extractor; 2048-dimensional pooled representation projected to 256."),
    ("Fusion", "Text-to-vision and text-to-audio multi-head attention, each with 4 heads; concatenated representation feeds an MLP classifier."),
    ("Classifier", "768-dimensional fused vector → 128 hidden units → 2 output logits. Index 0 is treated as REAL; index 1 as FAKE."),
    ("Audio branch", "An audio projection layer exists, but `predict_content` passes no audio embedding. The model substitutes a zero vector."),
    ("Live modalities", "Text + image; also text + sampled video frames by reusing the image branch. There is no temporal video or audio inference."),
]
for label, value in architecture:
    with st.container(border=True):
        st.markdown(f"**{label}**")
        st.write(value)

st.markdown("#### Preprocessing and output")
preprocess = [
    ("Text", "DistilBERT uncased tokenizer; truncation and padding to at most 128 tokens."),
    ("Image", "RGB conversion; resize to 224 × 224; tensor normalization with ImageNet means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]."),
    ("Prediction", "Softmax over two logits. FAKE probability ≥ 0.70 → FAKE; ≤ 0.30 → REAL; otherwise UNCERTAIN."),
    ("Confidence", "Maximum of REAL and FAKE probabilities. For an UNCERTAIN label, this is not confidence in the uncertain category."),
    ("Explanations", "No grounded feature attribution is emitted. The displayed first tokens are a tokenizer preview only."),
]
st.dataframe(preprocess, hide_index=True, alt="Model preprocessing and output details")

st.markdown("#### Evaluation artifacts included")
dataset_scores = read_project_csv("dataset_evaluation_results.csv")
if not dataset_scores.empty:
    st.dataframe(dataset_scores, hide_index=True, alt="Dataset names and metrics recorded in the saved evaluation CSV")
    st.caption("Dataset names and scores come from the supplied evaluation CSV. The repository does not establish which rows were training data, nor provide sample counts, licensing/provenance, or train/validation/test split details.")
else:
    st.info("No saved dataset evaluation table was found.")

st.markdown("#### Which single and combined input paths can run")
mode_guide = [
    {"Path": "Text only", "Executable here": "No", "Implementation / reason": "A text-only baseline score is listed in ablation_study_results.csv, but its trained baseline checkpoint is not included."},
    {"Path": "Image only", "Executable here": "No", "Implementation / reason": "A vision-only score is listed in the CSV, but no vision-only checkpoint is included. A blank caption would still run the fusion classifier."},
    {"Path": "Text + image", "Executable here": "Yes", "Implementation / reason": "DistilBERT + ResNet-50 + text-to-vision attention + classifier; active checkpoint."},
    {"Path": "Text + video", "Executable here": "Yes, sampled frames", "Implementation / reason": "Evenly sample frames; run the same text + image model on each; report each sampled-frame score and mean probabilities."},
    {"Path": "Video only", "Executable here": "No", "Implementation / reason": "The available classifier expects text and a visual input. No video-only model is included."},
    {"Path": "Image + video without text", "Executable here": "No", "Implementation / reason": "The live classifier also requires the text branch; image and video are both visual inputs."},
    {"Path": "Text + audio / any audio combination", "Executable here": "No", "Implementation / reason": "The model has an audio projection layer but the repository has no audio encoder; current inference passes a zero embedding."},
    {"Path": "Text + image + video", "Executable here": "Yes, visual sample set", "Implementation / reason": "Score the still image and sampled video frames using one shared visual branch; average probabilities. It is not three independent modality encoders."},
]
st.dataframe(mode_guide, hide_index=True, alt="Available single modality and combined inference paths")
st.caption("The video-frame path is an engineering extension of the image inference code. It has no temporal modeling and is not equivalent to a separately trained video detector.")

st.markdown("#### Available checkpoint files")
checkpoint_dir = MODEL_PATH.parent
checkpoints = sorted(checkpoint_dir.glob("*.pt")) if checkpoint_dir.exists() else []
if checkpoints:
    st.dataframe(
        [{"Checkpoint": file.name, "Size (MiB)": round(file.stat().st_size / (1024**2), 1), "Loaded by this app": file == MODEL_PATH} for file in checkpoints],
        hide_index=True,
        alt="PyTorch checkpoint files found in the models folder",
    )

st.warning(
    "The saved project results describe a tri-modal system, while this app's live paths omit audio input and video temporal modeling. "
    "Do not present those tri-modal benchmarks as measured scores for live text + image or frame-sampled video requests.",
    icon=":material/warning:",
)
st.caption("There is no separate REST inference API or database in this repository. Streamlit invokes the Python model engine in-process; history is session-only.")
