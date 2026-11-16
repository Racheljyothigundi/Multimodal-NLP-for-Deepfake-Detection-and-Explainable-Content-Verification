from datetime import datetime
from time import perf_counter
from uuid import uuid4

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from ui_common import MODEL_PATH, get_model, hero, inject_brand_styles, make_preview_jpeg, read_project_csv, render_result


inject_brand_styles()
hero(
    "New analysis · multimodal workspace",
    "Choose the inputs. See the AI path.",
    "Run the included trained checkpoint on text + image, text + sampled video, or text + image + sampled video. Every report names the model and AI components used.",
)

modes = ["Text + image", "Text + video", "Text + image + video"]
mode = st.segmented_control("Input combination", modes, default=modes[0], key="analysis_mode") or modes[0]
needs_image = mode in {"Text + image", "Text + image + video"}
needs_video = mode in {"Text + video", "Text + image + video"}
frame_count = 0
if needs_video:
    st.info(
        "Video is screened by sampling frames and reusing the image branch. The model does not analyze motion, audio, or lip-sync, "
        "and this frame workflow has not been separately validated as a video detector.",
        icon=":material/movie:",
    )
    frame_count = st.slider("Frames to sample across the video", min_value=2, max_value=12, value=8, step=2)

with st.container(border=True):
    st.markdown("#### Sample input")
    st.badge(f"Selected path: {mode}", color="blue", icon=":material/layers:")
    with st.form("sample_analysis", clear_on_submit=False):
        text = st.text_area(
            "Claim, headline, or caption",
            placeholder="Add the text associated with your media…",
            height=125,
            max_chars=12000,
            help="DistilBERT tokenizes this text and truncates it to at most 128 tokens.",
        )
        image_upload = None
        video_upload = None
        if needs_image:
            image_upload = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"image_upload_{mode}",
                help="Accepted formats: JPG, JPEG, PNG, WEBP.",
            )
        if needs_video:
            video_upload = st.file_uploader(
                "Upload a video",
                type=["mp4", "mov", "avi", "mkv"],
                key=f"video_upload_{mode}",
                help="MP4 is recommended. MOV, AVI, and MKV support depends on installed OpenCV codecs.",
            )
        submitted = st.form_submit_button("Run verification", type="primary", icon=":material/search:", width="stretch")

uploaded_image = None
image_bytes = b""
video_bytes = video_upload.getvalue() if video_upload is not None else b""
media_error = None
if image_upload is not None:
    try:
        image_bytes = image_upload.getvalue()
        uploaded_image = Image.open(image_upload).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        media_error = f"The image could not be decoded: {exc}"
        st.error(media_error, icon=":material/broken_image:")

previews = st.columns(2 if needs_image and needs_video else 1)
if uploaded_image is not None:
    with previews[0]:
        st.image(uploaded_image, caption=image_upload.name, alt="Uploaded still image", width="stretch")
        st.caption(f"Still image · {uploaded_image.width} × {uploaded_image.height} · {len(image_bytes) / (1024 * 1024):.2f} MB")
if video_upload is not None:
    preview_col = previews[1] if needs_image else previews[0]
    with preview_col:
        st.video(video_bytes, format=video_upload.type)
        st.caption(f"Video · {video_upload.name} · {len(video_bytes) / (1024 * 1024):.2f} MB")

if submitted:
    if not text.strip():
        st.warning("Add the text associated with the media to continue.", icon=":material/edit_note:")
    elif (needs_image and uploaded_image is None) or (needs_video and not video_bytes) or media_error:
        st.warning("Upload all required, readable media files to continue.", icon=":material/upload:")
    elif not MODEL_PATH.exists():
        st.error(f"The model checkpoint was not found at {MODEL_PATH}.", icon=":material/error:")
    else:
        status = st.status("Preparing model inference…", expanded=True)
        started = perf_counter()
        try:
            status.write(f"Loading checkpoint: {MODEL_PATH.name}")
            model = get_model()
            if not needs_video:
                status.write("Running DistilBERT + ResNet-50 fusion on the uploaded image.")
                from model_engine import predict_content

                output = predict_content(model, text, uploaded_image)
                representative_image = uploaded_image
                frame_scores = []
                used_mode = "Text + image"
                aggregation = "One image prediction; no temporal aggregation."
                duration = None
                analysis_path = "Text + image → DistilBERT + ResNet-50 → text-to-vision attention → classifier."
                filenames = image_upload.name
            else:
                status.write(f"Sampling up to {frame_count} video frames and scoring each with the text + image model.")
                from model_engine import predict_video_content

                progress_slot = st.empty()
                progress = progress_slot.progress(0, text="Sampling video frames…")

                def update_progress(current: int, total: int) -> None:
                    progress.progress(current / total, text=f"Scoring visual sample {current} of {total}")

                output = predict_video_content(
                    model,
                    text,
                    video_bytes,
                    video_filename=video_upload.name,
                    max_frames=frame_count,
                    additional_images=[uploaded_image] if uploaded_image is not None else None,
                    progress_callback=update_progress,
                )
                progress.empty()
                representative_image = uploaded_image if uploaded_image is not None else output["representative_frame"]
                frame_scores = output["frames"]
                if uploaded_image is not None:
                    used_mode = f"Text + image + video ({output['sampled_frame_count'] - 1} video frames)"
                    analysis_path = "Caption + still image + sampled video frames → shared ResNet-50 visual branch → mean class probabilities."
                    filenames = f"{image_upload.name} + {video_upload.name}"
                else:
                    used_mode = f"Text + video ({output['sampled_frame_count']} sampled frames)"
                    analysis_path = "Caption + uniformly sampled video frames → shared ResNet-50 visual branch → mean class probabilities."
                    filenames = video_upload.name
                aggregation = "Mean of probabilities across the uploaded still image (if present) and sampled video frames, then the checkpoint's 0.30 / 0.70 thresholds."
                duration = output["video_duration_seconds"]

            elapsed = perf_counter() - started
            record = {
                "analysis_id": uuid4().hex[:10].upper(),
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "filename": filenames,
                "modalities": used_mode,
                "text": text.strip(),
                "prediction": output["prediction"],
                "confidence": output["confidence"],
                "fake_probability": output["fake_probability"],
                "real_probability": output["real_probability"],
                "keywords": output["keywords"],
                "processing_seconds": elapsed,
                "preview_jpeg": make_preview_jpeg(representative_image),
                "model_checkpoint": MODEL_PATH.name,
                "ai_components": "DistilBERT text encoder + ResNet-50 visual encoder + 4-head text-to-vision attention + trained classifier; audio branch unused.",
                "analysis_path": analysis_path,
                "aggregation": aggregation,
                "frame_scores": frame_scores,
                "video_duration_seconds": duration,
            }
            st.session_state["latest_analysis"] = record
            st.session_state["analysis_history"].insert(0, record)
            status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Analysis could not be completed", state="error", expanded=True)
            st.error(
                "The model could not process this sample. For video, verify that OpenCV can decode the selected codec; "
                "the uploaded video is temporarily written for local decoding and removed afterward.",
                icon=":material/error:",
            )
            st.exception(exc)

latest = st.session_state.get("latest_analysis")
if latest:
    st.subheader("Verification report")
    result_col, sample_col = st.columns([1.05, 0.95], gap="large")
    with result_col:
        render_result(latest, show_preview=False)
    with sample_col:
        st.markdown("#### Analyzed media")
        if latest.get("preview_jpeg"):
            st.image(latest["preview_jpeg"], caption=f"Representative visual sample · {latest['filename']}", alt="Image or representative video frame processed by the AI", width="stretch")
        with st.expander("Submitted text"):
            st.write(latest["text"])

    if latest.get("frame_scores"):
        st.markdown("#### Per-sample scores")
        frame_table = pd.DataFrame(latest["frame_scores"])[
            ["sample_source", "frame_index", "timestamp_seconds", "prediction", "real_probability", "fake_probability"]
        ]
        frame_table["timestamp_seconds"] = frame_table["timestamp_seconds"].map(lambda value: f"{value:.2f}s" if value is not None else "—")
        st.dataframe(frame_table, hide_index=True, alt="REAL and FAKE probabilities for each image and sampled video frame")

    st.info("This is an AI screening result for research review, not proof of authenticity, manipulation, identity, or event accuracy.", icon=":material/gpp_maybe:")

with st.expander("Single and combined input guide"):
    st.dataframe(
        pd.DataFrame(
            [
                {"Input combination": "Text only", "Can run now?": "No", "Method / constraint": "CSV reports a baseline metric, but its model checkpoint is not included."},
                {"Input combination": "Image only", "Can run now?": "No", "Method / constraint": "CSV reports a vision baseline, but its checkpoint is not included. Blank text would still use the fusion model."},
                {"Input combination": "Text + image", "Can run now?": "Yes", "Method / constraint": "DistilBERT + ResNet-50 + text-to-vision attention; one image per run."},
                {"Input combination": "Text + video", "Can run now?": "Yes, frame screening", "Method / constraint": "Uniform frame sampling → same text + image model per frame → mean class probabilities. No temporal/audio analysis."},
                {"Input combination": "Text + image + video", "Can run now?": "Yes, visual sample set", "Method / constraint": "Score the still image and sampled video frames through the same visual branch, then average scores. Not three independent encoders."},
                {"Input combination": "Any audio combination", "Can run now?": "No", "Method / constraint": "No audio feature extractor is included; the audio branch receives zeros."},
            ]
        ),
        hide_index=True,
        alt="Single and combined input modes, their model availability, and processing method",
    )
    st.markdown("#### Saved single-modality baseline figures")
    st.caption("These metrics are read from the supplied evaluation CSV. They are reference results only; the baseline checkpoints are not included and these are not new per-upload predictions.")
    ablation = read_project_csv("ablation_study_results.csv")
    if not ablation.empty:
        single = ablation[ablation["Model Variant"].str.contains("Single-Modality", na=False)]
        if not single.empty:
            st.dataframe(single, hide_index=True, alt="Saved text-only and vision-only baseline metrics")
        else:
            st.info("No single-modality rows were found in the ablation CSV.")
    else:
        st.info("No ablation CSV is available.")
