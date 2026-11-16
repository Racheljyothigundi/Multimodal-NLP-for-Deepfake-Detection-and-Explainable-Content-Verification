# Verity — multimodal deepfake detection

Streamlit research dashboard for the trained model checkpoint and evaluation artifacts in this project.

## Run on Windows

Open PowerShell in the project folder:

```powershell
cd "C:\Users\rachel jyothi\Downloads\Multimodal_Deepfake_Project"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`. Stop the server with **Ctrl+C**.

If PowerShell blocks environment activation, run the environment directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The first analysis may need internet access to download pretrained DistilBERT and ResNet components. The app uses `models/unique_trimodal_deepfake_detector.pt`; keep `models/` and `results/` beside `app.py`.

## What is implemented

- Home, session overview, text + image, text + sampled-video, or text + image + sampled-video inference, session history, saved experiments and difficult cases, explainability status, model information, and settings/ethics pages.
- Live prediction runs through the included checkpoint. There is no separate REST API or database.
- Analysis history and compressed image previews remain in the current Streamlit session only.
- Evaluation CSVs and figures are shown as supplied; the app does not retrain the model or recalculate benchmark metrics.

## Model limits shown in the UI

The image path accepts a text claim/caption plus one JPG, JPEG, PNG, or WEBP image. The video path samples frames uniformly, runs the same text + image checkpoint on each frame, then averages REAL/FAKE probabilities. You can also add one still image to the text + video run; its score is included in that visual-sample average. This is frame screening, not a temporal video detector: it does not analyze motion, audio, or lip-sync and has not been separately evaluated. Video is temporarily written to the OS temp folder for OpenCV decoding, then removed.

Text-only and image-only scores appear in the supplied ablation CSV, but the corresponding baseline model checkpoints are not included. The app does not substitute blank or synthetic modalities and call that single-modality inference. The architecture has an audio projection layer, but inference supplies zeros because no audio feature extractor exists. The UI reports REAL/FAKE class probabilities and the model's 0.30 / 0.70 uncertainty thresholds.

No Grad-CAM, image heatmap, audio/video explanation, or feature attribution is generated. The displayed tokenizer tokens are not a rationale. The saved tri-modal metrics should not be presented as live text + image inference metrics. Dataset splits, sample counts, class balance, provenance, and license details are not documented in the supplied evaluation files.
