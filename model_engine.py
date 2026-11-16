import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from transformers import DistilBertModel, DistilBertTokenizerFast, CLIPModel, CLIPProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UniqueTriModalDeepfakeSystem(nn.Module):
    def __init__(self, embed_dim=256):
        super(UniqueTriModalDeepfakeSystem, self).__init__()
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        resnet = models.resnet50(pretrained=True)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        self.text_proj = nn.Sequential(nn.Linear(768, embed_dim), nn.LayerNorm(embed_dim))
        self.vision_proj = nn.Sequential(nn.Linear(2048, embed_dim), nn.LayerNorm(embed_dim))
        self.audio_proj = nn.Sequential(nn.Linear(768, embed_dim), nn.LayerNorm(embed_dim))
        
        self.text_vision_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.text_audio_cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, input_ids, attention_mask, pixel_values, audio_emb=None):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_seq = self.text_proj(text_out.last_hidden_state)
        text_cls = text_seq[:, 0, :]
        
        vis_out = self.vision_encoder(pixel_values)
        vis_flat = torch.flatten(vis_out, 1)
        vis_emb = self.vision_proj(vis_flat).unsqueeze(1)
        
        aud_emb = torch.zeros_like(vis_emb) if audio_emb is None else self.audio_proj(audio_emb).unsqueeze(1)
            
        tv_attended, _ = self.text_vision_cross_attn(query=text_seq, key=vis_emb, value=vis_emb)
        tv_cls = tv_attended[:, 0, :]
        
        ta_attended, _ = self.text_audio_cross_attn(query=text_seq, key=aud_emb, value=aud_emb)
        ta_cls = ta_attended[:, 0, :]
        
        fused_vector = torch.cat((text_cls, tv_cls, ta_cls), dim=1)
        return self.classifier(fused_vector)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

@torch.no_grad()
def load_deepfake_model(weights_path="models/unique_trimodal_deepfake_detector.pt"):
    model = UniqueTriModalDeepfakeSystem().to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def predict_content(model, title_text, image_obj=None):
    if image_obj is None:
        image_obj = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
    enc = tokenizer([title_text], truncation=True, padding=True, max_length=128, return_tensors="pt")
    input_ids = enc['input_ids'].to(device)
    attn_mask = enc['attention_mask'].to(device)
    
    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = img_transform(image_obj.convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attn_mask, pixel_values)
        probs = F.softmax(logits, dim=1)[0]
        fake_prob = probs[1].item()
        
    if fake_prob >= 0.70:
        prediction = "FAKE"
    elif fake_prob <= 0.30:
        prediction = "REAL"
    else:
        prediction = "UNCERTAIN"
        
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    key_words = [t for t in tokens if t not in ['[CLS]', '[SEP]', '[PAD]'] and not t.startswith('##')]
    
    return {
        "prediction": prediction,
        "confidence": max(fake_prob, 1 - fake_prob),
        "fake_probability": fake_prob,
        "real_probability": probs[0].item(),
        "keywords": key_words[:4] if len(key_words) >= 4 else key_words
    }


def predict_video_content(
    model,
    title_text,
    video_bytes,
    video_filename="sample.mp4",
    max_frames=8,
    additional_images=None,
    progress_callback=None,
):
    """Screen uniformly sampled video frames with the trained text+image checkpoint.

    This is frame-level image inference followed by mean probability aggregation. It
    does not model temporal motion, audio, or lip-sync consistency.
    """
    import cv2
    import numpy as np
    from pathlib import Path

    temp_path = None
    capture = None
    try:
        suffix = Path(video_filename).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_path = temp_video.name

        capture = cv2.VideoCapture(temp_path)
        if not capture.isOpened():
            raise ValueError("OpenCV could not open this video. Try an MP4 encoded with H.264.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total_frames < 1:
            raise ValueError("The video frame count could not be read. Try a standard MP4 video.")

        frame_indices = np.linspace(0, total_frames - 1, num=min(max_frames, total_frames), dtype=int).tolist()
        sampled = []
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled.append((frame_index, frame_index / fps if fps > 0 else None, Image.fromarray(rgb_frame), "Video frame"))

        for extra_image in additional_images or []:
            sampled.append((None, None, extra_image.convert("RGB"), "Uploaded image"))

        if not sampled:
            raise ValueError("No readable frames were found in this video.")

        frame_results = []
        for position, (frame_index, timestamp_seconds, frame_image, sample_source) in enumerate(sampled, start=1):
            prediction = predict_content(model, title_text, frame_image)
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp_seconds,
                    "sample_source": sample_source,
                    "prediction": prediction["prediction"],
                    "real_probability": prediction["real_probability"],
                    "fake_probability": prediction["fake_probability"],
                    "keywords": prediction["keywords"],
                    "image": frame_image,
                }
            )
            if progress_callback:
                progress_callback(position, len(sampled))

        real_probability = sum(frame["real_probability"] for frame in frame_results) / len(frame_results)
        fake_probability = sum(frame["fake_probability"] for frame in frame_results) / len(frame_results)
        if fake_probability >= 0.70:
            prediction = "FAKE"
        elif fake_probability <= 0.30:
            prediction = "REAL"
        else:
            prediction = "UNCERTAIN"

        return {
            "prediction": prediction,
            "confidence": max(real_probability, fake_probability),
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "keywords": frame_results[0]["keywords"],
            "frames": [
                {key: value for key, value in frame.items() if key != "image"}
                for frame in frame_results
            ],
            "representative_frame": next((item[2] for item in sampled if item[3] == "Video frame"), sampled[0][2]),
            "sampled_frame_count": len(sampled),
            "video_duration_seconds": (total_frames / fps) if fps > 0 else None,
        }
    finally:
        if capture is not None:
            capture.release()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
