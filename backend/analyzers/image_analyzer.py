from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from backend.models import ImageAnalysis

logger = logging.getLogger(__name__)

# Lazy-loaded CLIP model
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def unload_clip():
    """Free CLIP model from memory to reclaim GPU VRAM and CPU RAM."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        return

    import gc

    del _clip_model
    _clip_model = None
    _clip_preprocess = None
    _clip_tokenizer = None

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()
    logger.info("CLIP model unloaded")


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return

    try:
        import open_clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        model.eval()
        _clip_model = model
        _clip_preprocess = preprocess
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
        logger.info("CLIP model loaded on %s", device)
    except Exception as e:
        logger.warning("Failed to load CLIP: %s. Falling back to histogram similarity.", e)


def detect_faces(image_path: Path) -> list[tuple[int, int, int, int]]:
    """Detect faces using OpenCV DNN. Returns list of (x, y, w, h)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    proto = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

    try:
        # Try DNN-based detector first (more accurate)
        net = cv2.dnn.readNetFromCaffe(
            str(Path(cv2.data.haarcascades).parent / "deploy.prototxt"),
            str(Path(cv2.data.haarcascades).parent / "res10_300x300_ssd_iter_140000.caffemodel"),
        )
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))
        return faces
    except Exception:
        pass

    # Fallback: Haar cascade
    cascade = cv2.CascadeClassifier(proto)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [tuple(f) for f in faces]


def compute_clip_similarity(img1_path: Path, img2_path: Path) -> float:
    """Compute CLIP cosine similarity between two images. Returns 0-1."""
    _load_clip()

    if _clip_model is None:
        return _histogram_similarity(img1_path, img2_path)

    try:
        import torch

        device = next(_clip_model.parameters()).device
        img1 = _clip_preprocess(Image.open(img1_path).convert("RGB")).unsqueeze(0).to(device)
        img2 = _clip_preprocess(Image.open(img2_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            feat1 = _clip_model.encode_image(img1)
            feat2 = _clip_model.encode_image(img2)
            feat1 = feat1 / feat1.norm(dim=-1, keepdim=True)
            feat2 = feat2 / feat2.norm(dim=-1, keepdim=True)
            similarity = (feat1 @ feat2.T).item()

        return float(np.clip(similarity, 0.0, 1.0))
    except Exception as e:
        logger.warning("CLIP similarity failed: %s", e)
        return _histogram_similarity(img1_path, img2_path)


def _histogram_similarity(img1_path: Path, img2_path: Path) -> float:
    """Fallback: histogram-based similarity."""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        return 0.5

    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL))


def analyze_image_pair(img1_path: Path, img2_path: Path) -> ImageAnalysis:
    """Full analysis of an image pair."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    # Face detection on both images
    faces1 = detect_faces(img1_path)
    faces2 = detect_faces(img2_path)
    has_faces = len(faces1) > 0 or len(faces2) > 0

    # Compute face area ratio (largest face)
    face_ratio = 0.0
    for faces, img in [(faces1, img1), (faces2, img2)]:
        for x, y, w, h in faces:
            ratio = (w * h) / (img.width * img.height)
            face_ratio = max(face_ratio, ratio)

    # CLIP similarity
    similarity = compute_clip_similarity(img1_path, img2_path)

    return ImageAnalysis(
        has_faces=has_faces,
        face_ratio=round(face_ratio, 3),
        avg_resolution=(img1.width + img1.height + img2.width + img2.height) / 4,
        min_width=min(img1.width, img2.width),
        min_height=min(img1.height, img2.height),
        clip_similarity=round(similarity, 3),
    )
