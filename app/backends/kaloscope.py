"""Kaloscope 2.0 artist-style classifier (ONNX).

Ported largely unchanged from the original main.py. This is an artist similarity
classifier, distinct from the Danbooru taggers, so it has its own small API
rather than implementing the Tagger interface.
"""

from typing import Dict, List

import numpy as np
from PIL import Image

KALOSCOPE_REPO = "DraconicDragon/Kaloscope-onnx"
_MODEL_FILE = "v2.0/kaloscope_2-0.onnx"
_LABELS_FILE = "v2.0/class_mapping.csv"


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def _prepare_image(image: Image.Image, target_size: int = 448) -> np.ndarray:
    # ImageNet-style preprocessing (LSNet): RGB, [0,1], mean/std normalize, CHW.
    image = image.convert("RGB").resize((target_size, target_size), Image.Resampling.LANCZOS)
    img = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    return np.expand_dims(img, axis=0)


class KaloscopeClassifier:
    def __init__(self):
        self._session = None
        self._labels: Dict[int, str] = {}

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def _providers(self):
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        print("Warning: CUDAExecutionProvider not found for ONNX Runtime. Falling back to CPU.")
        return ["CPUExecutionProvider"]

    def load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        import pandas as pd
        from huggingface_hub import hf_hub_download

        print("Loading Kaloscope 2.0 (artist style classifier)...")
        model_path = hf_hub_download(repo_id=KALOSCOPE_REPO, filename=_MODEL_FILE)
        labels_path = hf_hub_download(repo_id=KALOSCOPE_REPO, filename=_LABELS_FILE)

        self._session = ort.InferenceSession(model_path, providers=self._providers())
        labels_df = pd.read_csv(labels_path)
        labels_df["class_name"] = labels_df["class_name"].str.strip("'")
        self._labels = dict(zip(labels_df["class_id"], labels_df["class_name"]))

    def infer(self, image: Image.Image, top_k: int = 10) -> List[Dict[str, float]]:
        self.load()
        try:
            size = int(self._session.get_inputs()[0].shape[2])
        except Exception:
            size = 448

        input_tensor = _prepare_image(image, target_size=size)
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: input_tensor})[0][0]

        probs = _softmax(logits)
        top_indices = np.argsort(probs)[-top_k:][::-1]
        return [
            {"name": self._labels.get(int(i), f"unknown_{int(i)}"), "score": float(probs[i])}
            for i in top_indices
        ]
