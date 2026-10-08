"""Kaloscope 3.0 Preview artist-style classifier (PyTorch, DINOv3 ViT-B/16).

This is an artist similarity classifier, distinct from the Danbooru taggers, so
it has its own small API rather than implementing the Tagger interface.

Uses the v1 artist classifier from heathcliff01/Kaloscope3.0-preview: a DINOv3
ViT-B/16 backbone with a linear head over 44,129 Danbooru artists. The backbone
code is vendored in ``app/backends/dinov3`` (DINOv3 License).
"""

import math
import threading
from typing import Dict, List

from PIL import Image, ImageOps

from app.config import settings

KALOSCOPE_REPO = "heathcliff01/Kaloscope3.0-preview"
# Pinned: the preview repo may change, and the weights and label order must match.
KALOSCOPE_REVISION = "5e4bfa229619b7c459ec126283827373df9de564"
_SUBFOLDER = "v1-artist-classifier"
_MODEL_FILE = "model.safetensors"
_LABELS_FILE = "class_mapping.csv"
MODEL_NAME = "kaloscope-3.0-preview"

_INPUT_SIZE = 512
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def _prepare_rgb(image: Image.Image) -> Image.Image:
    # Model card: EXIF orientation, transparency composited on white, then RGB.
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _build_model(state):
    """Backbone + artist head, following the v1-artist-classifier README."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    from app.backends.dinov3.hub.backbones import dinov3_vitb16

    class KaloscopeV3(nn.Module):
        def __init__(self, num_classes: int):
            super().__init__()
            self.backbone = dinov3_vitb16(pretrained=False)
            self.head = nn.Linear(1536, num_classes)

        def forward(self, images):
            feats = self.backbone.forward_features(images)
            pooled = torch.cat(
                (feats["x_norm_clstoken"], feats["x_norm_patchtokens"].mean(dim=1)), dim=-1
            )
            # The head was trained on FP32 L2-normalized CLS + mean-patch
            # features scaled by sqrt(1536). Skipping this gives wrong logits.
            head_input = F.normalize(pooled.float(), dim=-1) * math.sqrt(pooled.shape[-1])
            return self.head(head_input)

    head = {k[len("head."):]: v for k, v in state.items() if k.startswith("head.")}
    model = KaloscopeV3(num_classes=head["weight"].shape[0])
    model.backbone.load_state_dict(
        {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}, strict=True
    )
    model.head.load_state_dict(head, strict=True)
    return model.eval()


class KaloscopeClassifier:
    def __init__(self):
        self._model = None
        self._transform = None
        self._labels: Dict[int, str] = {}
        self._device = "cpu"
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        # Runs in worker threads: load once, and publish the model last so a
        # concurrent caller never sees it without its labels.
        with self._load_lock:
            if self._model is not None:
                return
            import pandas as pd
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
            from torchvision import transforms

            print("Loading Kaloscope 3.0 Preview (artist style classifier)...")

            def download(filename: str) -> str:
                return hf_hub_download(
                    repo_id=KALOSCOPE_REPO,
                    filename=filename,
                    subfolder=_SUBFOLDER,
                    revision=KALOSCOPE_REVISION,
                )

            model_path = download(_MODEL_FILE)
            labels_path = download(_LABELS_FILE)

            labels_df = pd.read_csv(labels_path)
            labels_df["class_name"] = labels_df["class_name"].str.strip("'")
            labels = dict(zip(labels_df["class_id"], labels_df["class_name"]))

            device = settings.resolve_device()
            model = _build_model(load_file(model_path, device="cpu")).to(device)
            if model.head.out_features != len(labels):
                raise RuntimeError("Kaloscope head size does not match class_mapping.csv")

            # Short side to 512, center crop 512 (no stretching), ImageNet norm.
            self._transform = transforms.Compose(
                [
                    _prepare_rgb,
                    transforms.Resize(_INPUT_SIZE),
                    transforms.CenterCrop(_INPUT_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(_MEAN, _STD),
                ]
            )
            self._labels = labels
            self._device = device
            self._model = model

    def infer(self, image: Image.Image, top_k: int = 10) -> List[Dict[str, float]]:
        import torch

        self.load()
        input_tensor = self._transform(image).unsqueeze(0).to(self._device)
        with torch.inference_mode():
            logits = self._model(input_tensor)[0].float()

        probs = logits.softmax(dim=-1)
        scores, indices = probs.topk(min(top_k, probs.numel()))
        return [
            {"name": self._labels.get(int(i), f"unknown_{int(i)}"), "score": float(s)}
            for s, i in zip(scores.tolist(), indices.tolist())
        ]
