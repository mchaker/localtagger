"""animetimm dbv4 backend (timm + safetensors, PyTorch).

animetimm models ship as safetensors only (no ONNX) and their repos are gated
behind an email-share agreement, so loading needs an HF token (see
:class:`app.config.Settings`). Inference follows the animetimm model-card recipe:
``timm.create_model('hf-hub:<repo>')`` + a torchvision transform built from the
repo's ``preprocess.json``, with per-tag thresholds in ``selected_tags.csv``.
"""

import json
from typing import List, Optional

from PIL import Image

from app.config import settings

from .base import TagResult, Tagger

# Danbooru tag categories used by animetimm's selected_tags.csv.
CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9


class AnimetimmTagger(Tagger):
    def __init__(self, spec):
        super().__init__(spec)
        self._model = None
        self._preprocessor = None
        self._names: List[str] = []
        self._categories = None
        self._device = "cpu"

    def load(self) -> None:
        if self._loaded:
            return

        import pandas as pd
        import torch
        from huggingface_hub import hf_hub_download
        from imgutils.preprocess import create_torchvision_transforms
        from timm import create_model

        token = settings.hf_token
        self._device = settings.resolve_device()

        model = create_model(f"hf-hub:{self.spec.repo}", pretrained=True)
        model.eval()
        model.to(self._device)
        self._model = model

        with open(
            hf_hub_download(
                repo_id=self.spec.repo,
                repo_type="model",
                filename="preprocess.json",
                token=token,
            ),
            "r",
        ) as f:
            self._preprocessor = create_torchvision_transforms(json.load(f)["test"])

        df_tags = pd.read_csv(
            hf_hub_download(
                repo_id=self.spec.repo,
                repo_type="model",
                filename="selected_tags.csv",
                token=token,
            ),
            keep_default_na=False,
        )
        self._names = df_tags["name"].tolist()
        self._categories = df_tags["category"].tolist()
        self._loaded = True

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        import torch

        input_ = self._preprocessor(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._model(input_)
            prediction = torch.sigmoid(output)[0].cpu().numpy()

        result = TagResult()
        for name, category, prob in zip(self._names, self._categories, prediction):
            prob = float(prob)
            cat = int(category)
            if cat == CATEGORY_RATING:
                result.rating[name] = prob
            elif cat == CATEGORY_CHARACTER:
                if prob >= character_threshold:
                    result.character[name] = prob
            else:  # general (and any other tag categories)
                if prob >= general_threshold:
                    result.general[name] = prob
        return result

    def unload(self) -> None:
        self._model = None
        self._preprocessor = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
