"""WD14-style tagger shipped only as timm + safetensors (no ONNX export).

Some WD-family taggers (e.g. ashen-sensored's EVA02 canary, a fine-tune of
SmilingWolf/wd-eva02-large-tagger-v3) publish safetensors weights only, so they
can't go through imgutils' ``get_wd14_tags`` (which needs an ONNX file). This
backend loads the weights via timm instead and replicates the exact
preprocessing WD14 v3 models use: pad to a square on white, resize, convert to
BGR, and scale to [-1, 1] (not the standard ImageNet transform implied by the
repo's ``pretrained_cfg``). Tags come from ``selected_tags.csv`` with the same
category scheme as WD14/animetimm: 0=general, 4=character, 9=rating.
"""

import json

import numpy as np
from PIL import Image

from app.config import settings

from .base import TagResult, Tagger

CATEGORY_GENERAL = 0
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9


class WD14SafetensorsTagger(Tagger):
    def __init__(self, spec):
        super().__init__(spec)
        self._model = None
        self._names = []
        self._categories = []
        self._target_size = 448
        self._device = "cpu"

    def load(self) -> None:
        if self._loaded:
            return

        import pandas as pd
        import timm
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        token = settings.hf_token
        self._device = settings.resolve_device()

        config_path = hf_hub_download(
            repo_id=self.spec.repo, filename="config.json", token=token
        )
        with open(config_path) as f:
            config = json.load(f)
        input_size = config.get("pretrained_cfg", {}).get("input_size")
        if input_size:
            self._target_size = int(input_size[-1])

        model = timm.create_model(
            config["architecture"], pretrained=False, num_classes=config["num_classes"]
        )
        model_path = hf_hub_download(
            repo_id=self.spec.repo, filename="model.safetensors", token=token
        )
        model.load_state_dict(load_file(model_path), strict=True)
        model.eval()
        model.to(self._device)
        self._model = model

        tags_path = hf_hub_download(
            repo_id=self.spec.repo, filename="selected_tags.csv", token=token
        )
        df_tags = pd.read_csv(tags_path, keep_default_na=False)
        self._names = df_tags["name"].tolist()
        self._categories = df_tags["category"].tolist()
        self._loaded = True

    def _prepare_image(self, image: Image.Image):
        import torch

        image = image.convert("RGBA")
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(image, (pad_left, pad_top))
        if max_dim != self._target_size:
            padded = padded.resize((self._target_size, self._target_size), Image.BICUBIC)

        arr = np.asarray(padded, dtype=np.float32)
        arr = arr[:, :, ::-1] / 127.5 - 1.0  # RGB -> BGR, scale to [-1, 1]
        return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)

    def _tag_one(
        self, image: Image.Image, general_threshold: float, character_threshold: float
    ) -> TagResult:
        import torch

        input_ = self._prepare_image(image).to(self._device)
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
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self._loaded = False
