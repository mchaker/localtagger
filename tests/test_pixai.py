"""PixAI integration tests without model downloads: python -m unittest discover -s tests."""

import dataclasses
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.backends import build_tagger
from app.backends.base import TagResult
from app.backends.pixai_onnx import IMAGE_SIZE, _prepare_image
from app.catalog import load_catalog
from app.config import settings
from app.main import create_app


SCORES = {
    "general": {"solo": 0.2},
    "character": {"new_character": 0.3},
    "copyright": {"new_series": 0.25},
    "style": {"new_artist": 0.2, "faint_artist": 0.1},
    "meta": {"new_medium": 0.2},
    "rating": {"rating:g": 0.9, "rating:s": 0.1},
}


# SCORES laid out like the ONNX export's tags.json: one block per category, in
# order, each at its own offset into a flat probability vector.
TAG_MAP = {"num_classes": 0, "categories": []}
for _name, _tags in SCORES.items():
    TAG_MAP["categories"].append({
        "name": _name, "offset": TAG_MAP["num_classes"], "count": len(_tags), "tags": list(_tags),
    })
    TAG_MAP["num_classes"] += len(_tags)
PROBS = np.array([score for tags in SCORES.values() for score in tags.values()])
CATEGORIES = [
    ("artist" if c["name"] == "style" else c["name"], c["offset"], c["tags"])
    for c in TAG_MAP["categories"]
]


class PixaiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)
        self.tagger = self.app.state.manager.get("pixai-v1.0")
        self.tagger._categories = CATEGORIES
        self.tagger._probabilities = Mock(return_value=PROBS)
        self.tagger._loaded = True
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        self.png = buf.getvalue()

    def post(self, **params):
        return self.client.post(
            "/interrogate", params={"model": "pixai-v1.0", **params},
            files={"file": ("image.png", self.png, "image/png")},
        )

    def test_discovery(self):
        models = {m["id"]: m for m in self.client.get("/models").json()["models"]}
        info = models["pixai-v1.0"]
        self.assertEqual(info["group"], "wdtagger")
        self.assertEqual(info["family"], "pixai_onnx")
        self.assertEqual(info["default_threshold"], 0.17)
        self.assertEqual(info["default_character_threshold"], 0.27)
        self.assertEqual(info["default_thresholds"], {
            "copyright": 0.24, "artist": 0.15, "meta": 0.17, "rating": 0.41,
        })
        self.assertIn("pixai-v0.9", models)

    def test_post_preserves_categories_and_uses_model_defaults(self):
        response = self.post()
        self.assertEqual(response.status_code, 200)
        result = response.json()[0]
        self.assertEqual(set(result["tags"]), {
            "solo", "new_character", "new_series", "new_artist", "new_medium",
        })
        self.assertEqual(result["artist"], {"new_artist": 0.2})
        self.assertEqual(result["copyright"], {"new_series": 0.25})
        self.assertEqual(result["meta"], {"new_medium": 0.2})
        self.assertEqual(result["rating"], {"rating:g": 0.9})

    def test_post_can_lower_or_raise_each_threshold(self):
        response = self.post(
            threshold=0.3, character_threshold=0.4, artist_threshold=0,
            copyright_threshold=0.3, meta_threshold=0.3, rating_threshold=0,
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()[0]
        self.assertEqual(result["tags"], {"new_artist": 0.2, "faint_artist": 0.1})
        self.assertEqual(result["rating"], SCORES["rating"])

    def test_get_uses_the_same_thresholds(self):
        with patch("app.routers.interrogate.download_image", return_value=Image.new("RGB", (8, 8))):
            response = self.client.get("/interrogate", params={
                "model": "pixai-v1.0", "url": "https://example.com/image.png",
                "artist_threshold": 0, "meta_threshold": 1,
            })
        self.assertEqual(response.status_code, 200)
        result = response.json()[0]
        self.assertIn("faint_artist", result["tags"])
        self.assertNotIn("new_medium", result["tags"])

    def test_batch_zip_keeps_category_tags_and_formatting(self):
        response = self.client.post("/interrogate", params={
            "model": "pixai-v1.0", "output_format": "zip", "artist_threshold": 0,
            "use_spaces": True, "trigger_word": "trigger",
        }, files=[
            ("file", ("first.png", self.png, "image/png")),
            ("file", ("second.png", self.png, "image/png")),
        ])
        self.assertEqual(response.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {"first.png", "first.txt", "second.png", "second.txt"})
            for name in ("first.txt", "second.txt"):
                text = archive.read(name).decode()
                self.assertTrue(text.startswith("trigger "))
                self.assertIn("faint-artist", text)
                self.assertIn("new-series", text)
                self.assertIn("new-medium", text)
                self.assertNotIn("rating:", text)

    def test_legacy_response_and_alias_still_work(self):
        legacy = self.app.state.manager.get("pixai-v0.9")
        legacy._loaded = True
        with patch.object(legacy, "_tag_one", return_value=TagResult(general={"solo": 0.8})):
            response = self.client.post("/interrogate/pixai", files={
                "file": ("image.png", self.png, "image/png"),
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["tags"], {"solo": 0.8})
        self.assertEqual(response.json()[0]["model"], "pixai-v0.9")


class PixaiOnnxTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_catalog(settings.catalog_path, ["pixai-v1.0"]).specs[0]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        (self.dir / "tags.json").write_text(json.dumps(TAG_MAP))

    def test_preprocessing_matches_rescale_pad(self):
        # 2:1 white image -> 1008 x 504 centred on black, scaled to [-1, 1].
        arr = _prepare_image(Image.new("RGB", (200, 100), (255, 255, 255)))
        self.assertEqual(arr.shape, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(arr.dtype, np.float32)
        self.assertTrue(np.all(arr[0, :, :252] == -1))
        self.assertTrue(np.all(arr[0, :, 252:756] == 1))
        self.assertTrue(np.all(arr[0, :, 756:] == -1))
        # Transparency is flattened onto white, not black.
        clear = _prepare_image(Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0)))
        self.assertTrue(np.all(clear == 1))

    def test_load_session_and_category_mapping(self):
        session = Mock()
        session.get_inputs.return_value = [SimpleNamespace(name="pixel_values")]
        session.run.return_value = [np.log(PROBS / (1 - PROBS))[None, :]]
        ort = SimpleNamespace(
            InferenceSession=Mock(return_value=session),
            get_available_providers=Mock(return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
        )
        download = Mock(side_effect=lambda repo_id, filename, token: str(self.dir / filename))
        tagger = build_tagger(self.spec)
        with patch.dict("sys.modules", {"onnxruntime": ort}), \
             patch("huggingface_hub.hf_hub_download", download), \
             patch.object(settings, "resolve_device", return_value="cuda"):
            tagger.load()
            tagger.load()
            ort.InferenceSession.assert_called_once_with(
                str(self.dir / "model.onnx"),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.assertEqual(
                {call.kwargs["filename"] for call in download.call_args_list},
                {"model.onnx", "model.onnx.data", "tags.json"},
            )
            result = tagger.tag([Image.new("RGB", (8, 8))])[0]
            tagger.unload()
            self.assertFalse(tagger.loaded)
            self.assertIsNone(tagger._session)

        pixel_values = session.run.call_args.args[1]["pixel_values"]
        self.assertEqual(pixel_values.shape, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(set(result.general), {"solo"})
        self.assertEqual(set(result.character), {"new_character"})
        self.assertEqual(set(result.copyright), {"new_series"})
        self.assertEqual(set(result.artist), {"new_artist"})
        self.assertEqual(set(result.meta), {"new_medium"})
        self.assertEqual(set(result.rating), {"rating:g"})
        self.assertAlmostEqual(result.artist["new_artist"], 0.2, places=6)

    def test_cpu_only_without_cuda(self):
        session = Mock()
        ort = SimpleNamespace(
            InferenceSession=Mock(return_value=session),
            get_available_providers=Mock(return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),
        )
        download = Mock(side_effect=lambda repo_id, filename, token: str(self.dir / filename))
        with patch.dict("sys.modules", {"onnxruntime": ort}), \
             patch("huggingface_hub.hf_hub_download", download), \
             patch.object(settings, "resolve_device", return_value="cpu"):
            build_tagger(self.spec).load()
        self.assertEqual(ort.InferenceSession.call_args.kwargs["providers"], ["CPUExecutionProvider"])


class PixaiTransformersFallbackTests(unittest.TestCase):
    def test_fp32_pipeline_and_lazy_reload(self):
        import torch

        spec = dataclasses.replace(
            load_catalog(settings.catalog_path, ["pixai-v1.0"]).specs[0],
            family="pixai_transformers", repo="pixai-labs/pixai-tagger-v1.0",
        )
        tagger = build_tagger(spec)
        factory = Mock(return_value=Mock())
        with patch.dict("sys.modules", {"transformers": SimpleNamespace(pipeline=factory)}), \
             patch.object(settings, "resolve_device", return_value="cpu"), \
             patch("torch.cuda.is_available", return_value=False):
            self.assertFalse(tagger.loaded)
            tagger.load()
            tagger.load()
            factory.assert_called_once()
            self.assertEqual(factory.call_args.kwargs["dtype"], torch.float32)
            self.assertEqual(factory.call_args.kwargs["model"], spec.repo)
            self.assertEqual(factory.call_args.kwargs["image_processor"], spec.repo)
            tagger.unload()
            self.assertFalse(tagger.loaded)
            self.assertIsNone(tagger._pipeline)
            tagger.load()
            self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
