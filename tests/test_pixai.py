"""PixAI integration tests without model downloads: python -m unittest discover -s tests."""

import io
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from app.backends import build_tagger
from app.backends.base import TagResult
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


def predict(image, *, threshold):
    return {"results": {
        category: {name: score for name, score in tags.items() if score > threshold[category]}
        for category, tags in SCORES.items()
    }}


class PixaiIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)
        self.tagger = self.app.state.manager.get("pixai-v1.0")
        self.tagger._pipeline = Mock(side_effect=predict)
        self.tagger._loaded = True
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        self.png = buf.getvalue()

    def post(self, **params):
        return self.client.post(
            "/interrogate", params={"model": "pixai-v1.0", **params},
            files={"file": ("image.png", self.png, "image/png")},
        )

    def test_discovery_and_optional_bf16(self):
        models = {m["id"]: m for m in self.client.get("/models").json()["models"]}
        info = models["pixai-v1.0"]
        self.assertEqual(info["default_threshold"], 0.17)
        self.assertEqual(info["default_character_threshold"], 0.27)
        self.assertEqual(info["default_thresholds"], {
            "copyright": 0.24, "artist": 0.15, "meta": 0.17, "rating": 0.41,
        })
        self.assertIn("pixai-v0.9", models)
        self.assertNotIn("pixai-v1.0-bf16", models)
        optional = load_catalog(settings.catalog_path, ["pixai-v1.0-bf16"])
        self.assertEqual(optional.specs[0].dtype, "bfloat16")

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


class PixaiLoadingTests(unittest.TestCase):
    def test_precision_selection_and_lazy_reload(self):
        import torch

        for device, bf16_supported, expected in [
            ("cpu", False, torch.float32),
            ("cuda", False, torch.float32),
            ("cuda", True, torch.bfloat16),
        ]:
            with self.subTest(device=device, bf16_supported=bf16_supported):
                spec = load_catalog(settings.catalog_path, ["pixai-v1.0-bf16"]).specs[0]
                tagger = build_tagger(spec)
                model = torch.nn.Module()
                model.backbone = torch.nn.Linear(2, 2)
                model.head = torch.nn.Linear(2, 2)
                original_head = model.head.weight.detach().clone()
                factory = Mock(return_value=SimpleNamespace(model=model))
                with patch.dict("sys.modules", {"transformers": SimpleNamespace(pipeline=factory)}), \
                     patch.object(settings, "resolve_device", return_value=device), \
                     patch("torch.cuda.is_bf16_supported", return_value=bf16_supported), \
                     patch("torch.cuda.is_available", return_value=False):
                    self.assertFalse(tagger.loaded)
                    tagger.load()
                    tagger.load()
                    factory.assert_called_once()
                    self.assertEqual(factory.call_args.kwargs["dtype"], torch.float32)
                    self.assertEqual(factory.call_args.kwargs["model"], spec.repo)
                    self.assertEqual(factory.call_args.kwargs["image_processor"], spec.repo)
                    self.assertEqual(model.backbone.weight.dtype, expected)
                    self.assertEqual(model.head.weight.dtype, torch.float32)
                    self.assertTrue(torch.equal(model.head.weight, original_head))
                    tagger.unload()
                    self.assertFalse(tagger.loaded)
                    self.assertIsNone(tagger._pipeline)
                    tagger.load()
                    self.assertEqual(factory.call_count, 2)


if __name__ == "__main__":
    unittest.main()
