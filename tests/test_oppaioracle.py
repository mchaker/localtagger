"""OppaiOracle tests without model downloads: python -m unittest discover -s tests."""

import csv
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.backends import build_tagger
from app.backends.oppaioracle import IMAGE_SIZE, _parse_tags, _prepare_image
from app.catalog import load_catalog
from app.config import settings
from app.main import create_app

MODEL_ID = "oppaioracle-x1.1"

# selected_tags.csv layout: PAD and UNK first, then prefixed names in any order.
ROWS = [
    ("<PAD>", 0.99), ("<UNK>", 0.99),
    ("gen:1girl", 0.95), ("gen:solo", 0.5),
    ("rating:general", 0.9), ("rating:explicit", 0.01),
    ("char:power_(chainsaw_man)", 0.8), ("char:makima_(chainsaw_man)", 0.3),
    ("copyright:chainsaw_man", 0.75),
    ("artist:faint_artist", 0.2),
    ("meta:highres", 0.72),
]
PROBS = np.array([score for _, score in ROWS], dtype=np.float32)


def write_tags_csv(directory: Path) -> None:
    with open(directory / "selected_tags.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tag_id", "name", "category"])
        for index, (name, _) in enumerate(ROWS):
            writer.writerow([index, name, 0])


class OppaiOracleTaggerTests(unittest.TestCase):
    def setUp(self):
        self.spec = load_catalog(settings.catalog_path, [MODEL_ID]).specs[0]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        write_tags_csv(self.dir)

    def test_preprocessing_letterboxes_without_upscaling(self):
        # 896 x 448 white -> 448 x 224 centred on grey 114, scaled to [-1, 1].
        pixels, mask = _prepare_image(Image.new("RGB", (896, 448), (255, 255, 255)))
        self.assertEqual(pixels.shape, (1, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertEqual(mask.shape, (1, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(mask.dtype, np.bool_)
        grey = (114 / 255 - 0.5) / 0.5
        self.assertTrue(np.allclose(pixels[0, :, :112], grey))
        self.assertTrue(np.all(pixels[0, :, 112:336] == 1))
        self.assertTrue(np.allclose(pixels[0, :, 336:], grey))
        self.assertTrue(mask[0, :112].all() and mask[0, 336:].all())
        self.assertFalse(mask[0, 112:336].any())

        # Small images stay their size, centred; only the image area is unmasked.
        _, mask = _prepare_image(Image.new("RGB", (100, 50)))
        self.assertEqual(int((~mask).sum()), 100 * 50)
        self.assertFalse(mask[0, 199:249, 174:274].any())

        # Transparency is flattened onto the grey pad colour.
        clear, _ = _prepare_image(Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0, 0)))
        self.assertTrue(np.allclose(clear, grey))

    def test_load_session_and_category_mapping(self):
        session = Mock()
        session.run.return_value = [PROBS[None, :]]
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
                {"model.onnx", "selected_tags.csv"},
            )
            result = tagger.tag([Image.new("RGB", (8, 8))])[0]
            tagger.unload()
            self.assertFalse(tagger.loaded)

        feeds = session.run.call_args.args[1]
        self.assertEqual(set(feeds), {"pixel_values", "padding_mask"})
        # Prefixes are stripped, PAD/UNK never surface, and all ratings are kept.
        self.assertEqual(result.general, {"1girl": PROBS[2]})
        self.assertEqual(result.character, {"power_(chainsaw_man)": PROBS[6]})
        self.assertEqual(result.copyright, {"chainsaw_man": PROBS[8]})
        self.assertEqual(result.artist, {})
        self.assertEqual(result.meta, {"highres": PROBS[10]})
        self.assertEqual(result.rating, {"general": PROBS[4], "explicit": PROBS[5]})


class OppaiOracleApiTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        write_tags_csv(Path(tmp.name))
        self.app = create_app()
        self.client = TestClient(self.app)
        tagger = self.app.state.manager.get(MODEL_ID)
        with open(Path(tmp.name) / "selected_tags.csv", encoding="utf-8", newline="") as f:
            tagger._categories = _parse_tags(csv.DictReader(f))
        tagger._probabilities = Mock(return_value=PROBS)
        tagger._loaded = True
        buf = io.BytesIO()
        Image.new("RGB", (8, 8)).save(buf, format="PNG")
        self.png = buf.getvalue()

    def post(self, **params):
        return self.client.post(
            "/interrogate", params={"model": MODEL_ID, **params},
            files={"file": ("image.png", self.png, "image/png")},
        )

    def test_discovery(self):
        models = {m["id"]: m for m in self.client.get("/models").json()["models"]}
        info = models[MODEL_ID]
        self.assertEqual(info["group"], "oppaioracle")
        self.assertEqual(info["family"], "oppaioracle")
        self.assertEqual(info["default_threshold"], 0.71)
        self.assertEqual(info["default_character_threshold"], 0.71)
        self.assertEqual(info["default_thresholds"], {"copyright": 0.71, "artist": 0.71, "meta": 0.71})

    def test_post_uses_model_defaults_and_lower_thresholds(self):
        result = self.post().json()[0]
        self.assertEqual(set(result["tags"]), {"1girl", "power_(chainsaw_man)", "chainsaw_man", "highres"})
        self.assertEqual(set(result["rating"]), {"general", "explicit"})

        result = self.post(threshold=0.4, character_threshold=0.25, artist_threshold=0.1).json()[0]
        self.assertEqual(set(result["tags"]), {
            "1girl", "solo", "power_(chainsaw_man)", "makima_(chainsaw_man)",
            "chainsaw_man", "faint_artist", "highres",
        })


if __name__ == "__main__":
    unittest.main()
