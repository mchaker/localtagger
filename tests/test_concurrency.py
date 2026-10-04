"""Tagging runs off the event loop and loads each model once: python -m unittest discover -s tests."""

import asyncio
import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from PIL import Image

from app.backends.base import TagResult, Tagger
from app.backends.kaloscope import KaloscopeClassifier
from app.main import create_app


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


class SlowTagger(Tagger):
    """Sleeps in load() and _tag_one() like a real model busy on the CPU."""

    def __init__(self, spec, delay):
        super().__init__(spec)
        self.delay = delay
        self.loads = 0
        self.started = threading.Event()
        self.finished = threading.Event()

    def load(self):
        self.loads += 1
        time.sleep(self.delay)
        self._loaded = True

    def _tag_one(self, image, general_threshold, character_threshold):
        self.started.set()
        time.sleep(self.delay)
        self.finished.set()
        return TagResult(general={"solo": 0.9})


class EventLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_slow_tagging_does_not_block_other_requests(self):
        app = create_app()
        manager = app.state.manager
        model_id = manager.catalog.default_model_id()
        tagger = SlowTagger(manager.catalog.get(model_id), delay=1.0)
        tagger._loaded = True
        manager._taggers[model_id] = tagger

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            slow = asyncio.create_task(client.post(
                f"/interrogate?model={model_id}", files={"file": ("a.png", _png(), "image/png")},
            ))
            self.assertTrue(await asyncio.to_thread(tagger.started.wait, 5))
            health = await client.get("/health")
            # Served while the tagger is still busy, not queued behind it.
            self.assertEqual(health.status_code, 200)
            self.assertFalse(tagger.finished.is_set())
            response = await slow

        self.assertEqual(response.status_code, 200)
        self.assertIn("solo", response.json()[0]["tags"])


class LoadOnceTests(unittest.TestCase):
    def _run_concurrently(self, target, n=4):
        results = [None] * n

        def run(i):
            results[i] = target()

        threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    def test_concurrent_first_requests_load_weights_once(self):
        manager = create_app().state.manager
        spec = manager.catalog.get(manager.catalog.default_model_id())
        tagger = SlowTagger(spec, delay=0.2)
        self._run_concurrently(lambda: tagger.tag([Image.new("RGB", (8, 8))]))
        self.assertEqual(tagger.loads, 1)

    def test_manager_builds_each_tagger_once(self):
        manager = create_app().state.manager
        model_id = manager.catalog.default_model_id()

        def slow_build(spec):
            time.sleep(0.1)
            return SlowTagger(spec, delay=0)

        with patch("app.manager.build_tagger", side_effect=slow_build) as build:
            taggers = self._run_concurrently(lambda: manager.get(model_id))
        build.assert_called_once()
        self.assertTrue(all(t is taggers[0] for t in taggers))

    def test_kaloscope_loads_once_with_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp) / "class_mapping.csv"
            labels.write_text("class_id,class_name\n0,'artist_a'\n")

            def slow_session(path, providers):
                time.sleep(0.2)
                return object()

            ort = SimpleNamespace(
                InferenceSession=slow_session,
                get_available_providers=lambda: ["CPUExecutionProvider"],
            )
            classifier = KaloscopeClassifier()
            with patch.dict("sys.modules", {"onnxruntime": ort}), \
                 patch("huggingface_hub.hf_hub_download", return_value=str(labels)) as download:
                self._run_concurrently(classifier.load)

        self.assertEqual(download.call_count, 2)  # model + labels, one load
        self.assertTrue(classifier.loaded)
        self.assertEqual(classifier._labels, {0: "artist_a"})


if __name__ == "__main__":
    unittest.main()
