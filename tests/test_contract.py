"""API contract tests: python -m unittest discover -s tests."""

import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import __version__
from app.main import create_app


class HealthTests(unittest.TestCase):
    def test_reports_version_and_build(self):
        with patch.dict(os.environ, {"GIT_SHA": "abc1234"}):
            body = TestClient(create_app()).get("/health").json()
        self.assertEqual(body, {"status": "ok", "version": __version__, "git_sha": "abc1234"})

    def test_git_sha_is_optional(self):
        with patch.dict(os.environ, {"GIT_SHA": ""}):
            body = TestClient(create_app()).get("/health").json()
        self.assertIsNone(body["git_sha"])


if __name__ == "__main__":
    unittest.main()
