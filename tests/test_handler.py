from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_handler_module():
    fake_runpod = types.SimpleNamespace(
        serverless=types.SimpleNamespace(start=lambda _config: None, progress_update=lambda *_args: None)
    )
    sys.modules["runpod"] = fake_runpod
    spec = importlib.util.spec_from_file_location("h3_handler", ROOT / "handler.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = load_handler_module()

    def test_safe_input_name_rejects_path_traversal(self):
        for value in ("../driver.mp4", "folder/driver.mp4", "/tmp/driver.mp4"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.handler.safe_input_name(value)

    def test_safe_input_name_accepts_expected_media(self):
        self.assertEqual(self.handler.safe_input_name("driver.mp4"), "driver.mp4")
        self.assertEqual(self.handler.safe_input_name("identity.png"), "identity.png")

    def test_history_video_paths_stays_inside_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "video"
            video_dir.mkdir()
            expected = video_dir / "result.mp4"
            expected.write_bytes(b"video")
            history = {
                "outputs": {
                    "18": {
                        "video": [
                            {"filename": "result.mp4", "subfolder": "video", "type": "output"},
                            {"filename": "escape.mp4", "subfolder": "../../tmp", "type": "output"},
                        ]
                    }
                }
            }
            with mock.patch.object(self.handler, "OUTPUT_DIR", root):
                self.assertEqual(self.handler.history_video_paths(history), [expected.resolve()])


if __name__ == "__main__":
    unittest.main()
