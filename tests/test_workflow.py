from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h3_workflow import build_h3_ref2va_workflow


class WorkflowTests(unittest.TestCase):
    def test_full_quality_reference_graph(self):
        workflow = build_h3_ref2va_workflow(
            "identity.png",
            "driver.mp4",
            prompt="Use <Picture 1>, <Audio 1>, and <Video 1>.",
        )
        node = workflow["9"]
        self.assertEqual(node["inputs"]["ref_images.ref_image_0"], ["1", 0])
        self.assertEqual(node["inputs"]["ref_videos.ref_video_0"], ["3", 0])
        self.assertEqual(node["inputs"]["ref_video_audios.ref_video_audio_0"], ["3", 1])
        self.assertEqual(workflow["13"]["inputs"]["steps"], 20)
        self.assertNotIn("8", workflow)

    def test_turbo_requires_four_steps(self):
        with self.assertRaises(ValueError):
            build_h3_ref2va_workflow("identity.png", "driver.mp4", prompt="x", turbo=True, steps=20)

    def test_valid_frame_grid_is_enforced(self):
        with self.assertRaises(ValueError):
            build_h3_ref2va_workflow("identity.png", "driver.mp4", prompt="x", length=120)

    def test_reference_prompt_requires_literal_media_tags(self):
        with self.assertRaisesRegex(ValueError, r"<Picture 1>.*<Video 1>.*<Audio 1>"):
            build_h3_ref2va_workflow(
                "identity.png",
                "driver.mp4",
                prompt="Use Picture 1 as identity and Video 1 as motion with Audio 1.",
            )


if __name__ == "__main__":
    unittest.main()
