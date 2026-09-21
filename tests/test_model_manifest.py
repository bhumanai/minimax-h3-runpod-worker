from __future__ import annotations

import unittest

import download_models


class ModelManifestTests(unittest.TestCase):
    def test_worker_downloads_both_official_h3_diffusion_checkpoints(self):
        files = {item.relative_path: item.size for item in download_models.MODEL_FILES}
        self.assertEqual(
            files["diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"],
            20_970_379_616,
        )
        self.assertEqual(
            files["diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"],
            20_970_379_616,
        )


if __name__ == "__main__":
    unittest.main()
