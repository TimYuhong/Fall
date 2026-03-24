import json
import os
import tempfile
import unittest

from training import export_model


class ExportModelTests(unittest.TestCase):
    def _make_workspace_temp_dir(self):
        temp_root = os.path.join(os.path.dirname(__file__), "_tmp")
        os.makedirs(temp_root, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=temp_root)

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__), "..", "config", "Radar.cfg")),
        "config/Radar.cfg not found",
    )
    def test_package_model_accepts_pth_checkpoint(self):
        cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "Radar.cfg")

        with self._make_workspace_temp_dir() as temp_dir:
            model_path = os.path.join(temp_dir, "raca_v2_best.pth")
            with open(model_path, "wb") as handle:
                handle.write(b"fake-checkpoint")

            output_dir = os.path.join(temp_dir, "deploy")
            result = export_model.package_model(
                model_path=model_path,
                output_dir=output_dir,
                cfg_path=cfg_path,
                class_names=["non-fall", "fall"],
                positive_labels=["fall"],
            )

            self.assertTrue(os.path.isfile(result["model_path"]))
            self.assertTrue(os.path.isfile(result["model_meta_path"]))
            self.assertEqual(result["model_meta"]["model_type"], "pth")

            with open(result["model_meta_path"], "r", encoding="utf-8") as handle:
                meta = json.load(handle)

            self.assertEqual(meta["model_type"], "pth")
            self.assertEqual(meta["class_names"], ["non-fall", "fall"])
            self.assertEqual(meta["positive_labels"], ["fall"])


if __name__ == "__main__":
    unittest.main()
