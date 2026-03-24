import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from unittest import mock

import numpy as np

from runtime import fall_predictor as fp


class FallPredictorTests(unittest.TestCase):
    def _make_workspace_temp_dir(self):
        temp_root = os.path.join(os.path.dirname(__file__), "_tmp")
        os.makedirs(temp_root, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=temp_root)

    def test_build_fall_predictor_returns_null_predictor(self):
        predictor = fp.build_fall_predictor()
        self.assertIsInstance(predictor, fp.NullFallPredictor)

    def test_null_predictor_returns_unavailable_prediction_for_clip(self):
        predictor = fp.NullFallPredictor()
        clip = fp.FallFeatureClip(
            timestamp_start_ms=1000,
            timestamp_end_ms=1100,
            RT=[[1.0, 2.0]],
            height_history=[(1000, 1.0, 0.0, 1.0)],
            tracked_target_state={"locked": True},
            runtime_cfg={"config_file": "config/Radar.cfg"},
        )
        result = predictor.predict(clip)
        self.assertFalse(result.available)
        self.assertEqual(result.label, "")

    def test_load_predictor_from_missing_path_returns_failure(self):
        result = fp.load_predictor_from_path("missing_predictor_file.py")
        self.assertFalse(result.success)
        self.assertIsInstance(result.predictor, fp.NullFallPredictor)
        self.assertIn("not found", result.message.lower())

    def test_load_predictor_plugin_supports_path_aware_builder(self):
        with self._make_workspace_temp_dir() as temp_dir:
            plugin_path = os.path.join(temp_dir, "demo_predictor.py")
            with open(plugin_path, "w", encoding="utf-8") as handle:
                handle.write(
                    textwrap.dedent(
                        """
                        from runtime import fall_predictor as fp

                        class DemoPredictor(fp.BaseFallPredictor):
                            def __init__(self, source_path):
                                self.source_path = source_path

                            def predict(self, clip):
                                return fp.FallPrediction(
                                    available=True,
                                    label="fall",
                                    score=1.0,
                                    probability=0.95,
                                )

                        def build_fall_predictor(model_path):
                            return DemoPredictor(model_path)
                        """
                    ).strip()
                )

            result = fp.load_predictor_from_path(plugin_path)
            self.assertTrue(result.success)
            self.assertEqual(result.model_path, plugin_path)
            prediction = result.predictor.predict(fp.FallFeatureClip(0, 0))
            self.assertTrue(prediction.available)
            self.assertEqual(prediction.label, "fall")

    def test_discover_default_model_path_prefers_latest_checkpoint(self):
        with self._make_workspace_temp_dir() as temp_dir:
            older_dir = os.path.join(temp_dir, "older_run")
            newer_dir = os.path.join(temp_dir, "newer_run")
            os.makedirs(older_dir, exist_ok=True)
            os.makedirs(newer_dir, exist_ok=True)

            older_path = os.path.join(older_dir, "raca_v2_best.pth")
            newer_path = os.path.join(newer_dir, "raca_v2_best.pth")

            for path in (older_path, newer_path):
                with open(path, "wb") as handle:
                    handle.write(b"checkpoint")

            now = time.time()
            os.utime(older_path, (now - 60, now - 60))
            os.utime(newer_path, (now, now))

            with mock.patch.dict(
                os.environ,
                {"RADAR_MODEL_PATH": "", "RADAR_DEFAULT_MODEL_PATH": ""},
                clear=False,
            ):
                discovered = fp.discover_default_model_path([temp_dir])

            self.assertEqual(discovered, os.path.abspath(newer_path))

    def test_discover_default_model_path_honors_env_override(self):
        with self._make_workspace_temp_dir() as temp_dir:
            env_model_path = os.path.join(temp_dir, "explicit_model.pth")
            with open(env_model_path, "wb") as handle:
                handle.write(b"checkpoint")

            with mock.patch.dict(
                os.environ,
                {"RADAR_MODEL_PATH": env_model_path, "RADAR_DEFAULT_MODEL_PATH": ""},
                clear=False,
            ):
                discovered = fp.discover_default_model_path([os.path.join(temp_dir, "missing")])

            self.assertEqual(discovered, os.path.abspath(env_model_path))

    def test_load_predictor_returns_failure_for_invalid_pth_checkpoint(self):
        temp_root = os.path.join(os.path.dirname(__file__), "_tmp")
        os.makedirs(temp_root, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False, dir=temp_root) as handle:
            temp_path = handle.name
        try:
            result = fp.load_predictor_from_path(temp_path)
            self.assertFalse(result.success)
            self.assertIn("raca_v2", result.message.lower())
        finally:
            os.remove(temp_path)

    def test_load_model_runtime_contract_prefers_model_meta(self):
        with self._make_workspace_temp_dir() as temp_dir:
            cfg_path = os.path.join(temp_dir, "Radar.cfg")
            model_path = os.path.join(temp_dir, "demo_model.pth")
            meta_path = os.path.join(temp_dir, "model_meta.json")

            with open(cfg_path, "w", encoding="utf-8") as handle:
                handle.write("frameCfg 0 2 64 0 50 1 0\n")
            with open(model_path, "wb") as handle:
                handle.write(b"checkpoint")

            cfg_sha256 = fp._compute_file_sha256(cfg_path)
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "cfg_sha256": cfg_sha256,
                        "clip_frames": 100,
                        "frame_periodicity_ms": 50.0,
                        "feature_shapes": {
                            "RD": [100, 128, 64],
                            "RA": [100, 361, 128],
                            "RE": [100, 361, 128],
                        },
                        "class_names": ["non-fall", "fall"],
                        "positive_labels": ["fall"],
                    },
                    handle,
                )

            runtime_cfg = {
                "num_range_bins": 128,
                "num_doppler_bins": 64,
                "frame_periodicity_ms": 50.0,
                "aoa_fov": {"azimuth": (-90.0, 90.0)},
                "config_file": cfg_path,
            }

            contract = fp.load_model_runtime_contract(model_path, runtime_cfg=runtime_cfg)
            validation = fp.validate_runtime_contract(contract, runtime_cfg)

            self.assertIsNotNone(contract)
            self.assertEqual(contract.source, "model_meta.json")
            self.assertEqual(contract.feature_shapes["RD"], (100, 128, 64))
            self.assertTrue(validation.valid)
            self.assertEqual(validation.status, "ready")

    def test_validate_runtime_contract_flags_cfg_mismatch(self):
        contract = fp.ModelRuntimeContract(
            model_path="demo_model.pth",
            source="checkpoint-defaults",
            clip_frames=100,
            frame_periodicity_ms=50.0,
            feature_shapes={
                "RD": (100, 128, 64),
                "RA": (100, 361, 128),
                "RE": (100, 361, 128),
            },
        )
        runtime_cfg = {
            "num_range_bins": 64,
            "num_doppler_bins": 64,
            "frame_periodicity_ms": 100.0,
            "aoa_fov": {"azimuth": (-90.0, 90.0)},
            "config_file": "",
        }

        validation = fp.validate_runtime_contract(contract, runtime_cfg)

        self.assertFalse(validation.valid)
        self.assertEqual(validation.status, "mismatch")
        self.assertIn("cfg mismatch", validation.message.lower())
        self.assertTrue(any("range_bins" in mismatch for mismatch in validation.mismatches))
        self.assertTrue(
            any("frame_periodicity_ms" in mismatch for mismatch in validation.mismatches)
        )

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "checkpoints",
                "lstm_20260320_133236",
                "raca_v2_best.pth",
            )
        ),
        "real RACA checkpoint not found",
    )
    def test_load_real_raca_checkpoint_succeeds(self):
        checkpoint_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "checkpoints",
            "lstm_20260320_133236",
            "raca_v2_best.pth",
        )
        result = fp.load_predictor_from_path(checkpoint_path)
        self.assertTrue(result.success)
        self.assertEqual(os.path.abspath(result.model_path), os.path.abspath(checkpoint_path))
        self.assertEqual(type(result.predictor).__name__, "RACAfallPredictor")

    @unittest.skipUnless(
        os.path.isfile(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "checkpoints",
                "lstm_20260320_133236",
                "raca_v2_best.pth",
            )
        ),
        "real RACA checkpoint not found",
    )
    def test_preload_torch_runtime_avoids_qt_dll_conflict(self):
        checkpoint_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "checkpoints",
                "lstm_20260320_133236",
                "raca_v2_best.pth",
            )
        )
        script = textwrap.dedent(
            f"""
            from runtime import fall_predictor as fp

            ok, message = fp.preload_torch_runtime()
            if not ok:
                raise SystemExit(message)

            import pyqtgraph as pg
            from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
            from pyqtgraph.opengl import GLViewWidget
            import matplotlib.pyplot as plt

            result = fp.load_predictor_from_path(r"{checkpoint_path}")
            if not result.success:
                raise SystemExit(result.message)
            print(result.message)
            """
        )
        completed = subprocess.run(
            [
                os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        ".venv",
                        "Scripts",
                        "python.exe",
                    )
                )
                if os.path.isfile(
                    os.path.join(
                        os.path.dirname(__file__),
                        "..",
                        ".venv",
                        "Scripts",
                        "python.exe",
                    )
                )
                else sys.executable,
                "-c",
                script,
            ],
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=(completed.stderr or completed.stdout or "subprocess failed").strip(),
        )

    def test_raca_predictor_inference_id_only_changes_on_new_inference(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        from runtime.raca_predictor import RACAfallPredictor

        class DummyModel(torch.nn.Module):
            def forward(self, rd, ra, re):
                batch = rd.shape[0]
                logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32, device=rd.device)
                return logits.repeat(batch, 1)

        predictor = RACAfallPredictor(
            model=DummyModel(),
            label_map={"non-fall": 0, "fall": 1},
            class_names=["non-fall", "fall"],
            device="cpu",
            window=2,
            stride=2,
            binary_mode=False,
        )

        frame_rdt = np.zeros((128, 64), dtype=np.float32)
        frame_art = np.zeros((361, 128), dtype=np.float32)
        frame_ert = np.zeros((361, 128), dtype=np.float32)

        def make_clip(timestamp_ms):
            return fp.FallFeatureClip(
                timestamp_start_ms=timestamp_ms,
                timestamp_end_ms=timestamp_ms + 50,
                RDT=frame_rdt,
                ART=frame_art,
                ERT=frame_ert,
            )

        first = predictor.predict(make_clip(0))
        self.assertFalse(first.available)

        second = predictor.predict(make_clip(50))
        self.assertTrue(second.available)
        self.assertEqual(second.metadata.get("inference_id"), 1)

        third = predictor.predict(make_clip(100))
        self.assertTrue(third.available)
        self.assertEqual(third.metadata.get("inference_id"), 1)

        fourth = predictor.predict(make_clip(150))
        self.assertTrue(fourth.available)
        self.assertEqual(fourth.metadata.get("inference_id"), 2)


if __name__ == "__main__":
    unittest.main()
