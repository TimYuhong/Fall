import os
import tempfile
import unittest

from runtime import replay_controller as rc


class ReplayControllerTests(unittest.TestCase):
    def _temp_root(self):
        temp_root = os.path.join(os.path.dirname(__file__), "_tmp")
        os.makedirs(temp_root, exist_ok=True)
        return temp_root

    def test_load_replay_source_rejects_missing_file(self):
        result = rc.load_replay_source("missing_file.bin", runtime_cfg={"config_file": "config/Radar.cfg"})
        self.assertFalse(result.success)
        self.assertIsInstance(result.source, rc.NullReplaySource)

    def test_load_replay_source_accepts_bin_placeholder(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False, dir=self._temp_root()) as handle:
            replay_path = handle.name
        try:
            result = rc.load_replay_source(replay_path, runtime_cfg={"config_file": "config/Radar.cfg"})
            self.assertTrue(result.success)
            self.assertIsInstance(result.source, rc.ReservedBinReplaySource)
            self.assertEqual(result.replay_path, replay_path)
            self.assertFalse(result.source.can_stream())
        finally:
            os.remove(replay_path)


if __name__ == "__main__":
    unittest.main()
