"""Tests for offline_feature_extractor module (import & unit-level)."""

import os
import sys
import unittest

import numpy as np

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from offline import feature_extractor as ofe


class CfgParsingTests(unittest.TestCase):
    """测试 cfg 解析是否产出正确的 config_params 字典。"""

    CFG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')),
        'config/Radar.cfg not found',
    )
    def test_parse_cfg_returns_required_keys(self):
        params = ofe.parse_cfg(self.CFG_PATH)
        required_keys = [
            'num_tx', 'num_rx', 'num_adc_samples_raw', 'num_range_bins',
            'num_doppler_bins', 'num_chirps_per_frame', 'frame_length',
            'range_resolution_m', 'doppler_resolution_mps',
        ]
        for key in required_keys:
            self.assertIn(key, params, f'config_params 缺少必要字段: {key}')

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')),
        'config/Radar.cfg not found',
    )
    def test_parse_cfg_values_positive(self):
        params = ofe.parse_cfg(self.CFG_PATH)
        self.assertGreater(int(params['num_tx']), 0)
        self.assertGreater(int(params['num_rx']), 0)
        self.assertGreater(int(params['frame_length']), 0)


class DecodeBinTests(unittest.TestCase):
    """测试原始帧解码逻辑（使用合成数据）。"""

    def _make_synthetic_bin(self, config_params, num_frames=3):
        """生成与 .bin 文件格式一致的合成数据。"""
        frame_length = int(config_params['frame_length'])
        raw = np.random.randint(-1000, 1000, size=num_frames * frame_length, dtype=np.int16)
        return raw

    def test_decode_synthetic_bin(self):
        """验证合成数据的 shape 正确性。"""
        config_params = {
            'num_adc_samples_raw': 64,
            'num_range_bins': 64,
            'num_chirps_per_frame': 192,  # 64 chirps * 3 TX
            'num_tx': 3,
            'num_rx': 4,
            'frame_length': 64 * 192 * 4 * 2,  # adc * chirps_per_frame * rx * IQ
        }
        num_test_frames = 3
        raw = self._make_synthetic_bin(config_params, num_test_frames)

        # 写入临时文件
        tmp_bin = os.path.join(os.path.dirname(__file__), '_test_synthetic.bin')
        try:
            raw.tofile(tmp_bin)
            frames = ofe.decode_bin_frames(tmp_bin, config_params)
            self.assertEqual(frames.shape[0], num_test_frames)

            # 验证输出 shape: (chirps, adc_samples, virt_ant)
            # chirps = num_chirps_per_frame / num_tx = 192 / 3 = 64
            # virt_ant = num_rx * num_tx = 4 * 3 = 12
            expected_chirps = int(config_params['num_chirps_per_frame']) // int(config_params['num_tx'])
            expected_adc = int(config_params['num_adc_samples_raw'])
            expected_virt = int(config_params['num_rx']) * int(config_params['num_tx'])
            self.assertEqual(frames.shape[1], expected_chirps)
            self.assertEqual(frames.shape[2], expected_adc)
            self.assertEqual(frames.shape[3], expected_virt)
            self.assertTrue(np.iscomplexobj(frames))
        finally:
            if os.path.exists(tmp_bin):
                os.remove(tmp_bin)


class ExtractorInitTests(unittest.TestCase):
    """测试 OfflineExtractor 初始化。"""

    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')),
        'config/Radar.cfg not found',
    )
    def test_extractor_init(self):
        cfg = os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')
        ext = ofe.OfflineExtractor(cfg)
        self.assertIsNotNone(ext.config_params)
        self.assertFalse(ext._dsp_initialized)


class SharedAlignedFeatureTests(unittest.TestCase):
    @unittest.skipUnless(
        os.path.isfile(os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')),
        'config/Radar.cfg not found',
    )
    def test_shared_helper_matches_offline_process_frame(self):
        cfg = os.path.join(os.path.dirname(__file__), '..', 'config', 'Radar.cfg')
        ext = ofe.OfflineExtractor(cfg)
        config_params = ext.config_params
        rng = np.random.default_rng(0)
        frame = (
            rng.standard_normal(
                (
                    int(config_params['num_doppler_bins']),
                    int(config_params['num_adc_samples_raw']),
                    int(config_params['num_rx']) * int(config_params['num_tx']),
                )
            )
            + 1j
            * rng.standard_normal(
                (
                    int(config_params['num_doppler_bins']),
                    int(config_params['num_adc_samples_raw']),
                    int(config_params['num_rx']) * int(config_params['num_tx']),
                )
            )
        ).astype(np.complex64)

        rd_from_extractor, ra_from_extractor, re_from_extractor = ext._process_frame(frame)
        rd_from_helper, ra_from_helper, re_from_helper = ofe.extract_training_aligned_frame_features(
            frame,
            ext.config_params,
        )

        np.testing.assert_allclose(rd_from_extractor, rd_from_helper, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ra_from_extractor, ra_from_helper, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(re_from_extractor, re_from_helper, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
