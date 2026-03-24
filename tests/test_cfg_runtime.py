import unittest
from pathlib import Path

import numpy as np

import DSP
import real_time_process as rtp
from iwr6843_tlv.detected_points import IWR6843AOP_TLV


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADAR_CFG = PROJECT_ROOT / "config" / "Radar.cfg"


class RadarCfgRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = IWR6843AOP_TLV(connect=False)._initialize(str(RADAR_CFG))

    def test_radar_cfg_parses_expected_runtime_fields(self):
        config = self.config
        self.assertEqual(config["num_tx"], 3)
        self.assertEqual(config["num_rx"], 4)
        self.assertEqual(config["num_loops"], 64)
        self.assertEqual(config["num_adc_samples_raw"], 128)
        self.assertEqual(config["num_chirps_per_frame"], 192)
        self.assertEqual(config["num_doppler_bins"], 64)
        self.assertEqual(config["num_range_bins"], 128)
        self.assertEqual(config["frame_length"], 128 * 64 * 3 * 4 * 2)
        self.assertAlmostEqual(config["range_resolution_m"], 0.0450721, places=6)
        self.assertAlmostEqual(config["doppler_resolution_mps"], 0.194340796, places=6)
        self.assertEqual(config["aoa_fov"]["azimuth"], (-90.0, 90.0))
        self.assertEqual(config["aoa_fov"]["elevation"], (-90.0, 90.0))

    def test_dsp_apply_runtime_config_tracks_cfg_values(self):
        DSP.apply_runtime_config(self.config)
        self.assertEqual(DSP.NUM_ADC_SAMPLES_RAW, 128)
        self.assertEqual(DSP.numRangeBins, 128)
        self.assertEqual(DSP.numDopplerBins, 64)
        self.assertAlmostEqual(DSP.RANGE_RESOLUTION, self.config["range_resolution_m"], places=6)
        self.assertAlmostEqual(DSP.DOPPLER_RESOLUTION, self.config["doppler_resolution_mps"], places=6)
        self.assertEqual(DSP.range_azimuth.shape[1], 128)
        self.assertEqual(DSP.range_elevation.shape[1], 128)

    def test_real_time_process_apply_runtime_config_tracks_cfg_values(self):
        rtp.apply_runtime_config(self.config)
        self.assertEqual(rtp.adc_sample, 128)
        self.assertEqual(rtp.chirp, 64)
        self.assertEqual(rtp.tx_num, 3)
        self.assertEqual(rtp.rx_num, 4)
        self.assertEqual(rtp.range_bins, 128)
        self.assertEqual(rtp.frame_length, 128 * 64 * 3 * 4 * 2)
        self.assertEqual(rtp.range_angle_padding_size, [128, 64, 64])
        self.assertEqual(len(rtp.b), rtp.frame_length * 2)

    def test_rda_time_respects_runtime_micro_doppler_shape(self):
        DSP.apply_runtime_config(self.config)
        rng = np.random.default_rng(0)
        adc_data = (
            rng.standard_normal(
                (
                    self.config["num_doppler_bins"],
                    self.config["num_range_bins"],
                    self.config["num_tx"] * self.config["num_rx"],
                )
            )
            + 1j
            * rng.standard_normal(
                (
                    self.config["num_doppler_bins"],
                    self.config["num_range_bins"],
                    self.config["num_tx"] * self.config["num_rx"],
                )
            )
        ).astype(np.complex64)

        _, _, dti = DSP.RDA_Time(
            adc_data,
            clutter_removal_enabled=False,
        )

        self.assertEqual(dti.shape, (1, self.config["num_doppler_bins"]))


if __name__ == "__main__":
    unittest.main()
