import queue
import unittest
from unittest import mock

import numpy as np

import real_time_process as rtp


class DataProcessorTests(unittest.TestCase):
    def _make_runtime_cfg(self):
        return {
            "num_adc_samples_raw": 128,
            "num_range_bins": 128,
            "num_doppler_bins": 64,
            "num_tx": 3,
            "num_rx": 4,
            "frame_length": 128 * 64 * 3 * 4 * 2,
            "frame_periodicity_ms": 50.0,
        }

    def test_process_frame_data_uses_aligned_features_only(self):
        processor = rtp.DataProcessor(
            "Processor",
            self._make_runtime_cfg(),
            queue.Queue(),
            pointcloud_queue=queue.Queue(maxsize=2),
            ml_feature_queue=queue.Queue(maxsize=2),
        )
        frame = np.ones((64, 128, 12), dtype=np.complex64)
        rd = np.ones((128, 64), dtype=np.float32)
        ra = np.full((361, 128), 2.0, dtype=np.float32)
        re = np.full((361, 128), 3.0, dtype=np.float32)
        pointcloud = np.array([[1.0, 0.1, 0.2, 0.3]], dtype=np.float32)

        with mock.patch.object(
            rtp,
            "extract_training_aligned_frame_features",
            return_value=(rd, ra, re),
        ) as mock_extract, mock.patch.object(
            rtp.DSP,
            "extract_pointcloud_from_angle_maps",
            return_value=pointcloud,
        ) as mock_pointcloud, mock.patch.object(
            rtp.DSP,
            "RDA_Time",
            side_effect=AssertionError("legacy heatmap DSP should not run"),
        ) as mock_rda, mock.patch.object(
            rtp.DSP,
            "Range_Angle",
            side_effect=AssertionError("legacy angle DSP should not run"),
        ) as mock_range_angle, mock.patch.object(
            rtp.gl,
            "get_value",
            return_value=0.45,
        ):
            processor.process_frame_data(frame, 5)

        mock_extract.assert_called_once_with(frame, processor.runtime_cfg)
        mock_rda.assert_not_called()
        mock_range_angle.assert_not_called()

        ml_frame = processor.ml_feature_queue.get_nowait()
        self.assertEqual(ml_frame.metadata["frame_index"], 5)
        self.assertEqual(ml_frame.timestamp_start_ms, 200.0)
        self.assertEqual(ml_frame.timestamp_end_ms, 250.0)
        np.testing.assert_array_equal(ml_frame.RDT, rd)
        np.testing.assert_array_equal(ml_frame.ART, ra)
        np.testing.assert_array_equal(ml_frame.ERT, re)

        queued_pointcloud = processor.pointcloud_queue.get_nowait()
        np.testing.assert_array_equal(queued_pointcloud, pointcloud)

        call_args = mock_pointcloud.call_args
        np.testing.assert_array_equal(call_args.args[0], ra)
        np.testing.assert_array_equal(call_args.args[1], re)
        self.assertIsNot(call_args.args[0], ra)
        self.assertIsNot(call_args.args[1], re)
        self.assertEqual(call_args.kwargs["threshold_ratio"], 0.45)

    def test_ml_queue_still_tracks_dropped_frames(self):
        processor = rtp.DataProcessor(
            "Processor",
            self._make_runtime_cfg(),
            queue.Queue(),
            ml_feature_queue=queue.Queue(maxsize=1),
        )
        frame = np.ones((64, 128, 12), dtype=np.complex64)
        rd = np.ones((128, 64), dtype=np.float32)
        ra = np.ones((361, 128), dtype=np.float32)
        re = np.ones((361, 128), dtype=np.float32)

        with mock.patch.object(
            rtp,
            "extract_training_aligned_frame_features",
            return_value=(rd, ra, re),
        ):
            processor.process_frame_data(frame, 1)
            processor.process_frame_data(frame, 2)

        self.assertEqual(processor.dropped_ml_frames, 1)
        latest_frame = processor.ml_feature_queue.get_nowait()
        self.assertEqual(latest_frame.metadata["frame_index"], 2)
        self.assertEqual(latest_frame.metadata["dropped_ml_frames"], 1)
        self.assertEqual(latest_frame.metadata["dropped_on_enqueue"], 1)
        self.assertEqual(latest_frame.metadata["warning"], "ML lagging")


if __name__ == "__main__":
    unittest.main()
