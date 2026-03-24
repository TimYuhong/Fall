import os
import threading as th
from ctypes import POINTER, c_int, c_short, cast, cdll

import numpy as np

import DSP
from runtime import fall_predictor as fp
from runtime.aligned_features import extract_training_aligned_frame_features
from support import globalvar as gl


IQ_CHANNEL = 2

# Radar.cfg defaults: 128 ADC samples, 64 chirps per frame, 3 TX, 4 RX.
adc_sample = 128
chirp = 64
tx_num = 3
rx_num = 4
range_bins = 128
frame_length = adc_sample * chirp * tx_num * rx_num * IQ_CHANNEL
range_angle_padding_size = [range_bins, 64, 64]

dll = cdll.LoadLibrary('libs/UDPCAPTUREADCRAWDATA.dll')
POINTCLOUD_DEBUG_LOGS = str(os.environ.get("RADAR_DEBUG_POINTCLOUD", "0")).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

a = None
b = None
a_ctypes_ptr = None
b_ctypes_ptr = None


def _allocate_capture_buffers():
    global a, b, a_ctypes_ptr, b_ctypes_ptr
    a = np.zeros(1, dtype=np.int32)
    b = np.zeros(frame_length * 2, dtype=np.int16)
    a_ctypes_ptr = cast(a.ctypes.data, POINTER(c_int))
    b_ctypes_ptr = cast(b.ctypes.data, POINTER(c_short))


def apply_runtime_config(runtime_cfg=None):
    runtime_cfg = runtime_cfg or {}

    global adc_sample, chirp, tx_num, rx_num
    global range_bins, frame_length, range_angle_padding_size

    adc_sample = int(runtime_cfg.get('num_adc_samples_raw', adc_sample))
    chirp = int(runtime_cfg.get('num_doppler_bins', chirp))
    tx_num = int(runtime_cfg.get('num_tx', tx_num))
    rx_num = int(runtime_cfg.get('num_rx', rx_num))
    range_bins = int(runtime_cfg.get('num_range_bins', range_bins))
    frame_length = int(
        runtime_cfg.get(
            'frame_length',
            adc_sample * chirp * tx_num * rx_num * IQ_CHANNEL,
        )
    )
    range_angle_padding_size = [range_bins, 64, 64]
    _allocate_capture_buffers()


apply_runtime_config()


class UdpListener(th.Thread):
    def __init__(self, name, bin_data, data_frame_length):
        th.Thread.__init__(self, name=name)
        self.bin_data = bin_data
        self.frame_length = data_frame_length

    def run(self):
        global a_ctypes_ptr, b_ctypes_ptr
        dll.captureudp(a_ctypes_ptr, b_ctypes_ptr, self.frame_length)


class DataProcessor(th.Thread):
    def __init__(
        self,
        name,
        config,
        bin_queue,
        rti_queue=None,
        dti_queue=None,
        rdi_queue=None,
        rai_queue=None,
        rei_queue=None,
        pointcloud_queue=None,
        ml_feature_queue=None,
    ):
        th.Thread.__init__(self, name=name)
        if isinstance(config, dict):
            self.runtime_cfg = dict(config)
            self.adc_sample = int(config['num_adc_samples_raw'])
            self.range_bins = int(config['num_range_bins'])
            self.chirp_num = int(config['num_doppler_bins'])
            self.tx_num = int(config['num_tx'])
            self.rx_num = int(config['num_rx'])
            self.frame_length = int(config['frame_length'])
            self.frame_periodicity_ms = float(config.get('frame_periodicity_ms', 50.0))
        else:
            self.runtime_cfg = {}
            self.adc_sample = config[0]
            self.chirp_num = config[1]
            self.tx_num = config[2]
            self.rx_num = config[3]
            self.range_bins = self.adc_sample
            self.frame_length = self.adc_sample * self.chirp_num * self.tx_num * self.rx_num * IQ_CHANNEL
            self.frame_periodicity_ms = 50.0

        self.range_angle_padding_size = [self.range_bins, 64, 64]
        self.bin_queue = bin_queue
        self.rti_queue = rti_queue
        self.dti_queue = dti_queue
        self.rdi_queue = rdi_queue
        self.rai_queue = rai_queue
        self.rei_queue = rei_queue
        self.pointcloud_queue = pointcloud_queue
        self.ml_feature_queue = ml_feature_queue
        self.dropped_ml_frames = 0

    def _enqueue_ml_frame(self, rd_aligned, ra_aligned, re_aligned, frame_index):
        if self.ml_feature_queue is None or not self.runtime_cfg:
            return

        frame_end_ms = frame_index * self.frame_periodicity_ms
        frame_start_ms = max(frame_end_ms - self.frame_periodicity_ms, 0.0)
        metadata = {
            'frame_index': frame_index,
            'dropped_ml_frames': self.dropped_ml_frames,
            'dropped_on_enqueue': 0,
        }
        ml_frame = fp.MLFeatureFrame(
            timestamp_start_ms=frame_start_ms,
            timestamp_end_ms=frame_end_ms,
            RDT=rd_aligned,
            ART=ra_aligned,
            ERT=re_aligned,
            metadata=metadata,
        )
        if self.ml_feature_queue.full():
            try:
                self.ml_feature_queue.get_nowait()
                self.dropped_ml_frames += 1
                ml_frame.metadata['dropped_ml_frames'] = self.dropped_ml_frames
                ml_frame.metadata['dropped_on_enqueue'] = 1
                ml_frame.metadata['warning'] = 'ML lagging'
            except Exception:
                pass
        self.ml_feature_queue.put(ml_frame)

    def _enqueue_pointcloud(self, ra_aligned, re_aligned, frame_index):
        if self.pointcloud_queue is None:
            return

        threshold_ratio = gl.get_value('pointcloud_threshold', 0.1)
        pointcloud = DSP.extract_pointcloud_from_angle_maps(
            np.array(ra_aligned, copy=True),
            np.array(re_aligned, copy=True),
            threshold_ratio=threshold_ratio,
        )

        if POINTCLOUD_DEBUG_LOGS and frame_index % 50 == 0:
            ra_max = np.max(ra_aligned) if ra_aligned.size > 0 else 0
            re_max = np.max(re_aligned) if re_aligned.size > 0 else 0
            ra_shape = ra_aligned.shape if hasattr(ra_aligned, 'shape') else 'N/A'
            re_shape = re_aligned.shape if hasattr(re_aligned, 'shape') else 'N/A'
            point_count = len(pointcloud) if pointcloud is not None and len(pointcloud) > 0 else 0
            print(
                f"[点云提取] 帧{frame_index}: "
                f"RA形状={ra_shape}, 最大值={ra_max:.2f} | "
                f"RE形状={re_shape}, 最大值={re_max:.2f} | "
                f"阈值={threshold_ratio:.2f} | 提取点数={point_count}"
            )

        if pointcloud is not None and len(pointcloud) > 0:
            if self.pointcloud_queue.full():
                try:
                    self.pointcloud_queue.get_nowait()
                except Exception:
                    pass
            self.pointcloud_queue.put(pointcloud)

    def process_frame_data(self, data, frame_index):
        rd_aligned, ra_aligned, re_aligned = extract_training_aligned_frame_features(
            data,
            self.runtime_cfg,
        )
        self._enqueue_ml_frame(rd_aligned, ra_aligned, re_aligned, frame_index)
        self._enqueue_pointcloud(ra_aligned, re_aligned, frame_index)

    def run(self):
        global frame_count
        frame_count = 0
        lastflar = 0
        while True:
            if lastflar != a_ctypes_ptr[0]:
                lastflar = a_ctypes_ptr[0]
                data = np.array(
                    b_ctypes_ptr[
                        self.frame_length * (1 - a_ctypes_ptr[0]):self.frame_length * (2 - a_ctypes_ptr[0])
                    ]
                )

                data = np.reshape(data, [-1, 4])
                data = data[:, 0:2:] + 1j * data[:, 2::]
                data = np.reshape(data, [self.chirp_num * self.tx_num, -1, self.adc_sample])
                data = data.transpose([0, 2, 1])

                tx_slices = [
                    data[tx_index:self.chirp_num * self.tx_num:self.tx_num, :, :]
                    for tx_index in range(self.tx_num)
                ]
                data = np.concatenate(tx_slices, axis=2)

                frame_count += 1

                try:
                    self.process_frame_data(data, frame_count)
                except Exception as exc:
                    if POINTCLOUD_DEBUG_LOGS or frame_count % 50 == 0:
                        import traceback

                        print(f"[实时特征链错误] 帧{frame_count}: {exc}")
                        traceback.print_exc()
