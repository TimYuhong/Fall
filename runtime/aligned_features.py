"""Training-aligned RD/RA/RE feature extraction shared by offline and realtime code."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

import dsp
import dsp.angle_estimation as Angle_dsp
import dsp.compensation as Compensation
from dsp.doppler_processing import doppler_processing
from dsp.range_processing import range_processing
from dsp.utils import Window


DEFAULT_ANGLE_RESOLUTION_DEG = 0.5
DEFAULT_CLIP_FRAMES = 100
DEFAULT_FRAME_PERIODICITY_MS = 50.0
AZIMUTH_VRX_INDICES = (2, 3, 4, 5)
ELEVATION_VRX_INDICES = (4, 5, 10, 11)
ELEVATION_PHASE_SIGNS = np.array([[1], [-1], [1], [-1]], dtype=np.complex64)


def resolve_angle_resolution_deg(
    config_params: Dict[str, Any],
    angle_resolution_deg: Optional[float] = None,
) -> float:
    if angle_resolution_deg is not None:
        return float(angle_resolution_deg)
    return float(config_params.get("aoa_angle_res", DEFAULT_ANGLE_RESOLUTION_DEG))


def resolve_angle_range_deg(config_params: Dict[str, Any]) -> float:
    aoa_fov = config_params.get("aoa_fov") or {}
    azimuth_fov = aoa_fov.get("azimuth", (-90.0, 90.0))
    return max(abs(float(azimuth_fov[0])), abs(float(azimuth_fov[1])))


def resolve_angle_bins(
    config_params: Dict[str, Any],
    angle_resolution_deg: Optional[float] = None,
) -> int:
    angle_res = resolve_angle_resolution_deg(config_params, angle_resolution_deg)
    angle_range = resolve_angle_range_deg(config_params)
    return int((angle_range * 2) // angle_res + 1)


def build_training_feature_shapes(
    config_params: Dict[str, Any],
    clip_frames: int = DEFAULT_CLIP_FRAMES,
    angle_resolution_deg: Optional[float] = None,
) -> Dict[str, Tuple[int, ...]]:
    range_bins = int(config_params["num_range_bins"])
    doppler_bins = int(config_params["num_doppler_bins"])
    angle_bins = resolve_angle_bins(config_params, angle_resolution_deg)
    return {
        "RD": (int(clip_frames), range_bins, doppler_bins),
        "RA": (int(clip_frames), angle_bins, range_bins),
        "RE": (int(clip_frames), angle_bins, range_bins),
    }


def extract_training_aligned_frame_features(
    frame: np.ndarray,
    config_params: Dict[str, Any],
    angle_resolution_deg: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract one RD/RA/RE frame using the same contract as offline training."""

    if frame.ndim != 3:
        raise ValueError(f"Expected frame with 3 dims, got shape={frame.shape}")
    if frame.shape[2] < max(ELEVATION_VRX_INDICES) + 1:
        raise ValueError(
            f"Expected at least {max(ELEVATION_VRX_INDICES) + 1} virtual antennas, "
            f"got shape={frame.shape}"
        )

    num_range_bins = int(config_params["num_range_bins"])
    num_adc_samples_raw = int(config_params["num_adc_samples_raw"])
    num_tx = int(config_params["num_tx"])
    angle_res = resolve_angle_resolution_deg(config_params, angle_resolution_deg)
    angle_range = resolve_angle_range_deg(config_params)
    angle_bins = resolve_angle_bins(config_params, angle_res)

    _, steering_vec = Angle_dsp.gen_steering_vec(angle_range, angle_res, len(AZIMUTH_VRX_INDICES))

    adc_data = np.transpose(frame, [0, 2, 1])  # (chirps, rx, adc)
    radar_cube = range_processing(
        2 * adc_data[:, :, :num_adc_samples_raw],
        Window.HANNING,
        2,
        fft_size=num_range_bins,
    )
    radar_cube = Compensation.clutter_removal(radar_cube, axis=0)

    range_doppler_fft, _ = doppler_processing(
        radar_cube,
        num_tx_antennas=num_tx,
        interleaved=False,
        clutter_removal_enabled=False,
        window_type_2d=Window.HANNING,
        accumulate=False,
    )

    rd = np.abs(range_doppler_fft).mean(axis=1)
    rd = np.fft.fftshift(rd, axes=1)
    rd = np.log1p(rd).astype(np.float32, copy=False)

    range_azimuth = np.zeros((angle_bins, num_range_bins), dtype=np.float32)
    range_elevation = np.zeros((angle_bins, num_range_bins), dtype=np.float32)

    for i in range(num_range_bins):
        range_azimuth[:, i], _ = dsp.aoa_capon(
            radar_cube[:, list(AZIMUTH_VRX_INDICES), i].T,
            steering_vec,
            magnitude=True,
        )
        range_elevation[:, i], _ = dsp.aoa_capon(
            radar_cube[:, list(ELEVATION_VRX_INDICES), i].T * ELEVATION_PHASE_SIGNS,
            steering_vec,
            magnitude=True,
        )

    ra = np.flip(np.abs(range_azimuth), axis=1).astype(np.float32, copy=False)
    re = np.flip(np.abs(range_elevation), axis=1).astype(np.float32, copy=False)
    return rd, ra, re
