"""Offline radar feature extraction — point cloud + RD/RA/RE feature maps.

Pipeline per frame:
    raw bin -> DSP (range FFT + doppler FFT + Capon AOA)
             -> RD  (Range-Doppler map)
             -> RA  (Range-Azimuth map)
             -> RE  (Range-Elevation map)
             -> PC  (3-D point cloud via grid detection on RA/RE)

Heatmap images (RTI/DTI) are for GUI display only and are NOT extracted.

Output layout per event clip::

    out_root/<label>/<clip_id>/
        RD.npy      # (num_frames, num_doppler_bins, num_range_bins)  float32
        RA.npy      # (num_frames, angle_bins, num_range_bins)        float32
        RE.npy      # (num_frames, angle_bins, num_range_bins)        float32
        PC.npy      # (total_points, 4) [range, x, y, z]             float32
        meta.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_CFG_PATH = os.path.join(PROJECT_ROOT, "config", "Radar.cfg")
EXTERNAL_CFG_PATH = r"f:\test\cfg\Radar.cfg"

EXPECTED_CAPTURE_LAYOUT = {
    "expected_capture_bytes": 39321600,
    "expected_frame_bytes": 393216,
    "expected_capture_frames": 100,
    "expected_frame_rate_fps": 20.0,
    "expected_duration_seconds": 5.0,
}

RAW_INT16_BYTES = np.dtype(np.int16).itemsize
JSONL_REQUIRED_FIELDS = ("bin_path", "label")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runtime.aligned_features import extract_training_aligned_frame_features


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ManifestSample:
    bin_path: str
    label: str
    clip_id: str
    frame_start: int = 0
    frame_end: Optional[int] = None
    subject_id: str = ""
    session_id: str = ""
    scene: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BinLayoutInfo:
    bin_path: str
    total_bytes: int
    frame_length_int16: int
    frame_bytes: int
    num_frames: int
    expected_capture_bytes: int
    expected_capture_frames: int


@dataclass
class EventFeatures:
    """Feature tensors for one event clip.

    RD : (num_frames, num_doppler_bins, num_range_bins)
    RA : (num_frames, angle_bins,       num_range_bins)
    RE : (num_frames, angle_bins,       num_range_bins)
    PC : (total_points, 4)  columns: [range, x, y, z]
    """
    sample: ManifestSample
    RD: np.ndarray
    RA: np.ndarray
    RE: np.ndarray
    PC: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.RD.shape[0])

    @property
    def feature_shapes(self) -> Dict[str, Tuple[int, ...]]:
        return {
            "RD": tuple(self.RD.shape),
            "RA": tuple(self.RA.shape),
            "RE": tuple(self.RE.shape),
            "PC": tuple(self.PC.shape),
        }


@dataclass
class ExtractionResult:
    sample: ManifestSample
    output_dir: str
    cfg_path: str
    cfg_sha256: str
    num_frames: int
    feature_shapes: Dict[str, Tuple[int, ...]]
    frame_periodicity_ms: float
    range_resolution_m: float
    doppler_resolution_mps: float
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def resolve_default_cfg_path() -> str:
    if os.path.exists(EXTERNAL_CFG_PATH):
        return EXTERNAL_CFG_PATH
    return REPO_CFG_PATH


def compute_file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_path_component(value: str, default: str) -> str:
    value = (value or "").strip()
    if not value:
        value = default
    value = re.sub(r'[\\/:*?"<>|]+', "_", value)
    value = re.sub(r"\s+", "_", value)
    return value or default


def derive_clip_id(
    bin_path: str,
    frame_start: int = 0,
    frame_end: Optional[int] = None,
) -> str:
    basename = os.path.splitext(os.path.basename(bin_path))[0]
    if frame_start == 0 and frame_end is None:
        return basename
    end_token = "EOF" if frame_end is None else f"{frame_end:05d}"
    return f"{basename}_f{frame_start:05d}_{end_token}"


def ensure_cfg_matches_repo(
    cfg_path: str, repo_cfg_path: str = REPO_CFG_PATH
) -> str:
    cfg_path = os.path.abspath(cfg_path)
    cfg_sha256 = compute_file_sha256(cfg_path)
    if os.path.exists(repo_cfg_path):
        if cfg_sha256 != compute_file_sha256(repo_cfg_path):
            raise ValueError(
                f"cfg mismatch: {cfg_path} vs {repo_cfg_path}. "
                "Offline extraction must use the same Radar.cfg as the realtime app."
            )
    return cfg_sha256


def parse_cfg(cfg_path: str) -> Dict[str, Any]:
    from iwr6843_tlv.detected_points import IWR6843AOP_TLV
    return IWR6843AOP_TLV(connect=False, config_file=cfg_path)._initialize(cfg_path)


def inspect_bin_layout(
    bin_path: str, config_params: Dict[str, Any]
) -> BinLayoutInfo:
    total_bytes = os.path.getsize(bin_path)
    frame_length_int16 = int(config_params["frame_length"])
    frame_bytes = frame_length_int16 * RAW_INT16_BYTES
    if total_bytes % frame_bytes != 0:
        raise ValueError(
            f"bin size is not an integer number of frames: "
            f"total_bytes={total_bytes}, frame_bytes={frame_bytes}"
        )
    return BinLayoutInfo(
        bin_path=os.path.abspath(bin_path),
        total_bytes=total_bytes,
        frame_length_int16=frame_length_int16,
        frame_bytes=frame_bytes,
        num_frames=total_bytes // frame_bytes,
        expected_capture_bytes=EXPECTED_CAPTURE_LAYOUT["expected_capture_bytes"],
        expected_capture_frames=EXPECTED_CAPTURE_LAYOUT["expected_capture_frames"],
    )


def decode_bin_frames(
    bin_path: str, config_params: Dict[str, Any]
) -> np.ndarray:
    """Decode a DCA1000 raw .bin into DSP-ready complex frames.

    Uses memory-mapping to avoid loading the entire file into RAM.

    Returns
    -------
    np.ndarray
        Shape: (num_frames, num_doppler_bins, num_adc_samples_raw, num_virtual_ant)
    """
    adc_samples = int(config_params["num_adc_samples_raw"])
    num_chirps = int(config_params["num_chirps_per_frame"])
    num_tx = int(config_params["num_tx"])
    num_rx = int(config_params["num_rx"])  # Fix: was missing, relied on global scope
    frame_length = int(config_params["frame_length"])
    frame_bytes = frame_length * RAW_INT16_BYTES

    total_bytes = os.path.getsize(bin_path)
    num_frames = total_bytes // frame_bytes
    if num_frames == 0:
        raise ValueError(
            f"No complete frames in {bin_path} "
            f"(total_bytes={total_bytes}, frame_bytes={frame_bytes})"
        )

    # Use memmap: only the accessed pages are loaded from disk
    raw = np.memmap(bin_path, dtype=np.int16, mode='r',
                    shape=(num_frames, frame_length))
    frames: List[np.ndarray] = []
    for i in range(num_frames):
        frame_raw = np.array(raw[i])  # copy one frame into RAM, then release
        # adcbufCfg: SAMPLE_SWAP=1 (Q-First), CHAN_INTERLEAVE=1 (Non-Interleaved)
        # Non-Interleaved layout: [chirp0_rx0_s0..sN, chirp0_rx1_s0..sN, ..., chirpM_rxK_s0..sN]
        # Each int16 pair: Q first, then I
        Q = frame_raw[0::2].astype(np.float32)
        I = frame_raw[1::2].astype(np.float32)
        complex_1d = (I + 1j * Q).astype(np.complex64)
        # Reshape to (num_chirps_per_frame, num_rx, adc_samples)
        cs = complex_1d.reshape(num_chirps, num_rx, adc_samples)
        # Separate TDM TX: interleave chirps by TX index
        # chirpCfg order: TX0(mask=1), TX2(mask=4), TX1(mask=2)
        # Result: (num_doppler_bins, adc_samples, num_vrx)
        frame = np.concatenate(
            [cs[tx::num_tx, :, :].transpose(0, 2, 1) for tx in range(num_tx)], axis=2
        )
        frames.append(frame)
    del raw  # release memmap
    return np.asarray(frames)


def decode_bin_frames_multi(
    bin_path: str,
    config_params: Dict[str, Any],
    frame_start: Optional[int] = None,
    frame_end: Optional[int] = None,
) -> np.ndarray:
    """Decode a DCA1000 recording that may be split into multiple sequential .bin files.

    DCA1000 splits long recordings into Raw_0.bin, Raw_1.bin, Raw_2.bin, ...
    This function auto-detects sibling files and only decodes the requested frame range.

    Parameters
    ----------
    frame_start : int, optional
        First frame index (global, across all files). Default: 0.
    frame_end : int, optional
        Exclusive end frame index. Default: all frames.

    Returns
    -------
    np.ndarray
        Shape: (frame_end-frame_start, num_doppler_bins, num_adc_samples_raw, num_virtual_ant)
    """
    import re
    bin_path = os.path.abspath(bin_path)
    dirpath = os.path.dirname(bin_path)
    basename = os.path.basename(bin_path)

    m = re.match(r'^(.+_Raw)_(\d+)\.bin$', basename)
    if m:
        stem = m.group(1)
        siblings = sorted(
            [f for f in os.listdir(dirpath)
             if re.match(rf'^{re.escape(stem)}_\d+\.bin$', f)],
            key=lambda f: int(re.search(r'(\d+)\.bin$', f).group(1))
        )
        bin_files = [os.path.join(dirpath, s) for s in siblings]
    else:
        bin_files = [bin_path]

    frame_length = int(config_params["frame_length"])
    frame_bytes = frame_length * RAW_INT16_BYTES

    # Compute per-file frame counts without loading data
    file_frame_counts = [
        os.path.getsize(bp) // frame_bytes for bp in bin_files
    ]
    total_frames = sum(file_frame_counts)

    fs = frame_start if frame_start is not None else 0
    fe = frame_end if frame_end is not None else total_frames
    fs = max(0, fs)
    fe = min(fe, total_frames)

    # Collect only the frames in [fs, fe) across files
    all_frames: List[np.ndarray] = []
    offset = 0
    for bp, nf in zip(bin_files, file_frame_counts):
        file_start = offset
        file_end = offset + nf
        # Overlap of [fs, fe) with [file_start, file_end)
        local_start = max(fs, file_start) - file_start
        local_end = min(fe, file_end) - file_start
        if local_start < local_end:
            raw = np.memmap(bp, dtype=np.int16, mode='r',
                            shape=(nf, frame_length))
            chunk = np.array(raw[local_start:local_end])  # copy needed slice
            del raw
            all_frames.append(_decode_raw_chunk(chunk, config_params))
        offset = file_end
        if offset >= fe:
            break

    return np.concatenate(all_frames, axis=0) if all_frames else np.empty((0,), dtype=np.float32)


def _decode_raw_chunk(chunk: np.ndarray, config_params: Dict[str, Any]) -> np.ndarray:
    """Decode a (N, frame_length) int16 array into complex DSP frames."""
    adc_samples = int(config_params["num_adc_samples_raw"])
    num_chirps = int(config_params["num_chirps_per_frame"])
    num_tx = int(config_params["num_tx"])
    num_rx = int(config_params["num_rx"])
    frames: List[np.ndarray] = []
    for i in range(chunk.shape[0]):
        frame_raw = chunk[i]
        # adcbufCfg: SAMPLE_SWAP=1 (Q-First), CHAN_INTERLEAVE=1 (Non-Interleaved)
        # Non-Interleaved layout: [chirp0_rx0_s0..sN, chirp0_rx1_s0..sN, ..., chirpM_rxK_s0..sN]
        # Each int16 pair: Q first, then I
        Q = frame_raw[0::2].astype(np.float32)
        I = frame_raw[1::2].astype(np.float32)
        complex_1d = (I + 1j * Q).astype(np.complex64)
        # Reshape to (num_chirps_per_frame, num_rx, adc_samples)
        cs = complex_1d.reshape(num_chirps, num_rx, adc_samples)
        # Separate TDM TX: chirpCfg order TX0(mask=1), TX2(mask=4), TX1(mask=2)
        # Result: (num_doppler_bins, adc_samples, num_vrx)
        frame = np.concatenate(
            [cs[tx::num_tx, :, :].transpose(0, 2, 1) for tx in range(num_tx)], axis=2
        )
        frames.append(frame)
    return np.asarray(frames)


def load_manifest(manifest_path: str) -> List[ManifestSample]:
    samples: List[ManifestSample] = []
    with open(manifest_path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL error at line {lineno}: {exc}") from exc
            missing = [f for f in JSONL_REQUIRED_FIELDS if not payload.get(f)]
            if missing:
                raise ValueError(f"Line {lineno} missing: {', '.join(missing)}")
            fs = int(payload.get("frame_start", 0))
            fe = payload.get("frame_end")
            fe = None if fe in (None, "") else int(fe)
            if fs < 0:
                raise ValueError(f"Line {lineno}: negative frame_start")
            if fe is not None and fe <= fs:
                raise ValueError(f"Line {lineno}: invalid frame range")
            clip_id = payload.get("clip_id") or derive_clip_id(
                payload["bin_path"], frame_start=fs, frame_end=fe
            )
            samples.append(ManifestSample(
                bin_path=os.path.abspath(payload["bin_path"]),
                label=str(payload["label"]),
                clip_id=str(clip_id),
                frame_start=fs,
                frame_end=fe,
                subject_id=str(payload.get("subject_id", "")),
                session_id=str(payload.get("session_id", "")),
                scene=str(payload.get("scene", "")),
                metadata=dict(payload.get("metadata", {})),
            ))
    if not samples:
        raise ValueError(f"No valid samples in manifest: {manifest_path}")
    return samples


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class OfflineExtractor:
    """Extract PC + RD/RA/RE features from raw DCA1000 .bin files.

    RTI and DTI heatmaps are intentionally omitted — they are for GUI display
    only and carry no additional training signal beyond RD/RA/RE.
    """

    def __init__(
        self,
        cfg_path: Optional[str] = None,
        pointcloud_threshold: float = 0.3,
        repo_cfg_path: str = REPO_CFG_PATH,
    ) -> None:
        self.cfg_path = os.path.abspath(cfg_path or resolve_default_cfg_path())
        self.repo_cfg_path = os.path.abspath(repo_cfg_path)
        self.cfg_sha256 = ensure_cfg_matches_repo(self.cfg_path, self.repo_cfg_path)
        self.config_params = parse_cfg(self.cfg_path)
        self.pointcloud_threshold = float(pointcloud_threshold)
        self._dsp_initialized = False
        self.num_range_bins = int(self.config_params["num_range_bins"])
        self.num_doppler_bins = int(self.config_params["num_doppler_bins"])
        self.padding_size = [self.num_range_bins, 64, 64]

    def _init_dsp(self) -> None:
        if self._dsp_initialized:
            return
        import DSP
        DSP.apply_runtime_config(self.config_params)
        self._dsp_initialized = True

    def _reset_dsp_state(self) -> None:
        import DSP
        DSP.rti_queue.clear()
        DSP.rdi_queue.clear()
        DSP.rai_queue.clear()
        DSP.rei_queue.clear()

    def _process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run DSP on one raw frame, return (RD, RA, RE).

        Mirrors DSP.RDA_Time (for RD) and DSP.Range_Angle (for RA/RE) exactly.

        RD : (num_doppler_bins, num_range_bins)   float32  log1p of fftshifted abs
        RA : (angle_bins, num_range_bins)          float32  Capon magnitude, flipped
        RE : (angle_bins, num_range_bins)          float32  Capon magnitude, flipped
        """
        return extract_training_aligned_frame_features(frame, self.config_params)

        import DSP
        import dsp
        import dsp.compensation as Compensation
        from dsp.range_processing import range_processing
        from dsp.doppler_processing import doppler_processing
        from dsp.utils import Window

        # Both RDA_Time and Range_Angle start with the same transpose
        # frame: (num_chirps, num_adc_samples, num_rx)  [from decode_bin_frames]
        adc_data = np.transpose(frame, [0, 2, 1])  # -> (chirps, rx, adc)

        # --- Shared range FFT + clutter removal ---
        # Mirrors: radar_cube = range_processing(2 * adc_data[:, :, :NUM_ADC_SAMPLES_RAW], ...)
        radar_cube = range_processing(
            2 * adc_data[:, :, :DSP.NUM_ADC_SAMPLES_RAW],
            Window.HANNING, 2,
            fft_size=DSP.numRangeBins,
        )  # shape: (chirps, rx, range_bins)
        radar_cube = Compensation.clutter_removal(radar_cube, axis=0)

        # ---- RD map (mirrors DSP.RDA_Time) ----
        range_doppler_fft, _aoa_input = doppler_processing(
            radar_cube,
            num_tx_antennas=DSP.NUM_TX,
            interleaved=False,
            clutter_removal_enabled=False,
            window_type_2d=Window.HANNING,
            accumulate=False,
        )  # shape: (range_bins, num_vrx, doppler_bins)

        # Mirrors: rdi_abs = np.transpose(np.fft.fftshift(np.abs(range_doppler_fft), axes=2), [0,2,1])
        # then rdi_queue.append — we take equivalent single-frame result
        # DSP stores (vrx, doppler, range) after transpose — we use ch0 vrx avg
        rdi_abs = np.abs(range_doppler_fft).mean(axis=1)   # (range_bins, doppler_bins)
        rdi_abs = np.fft.fftshift(rdi_abs, axes=1)         # center zero-doppler
        # Keep (range_bins, doppler_bins) to match DSP.rdi_queue convention
        rd = np.log1p(rdi_abs).astype(np.float32)          # (range_bins, doppler_bins)

        # ---- RA / RE maps: reuse DSP.Range_Angle but capture raw Capon output ----
        # DSP.Range_Angle applies SNR normalization for display; we undo that by
        # saving the raw flip-abs result before the SNR weighting step.
        DSP.rai_queue.clear()
        DSP.rei_queue.clear()

        import dsp
        import dsp.compensation as Compensation

        # Replicate Range_Angle internals without SNR weighting
        adc_data2 = np.transpose(frame, [0, 2, 1])
        rc2 = range_processing(
            2 * adc_data2[:, :, :DSP.NUM_ADC_SAMPLES_RAW],
            Window.HANNING, 2, fft_size=DSP.numRangeBins,
        )
        rc2 = Compensation.clutter_removal(rc2, axis=0)

        bins = DSP.BINS_PROCESSED
        range_azimuth = np.zeros((int(DSP.ANGLE_BINS), bins))
        range_elevation = np.zeros((int(DSP.ANGLE_BINS), bins))
        bw = np.zeros((DSP.VIRT_ANT, bins), dtype=np.complex64)

        for i in range(bins):
            range_azimuth[:, i], bw[:, i] = dsp.aoa_capon(
                rc2[:, [0, 1, 2, 3], i].T, DSP.steering_vec, magnitude=True)
            range_elevation[:, i], bw[:, i] = dsp.aoa_capon(
                (rc2[:, [0, 1, 8, 9], i] * np.array([1, -1, 1, -1])).T,
                DSP.steering_vec, magnitude=True)

        ra = np.flip(np.abs(range_azimuth), axis=1).astype(np.float32)
        re = np.flip(np.abs(range_elevation), axis=1).astype(np.float32)

        return rd, ra, re

    def _extract_event_from_decoded(
        self,
        decoded: np.ndarray,
        bin_path: str,
        label: str,
        clip_id: Optional[str],
        frame_start: int,
        frame_end: Optional[int],
        subject_id: str = "",
        session_id: str = "",
        scene: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventFeatures:
        """Extract features from pre-decoded frames (avoids re-reading bin file)."""
        import DSP
        self._init_dsp()
        self._reset_dsp_state()

        bin_path = os.path.abspath(bin_path)
        total = int(decoded.shape[0])

        if frame_start < 0 or frame_start >= total:
            raise ValueError(f"frame_start={frame_start} out of range (total={total})")
        if frame_end is None:
            frame_end = total
        if frame_end <= frame_start or frame_end > total:
            raise ValueError(f"Invalid frame range [{frame_start}, {frame_end}) total={total}")

        sample = ManifestSample(
            bin_path=bin_path, label=label,
            clip_id=clip_id or derive_clip_id(bin_path, frame_start, frame_end),
            frame_start=frame_start, frame_end=frame_end,
            subject_id=subject_id, session_id=session_id,
            scene=scene, metadata=dict(metadata or {}),
        )

        RD_frames: List[np.ndarray] = []
        RA_frames: List[np.ndarray] = []
        RE_frames: List[np.ndarray] = []

        for idx in range(frame_start, frame_end):
            rd, ra, re = self._process_frame(decoded[idx])
            RD_frames.append(rd)
            RA_frames.append(ra)
            RE_frames.append(re)

        pc_array = np.zeros((0, 4), dtype=np.float32)
        return EventFeatures(
            sample=sample,
            RD=np.stack(RD_frames, axis=0).astype(np.float32, copy=False),
            RA=np.stack(RA_frames, axis=0).astype(np.float32, copy=False),
            RE=np.stack(RE_frames, axis=0).astype(np.float32, copy=False),
            PC=pc_array,
        )

    def extract_event(
        self,
        bin_path: str,
        label: str,
        clip_id: Optional[str] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        subject_id: str = "",
        session_id: str = "",
        scene: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EventFeatures:
        """Extract RD/RA/RE + PC from one event clip."""
        import DSP
        self._init_dsp()
        self._reset_dsp_state()

        bin_path = os.path.abspath(bin_path)
        decoded = decode_bin_frames_multi(
            bin_path, self.config_params,
            frame_start=frame_start, frame_end=frame_end
        )
        total = int(decoded.shape[0])
        # After range-decode, frames are re-indexed from 0
        frame_end = total
        frame_start = 0

        sample = ManifestSample(
            bin_path=bin_path, label=label,
            clip_id=clip_id or derive_clip_id(bin_path, frame_start, frame_end),
            frame_start=frame_start, frame_end=frame_end,
            subject_id=subject_id, session_id=session_id,
            scene=scene, metadata=dict(metadata or {}),
        )

        RD_frames: List[np.ndarray] = []
        RA_frames: List[np.ndarray] = []
        RE_frames: List[np.ndarray] = []
        pc_list: List[np.ndarray] = []

        for idx in range(frame_start, frame_end):
            rd, ra, re = self._process_frame(decoded[idx])
            RD_frames.append(rd)
            RA_frames.append(ra)
            RE_frames.append(re)

        # PC extraction skipped
        pc_array = np.zeros((0, 4), dtype=np.float32)
        return EventFeatures(
            sample=sample,
            RD=np.stack(RD_frames, axis=0).astype(np.float32, copy=False),
            RA=np.stack(RA_frames, axis=0).astype(np.float32, copy=False),
            RE=np.stack(RE_frames, axis=0).astype(np.float32, copy=False),
            PC=pc_array,
        )

    def _build_meta(self, event: EventFeatures) -> Dict[str, Any]:
        # For multi-file recordings (Raw_0/Raw_1/Raw_2), inspect_bin_layout
        # would fail on Raw_0 alone. Use combined frame count instead.
        import re
        bin_path = event.sample.bin_path
        frame_length_int16 = int(self.config_params["frame_length"])
        frame_bytes = frame_length_int16 * RAW_INT16_BYTES
        m = re.match(r'^(.+_Raw)_(\d+)\.bin$', os.path.basename(bin_path))
        if m and int(m.group(2)) == 0:
            # Multi-file: sum all siblings
            dirpath = os.path.dirname(bin_path)
            stem = m.group(1)
            siblings = sorted(
                [f for f in os.listdir(dirpath)
                 if re.match(rf'^{re.escape(stem)}_\d+\.bin$', f)],
                key=lambda f: int(re.search(r'(\d+)\.bin$', f).group(1))
            )
            total_bytes = sum(os.path.getsize(os.path.join(dirpath, s)) for s in siblings)
        else:
            total_bytes = os.path.getsize(bin_path)
        num_frames = total_bytes // frame_bytes
        layout = BinLayoutInfo(
            bin_path=os.path.abspath(bin_path),
            total_bytes=total_bytes,
            frame_length_int16=frame_length_int16,
            frame_bytes=frame_bytes,
            num_frames=num_frames,
            expected_capture_bytes=EXPECTED_CAPTURE_LAYOUT["expected_capture_bytes"],
            expected_capture_frames=EXPECTED_CAPTURE_LAYOUT["expected_capture_frames"],
        )
        fp_ms = float(self.config_params.get("frame_periodicity_ms", 50.0))
        return {
            "bin_path": event.sample.bin_path,
            "cfg_path": self.cfg_path,
            "cfg_sha256": self.cfg_sha256,
            "label": event.sample.label,
            "clip_id": event.sample.clip_id,
            "subject_id": event.sample.subject_id,
            "session_id": event.sample.session_id,
            "scene": event.sample.scene,
            "frame_start": event.sample.frame_start,
            "frame_end": event.sample.frame_end,
            "num_frames": event.num_frames,
            "frame_periodicity_ms": fp_ms,
            "range_resolution_m": float(self.config_params.get("range_resolution_m", 0.0)),
            "doppler_resolution_mps": float(self.config_params.get("doppler_resolution_mps", 0.0)),
            "feature_shapes": {k: list(v) for k, v in event.feature_shapes.items()},
            "capture_layout": {
                **EXPECTED_CAPTURE_LAYOUT,
                "actual_file_bytes": layout.total_bytes,
                "actual_frame_bytes": layout.frame_bytes,
                "actual_num_frames_in_file": layout.num_frames,
            },
            "runtime_cfg_summary": {
                "num_adc_samples_raw": int(self.config_params["num_adc_samples_raw"]),
                "num_range_bins": int(self.config_params["num_range_bins"]),
                "num_chirps_per_frame": int(self.config_params["num_chirps_per_frame"]),
                "num_doppler_bins": int(self.config_params["num_doppler_bins"]),
                "num_tx": int(self.config_params["num_tx"]),
                "num_rx": int(self.config_params["num_rx"]),
                "frame_length": int(self.config_params["frame_length"]),
            },
            "model_contract": {
                "clip_frames": event.num_frames,
                "frame_periodicity_ms": fp_ms,
                "feature_shapes": {k: list(v) for k, v in event.feature_shapes.items()},
                "class_names": event.sample.metadata.get("class_names", []),
                "positive_labels": event.sample.metadata.get("positive_labels", []),
            },
            "source_metadata": dict(event.sample.metadata),
        }

    def save_event(
        self,
        event: EventFeatures,
        output_root: str,
        overwrite: bool = False,
    ) -> ExtractionResult:
        label_dir = sanitize_path_component(event.sample.label, "unlabeled")
        clip_dir = sanitize_path_component(event.sample.clip_id, "clip")
        output_dir = os.path.join(os.path.abspath(output_root), label_dir, clip_dir)
        if os.path.exists(output_dir):
            if not overwrite:
                raise FileExistsError(
                    f"Output dir exists: {output_dir}. Pass --overwrite to replace."
                )
        else:
            os.makedirs(output_dir, exist_ok=True)

        t0 = time.time()
        np.save(os.path.join(output_dir, "RD.npy"), event.RD)
        np.save(os.path.join(output_dir, "RA.npy"), event.RA)
        np.save(os.path.join(output_dir, "RE.npy"), event.RE)
        # PC skipped to reduce computation time
        with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(self._build_meta(event), fh, ensure_ascii=False, indent=2)

        return ExtractionResult(
            sample=event.sample,
            output_dir=output_dir,
            cfg_path=self.cfg_path,
            cfg_sha256=self.cfg_sha256,
            num_frames=event.num_frames,
            feature_shapes=event.feature_shapes,
            frame_periodicity_ms=float(self.config_params.get("frame_periodicity_ms", 50.0)),
            range_resolution_m=float(self.config_params.get("range_resolution_m", 0.0)),
            doppler_resolution_mps=float(self.config_params.get("doppler_resolution_mps", 0.0)),
            elapsed_seconds=time.time() - t0,
        )

    def extract_single_bin(
        self,
        bin_path: str,
        output_root: str,
        label: str = "debug",
        clip_id: Optional[str] = None,
        frame_start: int = 0,
        frame_end: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> ExtractionResult:
        event = self.extract_event(
            bin_path=bin_path, label=label, clip_id=clip_id,
            frame_start=frame_start, frame_end=frame_end, metadata=metadata,
        )
        return self.save_event(event, output_root=output_root, overwrite=overwrite)

    def extract_from_manifest(
        self,
        manifest_path: str,
        output_root: str,
        overwrite: bool = False,
        workers: int = 1,
    ) -> List[ExtractionResult]:
        samples = list(load_manifest(manifest_path))
        total = len(samples)
        t0 = time.time()

        if workers > 1:
            import concurrent.futures
            from collections import defaultdict
            cfg_path = self.cfg_path
            threshold = self.pointcloud_threshold
            angle_res = getattr(self, '_angle_res', None)

            # Group samples by bin_path so each worker reads a file only once
            groups: dict = defaultdict(list)
            for s in samples:
                groups[os.path.abspath(s.bin_path)].append(
                    dict(bin_path=s.bin_path, label=s.label, clip_id=s.clip_id,
                         frame_start=s.frame_start, frame_end=s.frame_end,
                         subject_id=s.subject_id, session_id=s.session_id,
                         scene=s.scene, metadata=s.metadata)
                )
            group_args = [
                (cfg_path, threshold, angle_res, output_root, overwrite, clip_list)
                for clip_list in groups.values()
            ]

            results = []
            ok = err = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit each clip as an individual task for real-time progress
                clip_futures = []
                for g_arg in group_args:
                    cfg_p, thr, ar, o_root, ow, clip_list = g_arg
                    for c in clip_list:
                        clip_futures.append(
                            executor.submit(
                                _worker_extract_single_clip,
                                (cfg_p, thr, ar, o_root, ow, c)
                            )
                        )
                for future in concurrent.futures.as_completed(clip_futures):
                    success, lbl, cid, elapsed, exc_msg = future.result()
                    ok += success
                    err += (1 - success)
                    idx = ok + err
                    elapsed_total = time.time() - t0
                    eta = elapsed_total / idx * max(total - idx, 0)
                    if success:
                        print(f"[{idx}/{total}] {lbl}/{cid} | {elapsed:.2f}s "
                              f"| elapsed={elapsed_total:.0f}s eta={eta:.0f}s", flush=True)
                    else:
                        print(f"[{idx}/{total}] ERROR {lbl}/{cid}: {exc_msg}", flush=True)
            print(f"[summary] ok={ok} errors={err} total={total}")
            return results

        # Single-process path: decode only the needed frame range per clip
        results: List[ExtractionResult] = []
        for idx, s in enumerate(samples, 1):
            bin_key = os.path.abspath(s.bin_path)
            decoded = decode_bin_frames_multi(
                bin_key, self.config_params,
                frame_start=s.frame_start, frame_end=s.frame_end
            )
            # decoded is now shape (frame_end-frame_start, ...) starting at index 0
            event = self._extract_event_from_decoded(
                decoded=decoded,
                bin_path=s.bin_path, label=s.label, clip_id=s.clip_id,
                frame_start=0, frame_end=len(decoded),
                subject_id=s.subject_id, session_id=s.session_id,
                scene=s.scene, metadata=s.metadata,
            )
            r = self.save_event(event, output_root, overwrite=overwrite)
            results.append(r)
            elapsed = time.time() - t0
            eta = elapsed / idx * (total - idx)
            print(
                f"[{idx}/{total}] {s.label}/{s.clip_id} "
                f"| {r.elapsed_seconds:.2f}s "
                f"| elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract PC + RD/RA/RE features from raw DCA1000 .bin files."
    )
    p.add_argument("--manifest", help="JSONL manifest of event samples.")
    p.add_argument("--bin", dest="bin_path", help="Single .bin file (debug mode).")
    p.add_argument("--cfg", default=resolve_default_cfg_path(), help="Radar.cfg path.")
    p.add_argument("--out", required=True, help="Output root directory.")
    p.add_argument("--label", default="debug", help="Label for single-bin mode.")
    p.add_argument("--clip-id", help="Clip id override (single-bin mode).")
    p.add_argument("--start", type=int, default=0, help="Frame start (single-bin).")
    p.add_argument("--end", type=int, help="Frame end exclusive (single-bin).")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Point cloud detection threshold ratio (default 0.3).")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output directory.")
    p.add_argument("--workers", type=int, default=1,
                   help="Number of parallel worker processes (default 1).")
    p.add_argument("--angle-res", type=float, default=None,
                   help="Override angle resolution in degrees (e.g. 1.0). Default: use Radar.cfg value.")
    return p


def _print_result(r: ExtractionResult) -> None:
    print(f"[done] {r.sample.label}/{r.sample.clip_id}")
    print(f"  output_dir    : {r.output_dir}")
    print(f"  frames        : {r.num_frames}")
    print(f"  feature_shapes: {r.feature_shapes}")
    print(f"  cfg_sha256    : {r.cfg_sha256}")
    print(f"  elapsed       : {r.elapsed_seconds:.2f}s")


def _worker_extract_single_clip(args_tuple):
    """Top-level picklable worker for a single clip. Decodes only the needed frame range."""
    cfg_path, threshold, angle_res, output_root, overwrite, c = args_tuple
    import DSP
    extractor = OfflineExtractor(cfg_path=cfg_path, pointcloud_threshold=threshold)
    if angle_res is not None:
        extractor._angle_res = angle_res
        import dsp.angle_estimation as Angle_dsp
        DSP.ANGLE_RES = angle_res
        DSP.ANGLE_BINS = int(DSP.ANGLE_RANGE * 2 // angle_res + 1)
        DSP.num_vec, DSP.steering_vec = Angle_dsp.gen_steering_vec(
            DSP.ANGLE_RANGE, DSP.ANGLE_RES, DSP.VIRT_ANT)
    s = ManifestSample(**c)
    try:
        decoded = decode_bin_frames_multi(
            os.path.abspath(s.bin_path), extractor.config_params,
            frame_start=s.frame_start, frame_end=s.frame_end
        )
        event = extractor._extract_event_from_decoded(
            decoded=decoded,
            bin_path=s.bin_path, label=s.label, clip_id=s.clip_id,
            frame_start=0, frame_end=len(decoded),
            subject_id=s.subject_id, session_id=s.session_id,
            scene=s.scene, metadata=s.metadata,
        )
        r = extractor.save_event(event, output_root, overwrite=overwrite)
        return (True, s.label, s.clip_id, r.elapsed_seconds, None)
    except Exception as exc:
        return (False, s.label, s.clip_id, 0.0, str(exc))


def _worker_extract_group(args_tuple):
    """Top-level picklable worker that processes all clips from one bin recording.

    Each clip is decoded independently using only its frame range (memmap),
    so memory usage is O(100 frames) per clip, not O(total_frames).
    """
    cfg_path, threshold, angle_res, output_root, overwrite, clip_list = args_tuple
    import DSP
    extractor = OfflineExtractor(cfg_path=cfg_path, pointcloud_threshold=threshold)
    if angle_res is not None:
        extractor._angle_res = angle_res
        import dsp.angle_estimation as Angle_dsp
        DSP.ANGLE_RES = angle_res
        DSP.ANGLE_BINS = int(DSP.ANGLE_RANGE * 2 // angle_res + 1)
        DSP.num_vec, DSP.steering_vec = Angle_dsp.gen_steering_vec(
            DSP.ANGLE_RANGE, DSP.ANGLE_RES, DSP.VIRT_ANT)
    results = []
    for c in clip_list:
        s = ManifestSample(**c)
        try:
            # Decode only the needed frames (memmap, low memory)
            decoded = decode_bin_frames_multi(
                os.path.abspath(s.bin_path), extractor.config_params,
                frame_start=s.frame_start, frame_end=s.frame_end
            )
            event = extractor._extract_event_from_decoded(
                decoded=decoded,
                bin_path=s.bin_path, label=s.label, clip_id=s.clip_id,
                frame_start=0, frame_end=len(decoded),
                subject_id=s.subject_id, session_id=s.session_id,
                scene=s.scene, metadata=s.metadata,
            )
            r = extractor.save_event(event, output_root, overwrite=overwrite)
            results.append((True, s.label, s.clip_id, r.elapsed_seconds, None))
        except Exception as exc:
            results.append((False, s.label, s.clip_id, 0.0, str(exc)))
    return results


def _worker_extract(args_tuple):
    """Top-level picklable worker for multiprocessing."""
    cfg_path, threshold, angle_res, output_root, overwrite, s_dict = args_tuple
    import DSP
    extractor = OfflineExtractor(cfg_path=cfg_path, pointcloud_threshold=threshold)
    if angle_res is not None:
        extractor.config_params["aoa_angle_res"] = angle_res
        DSP.apply_runtime_config(extractor.config_params)
        # Override ANGLE_RES and rebuild steering vector
        DSP.ANGLE_RES = angle_res
        DSP.ANGLE_BINS = int(DSP.ANGLE_RANGE * 2 // angle_res + 1)
        import dsp.angle_estimation as Angle_dsp
        DSP.num_vec, DSP.steering_vec = Angle_dsp.gen_steering_vec(
            DSP.ANGLE_RANGE, DSP.ANGLE_RES, DSP.VIRT_ANT)
    s = ManifestSample(**s_dict)
    try:
        event = extractor.extract_event(
            bin_path=s.bin_path, label=s.label, clip_id=s.clip_id,
            frame_start=s.frame_start, frame_end=s.frame_end,
            subject_id=s.subject_id, session_id=s.session_id,
            scene=s.scene, metadata=s.metadata,
        )
        r = extractor.save_event(event, output_root, overwrite=overwrite)
        return (True, s.label, s.clip_id, r.elapsed_seconds, None)
    except Exception as exc:
        return (False, s.label, s.clip_id, 0.0, str(exc))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    if bool(args.manifest) == bool(args.bin_path):
        _build_argument_parser().error("Use exactly one of --manifest or --bin.")

    extractor = OfflineExtractor(
        cfg_path=args.cfg, pointcloud_threshold=args.threshold
    )

    # Apply angle resolution override
    angle_res = getattr(args, 'angle_res', None)
    if angle_res is not None:
        import DSP
        import dsp.angle_estimation as Angle_dsp
        DSP.ANGLE_RES = angle_res
        DSP.ANGLE_BINS = int(DSP.ANGLE_RANGE * 2 // angle_res + 1)
        DSP.num_vec, DSP.steering_vec = Angle_dsp.gen_steering_vec(
            DSP.ANGLE_RANGE, DSP.ANGLE_RES, DSP.VIRT_ANT)
        print(f"[info] angle_res={angle_res} ANGLE_BINS={DSP.ANGLE_BINS}")

    if args.manifest:
        workers = getattr(args, 'workers', 1)
        angle_res = getattr(args, 'angle_res', None)
        if angle_res is not None:
            extractor._angle_res = angle_res
        if workers > 1:
            print(f"[info] parallel workers={workers}")
        results = extractor.extract_from_manifest(
            args.manifest, args.out, overwrite=args.overwrite, workers=workers
        )
        if workers == 1:
            print(f"[summary] extracted {len(results)} event(s)")
        return 0

    _print_result(extractor.extract_single_bin(
        bin_path=args.bin_path, output_root=args.out,
        label=args.label, clip_id=args.clip_id,
        frame_start=args.start, frame_end=args.end,
        overwrite=args.overwrite,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
