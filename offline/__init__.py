"""Offline data extraction utilities.

Main entry point: OfflineExtractor in feature_extractor.py

Features extracted per event clip:
    RD.npy  - Range-Doppler map  (num_frames, range_bins, doppler_bins)   -- matches DSP.rdi_queue convention
    RA.npy  - Range-Azimuth map  (num_frames, angle_bins, range_bins)     -- raw Capon magnitude
    RE.npy  - Range-Elevation map(num_frames, angle_bins, range_bins)     -- raw Capon magnitude
    PC.npy  - Point cloud        (total_points, 4) [range, x, y, z]
    meta.json
"""
