"""Quick visualisation of extracted radar features (RD / RA / RE / PC).

Usage::

    python tools/visualize_features.py \\
        --sample-dir F:/Features/extracted_test/fall/fall_S01_dormitory_02_Raw_0_f00000_00100

    # animate over all frames
    python tools/visualize_features.py \\
        --sample-dir F:/Features/extracted_test/fall/fall_S01_dormitory_02_Raw_0_f00000_00100 \\
        --animate

    # show a single specific frame
    python tools/visualize_features.py \\
        --sample-dir F:/Features/extracted_test/fall/fall_S01_dormitory_02_Raw_0_f00000_00100 \\
        --frame 30
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def load_sample(sample_dir: str):
    RD = np.load(os.path.join(sample_dir, "RD.npy"))  # (T, doppler, range)
    RA = np.load(os.path.join(sample_dir, "RA.npy"))  # (T, angle, range)
    RE = np.load(os.path.join(sample_dir, "RE.npy"))  # (T, angle, range)
    
    # PC.npy is optional
    pc_path = os.path.join(sample_dir, "PC.npy")
    if os.path.exists(pc_path):
        PC = np.load(pc_path)  # (N, 4)
    else:
        PC = np.zeros((0, 4))
        
    with open(os.path.join(sample_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return RD, RA, RE, PC, meta


def _log_mag(arr: np.ndarray) -> np.ndarray:
    """Log-magnitude for better visualisation contrast."""
    # arr may be real (magnitude) or complex — handle both
    if np.iscomplexobj(arr):
        return 20 * np.log10(np.abs(arr) + 1e-6)
    else:
        return 20 * np.log10(np.abs(arr) + 1.0)  # +1 avoids log(0) for real magnitudes


def apply_cfar_2d(img_db: np.ndarray, guard=6, noise=6, l_bound=2.5) -> np.ndarray:
    """
    Applies 2D Cell-Averaging CFAR on a log-magnitude image by performing
    1D CA-CFAR along both axes sequentially (as standard in Radar DSP pipelines)
    using the project's native dsp.cfar module.
    """
    import sys
    # Ensure we can import the project's native dsp module
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    try:
        from dsp import cfar
    except ImportError:
        print("[!] Could not import dsp.cfar. Ensure you are running from the project root.")
        return np.ones_like(img_db, dtype=bool)

    # 1D CFAR along the first axis (e.g. Range, or Angle)
    mask_ax0 = cfar.ca(img_db, guard_len=guard, noise_len=noise, l_bound=l_bound, mode='wrap')
    # 1D CFAR along the second axis (e.g. Doppler, or Range), transposed to process rows
    mask_ax1 = cfar.ca(img_db.T, guard_len=guard, noise_len=noise, l_bound=l_bound, mode='wrap').T
    
    # Target is confirmed if it passes the threshold test in BOTH dimensions
    return mask_ax0 & mask_ax1


def plot_frame(
    RD: np.ndarray,
    RA: np.ndarray,
    RE: np.ndarray,
    PC: np.ndarray,
    frame_idx: int,
    meta: dict,
    save_path: Optional[str] = None,
    cmap: str = "jet",
    use_cfar: bool = False,
    cfar_thresh: float = 12.0
) -> None:
    label = meta.get("label", "?")
    clip_id = meta.get("clip_id", "?")
    T = RD.shape[0]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"Label: {label}  |  Clip: {clip_id}  |  Frame {frame_idx+1}/{T}",
        fontsize=13, fontweight="bold"
    )

    # --- RD map ---
    ax1 = fig.add_subplot(2, 3, 1)
    rd_log = _log_mag(RD[frame_idx])
    if use_cfar:
        mask = apply_cfar_2d(rd_log, l_bound=cfar_thresh)
        rd_log[~mask] = rd_log.min()
    im1 = ax1.imshow(
        rd_log, aspect="auto", origin="lower",
        cmap=cmap,
        extent=[0, RD.shape[2], -RD.shape[1]//2, RD.shape[1]//2]
    )
    ax1.set_title("Range-Doppler (RD)")
    ax1.set_xlabel("Range bin")
    ax1.set_ylabel("Doppler bin")
    plt.colorbar(im1, ax=ax1, label="dB")

    # --- RA map ---
    ax2 = fig.add_subplot(2, 3, 2)
    ra_log = _log_mag(RA[frame_idx])
    if use_cfar:
        mask = apply_cfar_2d(ra_log, l_bound=cfar_thresh)
        ra_log[~mask] = ra_log.min()
    im2 = ax2.imshow(
        ra_log, aspect="auto", origin="lower",
        cmap=cmap,
        extent=[0, RA.shape[2], -90, 90]
    )
    ax2.set_title("Range-Azimuth (RA)")
    ax2.set_xlabel("Range bin")
    ax2.set_ylabel("Azimuth (deg)")
    plt.colorbar(im2, ax=ax2, label="dB")

    # --- RE map ---
    ax3 = fig.add_subplot(2, 3, 3)
    re_log = _log_mag(RE[frame_idx])
    if use_cfar:
        mask = apply_cfar_2d(re_log, l_bound=cfar_thresh)
        re_log[~mask] = re_log.min()
    im3 = ax3.imshow(
        re_log, aspect="auto", origin="lower",
        cmap=cmap,
        extent=[0, RE.shape[2], -90, 90]
    )
    ax3.set_title("Range-Elevation (RE)")
    ax3.set_xlabel("Range bin")
    ax3.set_ylabel("Elevation (deg)")
    plt.colorbar(im3, ax=ax3, label="dB")

    # --- Point Cloud 3D (all frames) ---
    ax4 = fig.add_subplot(2, 3, 4, projection="3d")
    if PC.shape[0] > 0:
        ax4.scatter(
            PC[:, 1], PC[:, 2], PC[:, 3],
            c=PC[:, 0], cmap="plasma", s=1, alpha=0.4
        )
    ax4.set_title(f"Point Cloud (all frames, N={PC.shape[0]})")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_zlabel("Z (m)")

    # --- RD over time (mean per doppler bin) ---
    ax5 = fig.add_subplot(2, 3, 5)
    rd_mean = _log_mag(RD).mean(axis=2)  # (T, doppler)
    im5 = ax5.imshow(
        rd_mean.T, aspect="auto", origin="lower",
        cmap=cmap,
        extent=[0, T, -RD.shape[1]//2, RD.shape[1]//2]
    )
    ax5.axvline(frame_idx, color="red", linewidth=1.5, label="current")
    ax5.set_title("RD over time (mean across range)")
    ax5.set_xlabel("Frame")
    ax5.set_ylabel("Doppler bin")
    ax5.legend(fontsize=8)
    plt.colorbar(im5, ax=ax5, label="dB")

    # --- RA over time (mean per angle bin) ---
    ax6 = fig.add_subplot(2, 3, 6)
    ra_mean = _log_mag(RA).mean(axis=2)  # (T, angle)
    im6 = ax6.imshow(
        ra_mean.T, aspect="auto", origin="lower",
        cmap=cmap,
        extent=[0, T, -90, 90]
    )
    ax6.axvline(frame_idx, color="red", linewidth=1.5, label="current")
    ax6.set_title("RA over time (mean across range)")
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Azimuth (deg)")
    ax6.legend(fontsize=8)
    plt.colorbar(im6, ax=ax6, label="dB")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[saved] {save_path}")
    else:
        plt.show()
    plt.close(fig)


def animate_frames(
    RD: np.ndarray,
    RA: np.ndarray,
    RE: np.ndarray,
    PC: np.ndarray,
    meta: dict,
    interval_ms: int = 100,
    save_gif: Optional[str] = None,
    cmap: str = "jet",
    use_cfar: bool = False,
    cfar_thresh: float = 12.0
) -> None:
    T = RD.shape[0]
    label = meta.get("label", "?")
    clip_id = meta.get("clip_id", "?")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Label: {label}  |  {clip_id}", fontsize=12, fontweight="bold")

    rd_log_all = _log_mag(RD)
    ra_log_all = _log_mag(RA)
    re_log_all = _log_mag(RE)

    if use_cfar:
        print("[*] Applying 2D CFAR across all frames... this might take a moment.")
        for i in range(T):
            mask_rd = apply_cfar_2d(rd_log_all[i], l_bound=cfar_thresh)
            rd_log_all[i][~mask_rd] = rd_log_all[i].min()
            
            mask_ra = apply_cfar_2d(ra_log_all[i], l_bound=cfar_thresh)
            ra_log_all[i][~mask_ra] = ra_log_all[i].min()
            
            mask_re = apply_cfar_2d(re_log_all[i], l_bound=cfar_thresh)
            re_log_all[i][~mask_re] = re_log_all[i].min()

    vmin_rd, vmax_rd = rd_log_all.min(), rd_log_all.max()
    vmin_ra, vmax_ra = ra_log_all.min(), ra_log_all.max()

    im_rd = axes[0].imshow(
        rd_log_all[0], aspect="auto", origin="lower",
        cmap=cmap, vmin=vmin_rd, vmax=vmax_rd,
        extent=[0, RD.shape[2], -RD.shape[1]//2, RD.shape[1]//2]
    )
    axes[0].set_title("RD"); axes[0].set_xlabel("Range"); axes[0].set_ylabel("Doppler")

    im_ra = axes[1].imshow(
        ra_log_all[0], aspect="auto", origin="lower",
        cmap=cmap, vmin=vmin_ra, vmax=vmax_ra,
        extent=[0, RA.shape[2], -90, 90]
    )
    axes[1].set_title("RA"); axes[1].set_xlabel("Range"); axes[1].set_ylabel("Azimuth")

    im_re = axes[2].imshow(
        re_log_all[0], aspect="auto", origin="lower",
        cmap=cmap, vmin=vmin_ra, vmax=vmax_ra,
        extent=[0, RE.shape[2], -90, 90]
    )
    axes[2].set_title("RE"); axes[2].set_xlabel("Range"); axes[2].set_ylabel("Elevation")

    title = fig.text(0.5, 0.92, f"Frame 0/{T}", ha="center", fontsize=10)

    def update(i):
        im_rd.set_data(rd_log_all[i])
        im_ra.set_data(ra_log_all[i])
        im_re.set_data(re_log_all[i])
        title.set_text(f"Frame {i+1}/{T}")
        return im_rd, im_ra, im_re, title

    ani = animation.FuncAnimation(
        fig, update, frames=T, interval=interval_ms, blit=True
    )

    if save_gif:
        ani.save(save_gif, writer="pillow", fps=1000//interval_ms)
        print(f"[saved] {save_gif}")
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visualise extracted radar features.")
    p.add_argument("--sample-dir", required=True,
                   help="Path to extracted sample dir (contains RD/RA/RE/PC.npy + meta.json).")
    p.add_argument("--frame", type=int, default=None,
                   help="Frame index to display (default: middle frame).")
    p.add_argument("--animate", action="store_true",
                   help="Animate all frames.")
    p.add_argument("--save", default=None,
                   help="Save output to file (.png for single frame, .gif for animation).")
    p.add_argument("--interval", type=int, default=100,
                   help="Animation interval in ms (default: 100).")
    p.add_argument("--cmap", default="jet",
                   help="Colormap to use (default: jet). Try 'magma', 'plasma', 'viridis', etc.")
    p.add_argument("--cfar", action="store_true",
                   help="Apply 2D CA-CFAR detection to the heatmaps to filter out noise.")
    p.add_argument("--cfar-thresh", type=float, default=2.5,
                   help="CFAR threshold in dB (default: 2.5). Higher = less points, lower = more points.")
    args = p.parse_args()

    RD, RA, RE, PC, meta = load_sample(args.sample_dir)
    print(f"RD : {RD.shape}  RA : {RA.shape}  RE : {RE.shape}  PC : {PC.shape}")
    print(f"label: {meta.get('label')}  clip_id: {meta.get('clip_id')}")

    if args.animate:
        animate_frames(RD, RA, RE, PC, meta,
                       interval_ms=args.interval,
                       save_gif=args.save,
                       cmap=args.cmap,
                       use_cfar=args.cfar,
                       cfar_thresh=args.cfar_thresh)
    else:
        frame_idx = args.frame if args.frame is not None else RD.shape[0] // 2
        plot_frame(RD, RA, RE, PC, frame_idx, meta, save_path=args.save, cmap=args.cmap,
                   use_cfar=args.cfar, cfar_thresh=args.cfar_thresh)


if __name__ == "__main__":
    main()
