"""Offline batch inference script for RACA v2 fall detection model.

Runs the trained model on one or more extracted sample directories and prints
prediction results without requiring the GUI to be running.

Usage::

    # Single sample
    python tools/offline_infer.py \\
        --checkpoint checkpoints/lstm_20260320_133236/raca_v2_best.pth \\
        --sample-dir dataset/extracted/fall/fall_S01_parlor_21_Raw_0

    # Whole class folder (all clips)
    python tools/offline_infer.py \\
        --checkpoint checkpoints/lstm_20260320_133236/raca_v2_best.pth \\
        --sample-dir dataset/extracted/fall

    # Everything
    python tools/offline_infer.py \\
        --checkpoint checkpoints/lstm_20260320_133236/raca_v2_best.pth \\
        --sample-dir dataset/extracted
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_sample_dir(path: str) -> bool:
    """True if `path` contains RD.npy, RA.npy, RE.npy."""
    return all(os.path.exists(os.path.join(path, f)) for f in ("RD.npy", "RA.npy", "RE.npy"))


def _collect_sample_dirs(root: str) -> list[str]:
    """Recursively collect all leaf directories that are valid sample dirs."""
    if _is_sample_dir(root):
        return [root]
    hits = []
    for entry in sorted(os.scandir(root), key=lambda e: e.name):
        if entry.is_dir():
            hits.extend(_collect_sample_dirs(entry.path))
    return hits


def _load_sample(sample_dir: str):
    RD = np.load(os.path.join(sample_dir, "RD.npy"))  # (T, range, doppler)
    RA = np.load(os.path.join(sample_dir, "RA.npy"))  # (T, angle, range)
    RE = np.load(os.path.join(sample_dir, "RE.npy"))  # (T, angle, range)
    import json
    with open(os.path.join(sample_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    true_label = meta.get("label", "?")
    clip_id = meta.get("clip_id", os.path.basename(sample_dir))
    return RD, RA, RE, true_label, clip_id


def _sorted_failed_records(failed_records: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        failed_records,
        key=lambda item: (
            str(item["true"]).lower(),
            str(item["pred"]).lower(),
            -float(item["prob"]),
            str(item["path"]),
        ),
    )


def _export_failed_records_csv(failed_records: list[dict[str, object]], csv_path: str) -> None:
    rows = _sorted_failed_records(failed_records)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["true", "pred", "prob", "path"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "true": row["true"],
                    "pred": row["pred"],
                    "prob": f"{float(row['prob']):.6f}",
                    "path": row["path"],
                }
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Offline batch inference for RACA v2 fall detection.")
    p.add_argument("--checkpoint", required=True,
                   help="Path to raca_v2_best.pth")
    p.add_argument("--sample-dir", required=True,
                   help="Path to a single sample dir, a class folder, or the full extracted root.")
    import torch
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", default=default_device, choices=["cpu", "cuda"],
                   help=f"Inference device (default: {default_device})")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Fall probability threshold (default: 0.5)")
    p.add_argument("--errors-only", action="store_true",
                   help="Only print samples that were predicted incorrectly.")
    p.add_argument("--export-errors", type=str, default="",
                   help="Optional file path to export the list of failed sample directories (e.g. errors.txt).")
    p.add_argument("--export-errors-csv", type=str, default="",
                   help="Optional CSV path for structured failed sample export.")
    p.add_argument("--visualize-errors", action="store_true",
                   help="Render plots for failed samples after inference.")
    p.add_argument("--visualize-out-dir", type=str, default="bad_case_plots",
                   help="Directory used when --visualize-errors is enabled.")
    p.add_argument("--visualize-limit", type=int, default=0,
                   help="Maximum number of failed samples to visualize (0 means all).")
    p.add_argument("--analyze-errors", action="store_true",
                   help="Print a grouped bad-case summary and save a Markdown report.")
    p.add_argument("--analysis-report", type=str, default=os.path.join("bad_case_plots", "bad_case_report.md"),
                   help="Markdown report path used when --analyze-errors is enabled.")
    args = p.parse_args()

    # --- Load model via raca_predictor factory ---
    from runtime.raca_predictor import load_raca_predictor
    result = load_raca_predictor(
        checkpoint=args.checkpoint,
        device=args.device,
        threshold=args.threshold,
        stride=100,   # for offline: infer once per clip (no overlap needed)
        window=100,
    )
    if not result.success:
        print(f"[ERROR] Failed to load model: {result.message}")
        sys.exit(1)

    predictor = result.predictor
    binary_mode = getattr(predictor, "_binary_mode", False)
    print(f"[OK] Model loaded: {result.message}")
    print(f"    binary_mode = {binary_mode},  threshold = {args.threshold},  device = {args.device}\n")

    # --- Collect sample dirs ---
    sample_dirs = _collect_sample_dirs(args.sample_dir)
    if not sample_dirs:
        print(f"[ERROR] No valid sample directories found under: {args.sample_dir}")
        sys.exit(1)

    print(f"Found {len(sample_dirs)} sample(s)\n")
    print(f"{'Clip ID':<45} {'True':<10} {'Pred':<10} {'Fall%':>6}  Result")
    print("-" * 88)

    # --- Run inference per clip ---
    correct = 0
    total = 0
    failed_paths = []
    failed_records = []
    plot_map = {}

    from runtime.fall_predictor import FallFeatureClip

    for sample_dir in sample_dirs:
        try:
            RD, RA, RE, true_label, clip_id = _load_sample(sample_dir)
        except Exception as e:
            print(f"  [skip] {sample_dir}: {e}")
            continue

        T = RD.shape[0]

        # Feed all frames into the predictor's sliding buffer
        predictor.reset()
        prediction = None
        for t in range(T):
            clip = FallFeatureClip(
                timestamp_start_ms=t * 50.0,
                timestamp_end_ms=(t + 1) * 50.0,
                RDT=RD[t],   # (range, doppler)
                ART=RA[t],   # (angle, range)
                ERT=RE[t],   # (angle, range)
            )
            prediction = predictor.predict(clip)

        if prediction is None or not prediction.available:
            print(f"  [skip] {clip_id}: prediction unavailable (not enough frames?)")
            continue

        pred_label = prediction.label    # 'fall' or 'non-fall'
        fall_prob  = prediction.probability

        # Ground truth: only 'fall' and 'falls' count as positive
        is_true_fall = true_label.lower().strip() in ("fall", "falls")
        is_pred_fall = pred_label.lower() == "fall"

        correct += int(is_true_fall == is_pred_fall)
        total   += 1

        is_correct = (is_true_fall == is_pred_fall)
        status = "[OK]  CORRECT" if is_correct else "[FAIL] WRONG"

        if not is_correct:
            abs_sample_dir = os.path.abspath(sample_dir)
            failed_paths.append(abs_sample_dir)
            failed_records.append(
                {
                    "true": true_label,
                    "pred": pred_label,
                    "prob": float(fall_prob),
                    "path": abs_sample_dir,
                }
            )

        if args.errors_only and is_correct:
            continue

        short_clip = clip_id[-44:] if len(clip_id) > 44 else clip_id
        print(f"  {short_clip:<45} {true_label:<10} {pred_label:<10} {fall_prob:>6.1%}  {status}")

    print("-" * 88)
    if total > 0:
        accuracy = correct / total * 100
        print(f"\n[Summary]  {correct}/{total} correct  -  {accuracy:.1f}% accuracy on {total} clips")
    else:
        print("[Summary]  No clips were scored.")

    if args.export_errors and failed_paths:
        try:
            with open(args.export_errors, "w", encoding="utf-8") as fe:
                for fp in failed_paths:
                    fe.write(f"{fp}\n")
            print(f"\n[Export] Saved {len(failed_paths)} failed sample paths to: {args.export_errors}")
        except Exception as e:
            print(f"\n[Export Error] Failed to write to {args.export_errors}: {e}")

    if args.export_errors_csv and failed_records:
        try:
            _export_failed_records_csv(failed_records, args.export_errors_csv)
            print(f"[Export] Saved {len(failed_records)} failed sample rows to CSV: {args.export_errors_csv}")
        except Exception as e:
            print(f"[Export Error] Failed to write CSV {args.export_errors_csv}: {e}")

    if args.visualize_errors:
        if not failed_paths:
            print("\n[Visualize] No failed samples to visualize.")
        else:
            try:
                from tools.visualize_bad_cases import visualize_samples

                plot_map = visualize_samples(
                    failed_paths,
                    out_dir=args.visualize_out_dir,
                    limit=args.visualize_limit,
                )
                print(f"[Visualize] Generated {len(plot_map)} plot(s) in: {args.visualize_out_dir}")
            except Exception as e:
                print(f"\n[Visualize Error] Failed to visualize bad cases: {e}")

    if args.analyze_errors:
        if not failed_records:
            print("\n[Analysis] No failed samples to analyze.")
        else:
            try:
                from tools.visualize_bad_cases import analyze_bad_cases

                analyze_bad_cases(
                    failed_records,
                    plot_map=plot_map,
                    report_path=args.analysis_report,
                )
            except Exception as e:
                print(f"\n[Analysis Error] Failed to analyze bad cases: {e}")


if __name__ == "__main__":
    main()
