"""Package a trained model artifact for the realtime application.

Copies the model file to an output directory and emits ``model_meta.json``
describing the feature contract (RD/RA/RE/PC) and cfg binding.

Usage::

    python -m training.export_model \\
        --model F:/models/raca_v2_best.pth \\
        --out F:/deploy/fall_v1 \\
        --cfg config/Radar.cfg \\
        --sample-meta F:/Features/fall/clip001/meta.json \\
        --class-names fall non-fall \\
        --positive-labels fall
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Dict, List, Optional

from offline.feature_extractor import compute_file_sha256, resolve_default_cfg_path


SUPPORTED_SUFFIXES = {".pt", ".jit", ".ts", ".py", ".pth"}


def build_model_meta(
    model_path: str,
    cfg_path: str,
    sample_meta: Optional[Dict[str, Any]],
    class_names: List[str],
    positive_labels: List[str],
) -> Dict[str, Any]:
    cfg_sha256 = compute_file_sha256(cfg_path)
    sample_meta = sample_meta or {}
    contract = dict(sample_meta.get("model_contract", {}))
    return {
        "model_file": os.path.basename(model_path),
        "model_type": os.path.splitext(model_path)[1].lower().lstrip("."),
        "class_names": class_names or list(contract.get("class_names", [])),
        "positive_labels": positive_labels or list(contract.get("positive_labels", [])),
        "cfg_path": os.path.abspath(cfg_path),
        "cfg_sha256": cfg_sha256,
        "clip_frames": int(contract.get("clip_frames", sample_meta.get("num_frames", 0))),
        "frame_periodicity_ms": float(
            contract.get("frame_periodicity_ms",
                         sample_meta.get("frame_periodicity_ms", 0.0))
        ),
        # Feature contract: RD, RA, RE, PC only
        "feature_shapes": dict(
            contract.get("feature_shapes", sample_meta.get("feature_shapes", {}))
        ),
    }


def package_model(
    model_path: str,
    output_dir: str,
    cfg_path: str,
    sample_meta_path: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    positive_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    model_path = os.path.abspath(model_path)
    cfg_path = os.path.abspath(cfg_path)
    output_dir = os.path.abspath(output_dir)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    suffix = os.path.splitext(model_path)[1].lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported suffix {suffix}. Supported: {sorted(SUPPORTED_SUFFIXES)}")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"cfg not found: {cfg_path}")

    sample_meta: Optional[Dict[str, Any]] = None
    if sample_meta_path:
        with open(sample_meta_path, "r", encoding="utf-8") as fh:
            sample_meta = json.load(fh)

    os.makedirs(output_dir, exist_ok=True)
    target = os.path.join(output_dir, os.path.basename(model_path))
    shutil.copy2(model_path, target)

    meta = build_model_meta(
        model_path=target,
        cfg_path=cfg_path,
        sample_meta=sample_meta,
        class_names=list(class_names or []),
        positive_labels=list(positive_labels or []),
    )
    meta_path = os.path.join(output_dir, "model_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    return {"output_dir": output_dir, "model_path": target, "model_meta_path": meta_path, "model_meta": meta}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Package a trained model and emit model_meta.json."
    )
    p.add_argument("--model", required=True, help="Model artifact (.pth/.pt/.jit/.ts/.py).")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--cfg", default=resolve_default_cfg_path(), help="Radar.cfg path.")
    p.add_argument("--sample-meta",
                   help="meta.json from an extracted sample to copy feature contract from.")
    p.add_argument("--class-names", nargs="*", default=[],
                   help="Class names (overrides model contract).")
    p.add_argument("--positive-labels", nargs="*", default=[],
                   help="Positive labels (overrides model contract).")
    args = p.parse_args(argv)

    result = package_model(
        model_path=args.model,
        output_dir=args.out,
        cfg_path=args.cfg,
        sample_meta_path=args.sample_meta,
        class_names=args.class_names,
        positive_labels=args.positive_labels,
    )
    print(f"[model] packaged to      {result['model_path']}")
    print(f"[model] metadata written {result['model_meta_path']}")
    print(f"[model] positive_labels  {result['model_meta']['positive_labels']}")
    print(f"[model] feature_shapes   {result['model_meta']['feature_shapes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
