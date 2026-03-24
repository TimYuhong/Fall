"""Build a training manifest from extracted event feature directories.

Expected per-sample directory layout (produced by offline/feature_extractor.py)::

    features_root/<label>/<clip_id>/
        RD.npy
        RA.npy
        RE.npy
        PC.npy
        meta.json

Usage::

    python -m training.build_manifest \\
        --features-root F:/Features \\
        --out F:/manifests/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


REQUIRED_FILES = ("RD.npy", "RA.npy", "RE.npy", "meta.json")  # PC.npy is optional


@dataclass
class TrainingSample:
    sample_dir: str
    label: str
    clip_id: str
    meta_path: str
    rd_path: str
    ra_path: str
    re_path: str
    pc_path: str
    subject_id: str = ""
    session_id: str = ""
    scene: str = ""
    num_frames: int = 0
    cfg_sha256: str = ""
    feature_shapes: Dict[str, Any] = None  # type: ignore[assignment]
    class_names: List[str] = None          # type: ignore[assignment]
    positive_labels: List[str] = None      # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.feature_shapes is None:
            self.feature_shapes = {}
        if self.class_names is None:
            self.class_names = []
        if self.positive_labels is None:
            self.positive_labels = []

    def to_record(self) -> Dict[str, Any]:
        return {
            "sample_dir": self.sample_dir,
            "label": self.label,
            "clip_id": self.clip_id,
            "meta_path": self.meta_path,
            "rd_path": self.rd_path,
            "ra_path": self.ra_path,
            "re_path": self.re_path,
            "pc_path": self.pc_path,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "scene": self.scene,
            "num_frames": self.num_frames,
            "cfg_sha256": self.cfg_sha256,
            "feature_shapes": self.feature_shapes,
            "class_names": self.class_names,
            "positive_labels": self.positive_labels,
        }


def _iter_sample_dirs(features_root: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(features_root):
        if all(name in files for name in REQUIRED_FILES):
            yield os.path.abspath(root)


def scan_feature_root(features_root: str) -> List[TrainingSample]:
    """Scan extracted event directories and return manifest-ready samples."""
    samples: List[TrainingSample] = []
    for sample_dir in sorted(_iter_sample_dirs(features_root)):
        meta_path = os.path.join(sample_dir, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        contract = dict(meta.get("model_contract", {}))
        samples.append(TrainingSample(
            sample_dir=sample_dir,
            label=str(meta.get("label", "")),
            clip_id=str(meta.get("clip_id", os.path.basename(sample_dir))),
            meta_path=meta_path,
            rd_path=os.path.join(sample_dir, "RD.npy"),
            ra_path=os.path.join(sample_dir, "RA.npy"),
            re_path=os.path.join(sample_dir, "RE.npy"),
            pc_path=os.path.join(sample_dir, "PC.npy") if os.path.exists(os.path.join(sample_dir, "PC.npy")) else "",
            subject_id=str(meta.get("subject_id", "")),
            session_id=str(meta.get("session_id", "")),
            scene=str(meta.get("scene", "")),
            num_frames=int(meta.get("num_frames", 0)),
            cfg_sha256=str(meta.get("cfg_sha256", "")),
            feature_shapes=dict(meta.get("feature_shapes", {})),
            class_names=list(contract.get("class_names", [])),
            positive_labels=list(contract.get("positive_labels", [])),
        ))
    return samples


def write_manifest(samples: List[TrainingSample], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_record(), ensure_ascii=False) + "\n")


def build_summary(samples: List[TrainingSample]) -> Dict[str, Any]:
    labels: Dict[str, int] = {}
    cfg_hashes: set = set()
    for s in samples:
        labels[s.label] = labels.get(s.label, 0) + 1
        if s.cfg_sha256:
            cfg_hashes.add(s.cfg_sha256)
    return {
        "num_samples": len(samples),
        "labels": labels,
        "cfg_sha256_values": sorted(cfg_hashes),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a JSONL training manifest from extracted event feature dirs."
    )
    p.add_argument("--features-root", required=True,
                   help="Root directory containing extracted event samples.")
    p.add_argument("--out", required=True, help="Output JSONL manifest path.")
    args = p.parse_args(argv)

    samples = scan_feature_root(args.features_root)
    if not samples:
        raise SystemExit(f"No extracted samples found under: {args.features_root}")

    write_manifest(samples, args.out)
    summary = build_summary(samples)
    print(f"[manifest] wrote {summary['num_samples']} sample(s) to {args.out}")
    print(f"[manifest] labels: {summary['labels']}")
    print(f"[manifest] cfg_sha256: {summary['cfg_sha256_values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
