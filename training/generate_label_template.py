"""Auto-generate a JSONL manifest directly from .bin file names.

File naming convention expected::

    <label>_<SubjectID>_<scene>_<seq>_Raw_<idx>.bin
    e.g.  fall_S01_dormitory_02_Raw_0.bin
          walk_S03_corridor_01_Raw_0.bin

The label is derived from the leading token(s) before the first subject-ID
pattern (S\\d+ or P\\d+).  No manual review step is needed.

For weak-label recordings (e.g. 100 frames total, only the middle ~40 contain
the event), use --frame-start / --frame-end to slice the event window
automatically for every bin file in the scan.  Alternatively, per-file
overides can be applied with --frame-map (a JSON file).

Usage examples::

    # Derive label from filename, use all frames
    python -m training.generate_label_template \\
        --data-root F:/Data_bin \\
        --out manifests/labels.jsonl

    # Clip the middle 40 frames (30..70) from every 100-frame recording
    python -m training.generate_label_template \\
        --data-root F:/Data_bin \\
        --out manifests/labels.jsonl \\
        --frame-start 30 --frame-end 70

    # Per-file frame ranges via a JSON map
    python -m training.generate_label_template \\
        --data-root F:/Data_bin \\
        --out manifests/labels.jsonl \\
        --frame-map F:/frame_map.json

frame_map.json format::

    {
        "fall_S01_dormitory_02_Raw_0": {"frame_start": 30, "frame_end": 70},
        "walk_S01_corridor_01_Raw_0": {"frame_start": 0,  "frame_end": 100}
    }
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


IGNORED_TOP_LEVEL_DIRS = {"Features"}

# Matches subject/participant ID tokens like S01, P03
SUBJECT_PATTERN = re.compile(r"_((?:S|P)\d+)(?:_|$)", re.IGNORECASE)
# Matches trailing sequence + Raw index, e.g. _02_Raw_0 or _02
SEQUENCE_PATTERN = re.compile(r"_(\d+)(?:_Raw_\d+)?$", re.IGNORECASE)


def _normalize(value: str) -> str:
    """Lower-case, collapse spaces/hyphens to underscores."""
    value = value.strip().replace("-", "_")
    value = re.sub(r"\s+", "_", value)
    return re.sub(r"_+", "_", value).lower()


def _derive_subject_id(stem: str) -> str:
    m = SUBJECT_PATTERN.search(stem)
    return m.group(1).upper() if m else ""


def _derive_sequence_id(stem: str) -> str:
    m = SEQUENCE_PATTERN.search(stem)
    return m.group(1) if m else ""


def derive_label_from_stem(stem: str) -> str:
    """Extract the label prefix from a bin file stem.

    Strategy: take everything before the first subject-ID token (S\\d+/P\\d+).
    Falls back to the first underscore-separated token if no subject-ID found.

    Examples::

        fall_S01_dormitory_02_Raw_0   ->  "fall"
        walk_S03_corridor_01_Raw_0    ->  "walk"
        non_fall_S02_lab_03_Raw_0     ->  "non_fall"
        generic_event_01              ->  "generic"  (first token)
    """
    m = SUBJECT_PATTERN.search(stem)
    if m:
        prefix = stem[: m.start()]  # everything before _S01_
    else:
        # No subject token — fall back to first underscore-separated token
        prefix = stem.split("_")[0]
    return _normalize(prefix) or "unknown"


def _build_record(
    bin_path: str,
    data_root: str,
    frame_start: Optional[int],
    frame_end: Optional[int],
) -> Dict[str, Any]:
    rel = os.path.relpath(bin_path, data_root)
    parts = rel.split(os.sep)
    stem = os.path.splitext(os.path.basename(bin_path))[0]

    # Structural fields from path
    scene = parts[0] if len(parts) >= 3 else ""   # e.g. dormitory
    # parts layout: scene / label_dir / filename  (3-level)
    # or:           label_dir / filename           (2-level, no scene)

    subject_id = _derive_subject_id(stem)
    sequence_id = _derive_sequence_id(stem)
    session_id = f"{scene}_{sequence_id}" if scene and sequence_id else sequence_id

    # Label comes from the filename, not the directory
    label = derive_label_from_stem(stem)

    record: Dict[str, Any] = {
        "bin_path": os.path.abspath(bin_path),
        "label": label,
        "clip_id": stem,
        "subject_id": subject_id,
        "session_id": session_id,
        "scene": scene,
        "metadata": {
            "label_source": "filename",
            "source_relative_path": rel,
        },
    }
    if frame_start is not None:
        record["frame_start"] = frame_start
    if frame_end is not None:
        record["frame_end"] = frame_end
    return record


def scan_bins(
    data_root: str,
    default_frame_start: Optional[int] = None,
    default_frame_end: Optional[int] = None,
    frame_map: Optional[Dict[str, Dict[str, int]]] = None,
) -> List[Dict[str, Any]]:
    """Scan data_root for .bin files and build manifest records.

    Parameters
    ----------
    data_root:
        Root directory to scan recursively.
    default_frame_start / default_frame_end:
        Applied to every file unless overridden in frame_map.
    frame_map:
        Per-stem overrides: ``{stem: {"frame_start": int, "frame_end": int}}``.
        Stems are matched against the file base name without extension.
    """
    data_root = os.path.abspath(data_root)
    frame_map = frame_map or {}
    records: List[Dict[str, Any]] = []

    for root, dirs, files in os.walk(data_root):
        dirs[:] = [d for d in dirs if d not in IGNORED_TOP_LEVEL_DIRS]
        for filename in sorted(files):
            if not filename.lower().endswith(".bin"):
                continue
            bin_path = os.path.join(root, filename)
            stem = os.path.splitext(filename)[0]

            # Per-file overrides take priority over global defaults
            override = frame_map.get(stem, {})
            fs = override.get("frame_start", default_frame_start)
            fe = override.get("frame_end", default_frame_end)

            records.append(_build_record(bin_path, data_root, fs, fe))

    return sorted(records, key=lambda r: r["bin_path"].lower())


def write_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_frame_map(path: str) -> Dict[str, Dict[str, int]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"frame_map must be a JSON object, got: {type(data)}")
    return data  # type: ignore[return-value]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Auto-generate a JSONL manifest from .bin filenames. "
            "Labels are parsed directly from the file stem — no manual review needed."
        )
    )
    p.add_argument("--data-root", required=True,
                   help="Root directory containing raw .bin files.")
    p.add_argument("--out", required=True, help="Output JSONL manifest path.")
    p.add_argument(
        "--frame-start", type=int, default=None,
        help="Global frame_start applied to every bin (0-based, inclusive). "
             "Use to skip lead-in frames before the event.",
    )
    p.add_argument(
        "--frame-end", type=int, default=None,
        help="Global frame_end applied to every bin (exclusive). "
             "E.g. --frame-start 30 --frame-end 70 extracts frames [30, 70).",
    )
    p.add_argument(
        "--frame-map", default=None,
        help="Path to a JSON file with per-stem frame ranges "
             '({"stem": {"frame_start": N, "frame_end": M}}). '
             "Overrides --frame-start / --frame-end for matched stems.",
    )
    args = p.parse_args(argv)

    frame_map: Dict[str, Dict[str, int]] = {}
    if args.frame_map:
        frame_map = _load_frame_map(args.frame_map)
        print(f"[template] loaded frame_map with {len(frame_map)} stem override(s)")

    records = scan_bins(
        data_root=args.data_root,
        default_frame_start=args.frame_start,
        default_frame_end=args.frame_end,
        frame_map=frame_map,
    )
    if not records:
        raise SystemExit(f"No .bin files found under: {args.data_root}")

    write_jsonl(records, args.out)

    # Print summary
    label_counts: Dict[str, int] = {}
    for r in records:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    print(f"[template] wrote {len(records)} record(s) to {args.out}")
    print(f"[template] labels found: {label_counts}")
    if args.frame_start is not None or args.frame_end is not None:
        print(f"[template] global frame window: [{args.frame_start}, {args.frame_end})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
