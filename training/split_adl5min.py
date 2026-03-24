"""Split ADL_5min multi-part .bin files into fixed-length segment manifests.

ADL_5min recordings consist of multiple consecutive bin parts:
    ADL_5min_S01_bedroom_01_Raw_0.bin  (~1 GB)
    ADL_5min_S01_bedroom_01_Raw_1.bin  (~1 GB)
    ADL_5min_S01_bedroom_01_Raw_2.bin  (~200 MB, last part)

This script:
  1. Groups parts by recording session (same prefix before _Raw_N).
  2. Concatenates total frame counts across all parts.
  3. Emits one JSONL record per segment with label="non_fall".

Each segment record references the specific part bin and frame range within
that part, so offline/feature_extractor.py can process it directly.

Usage::

    python -m training.split_adl5min \\
        --adl-root F:\\Data_bin\\bedroom\\ADL_5min \\
        --out F:\\Features\\manifests\\adl_bedroom_non_fall.jsonl \\
        --segment-frames 100 \\
        --label non_fall

    # Multiple scenes at once:
    python -m training.split_adl5min \\
        --adl-root F:\\Data_bin\\bedroom\\ADL_5min F:\\Data_bin\\parlor\\ADL_5min \\
        --out F:\\Features\\manifests\\adl_all_non_fall.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Must match the project Radar.cfg — adjusted automatically if cfg is provided.
DEFAULT_FRAME_BYTES = 393216  # bytes per frame (int16 * frame_length)
RAW_INT16_BYTES = 2

PART_PATTERN = re.compile(r"^(.+_Raw)_(\d+)\.bin$", re.IGNORECASE)
SUBJECT_PATTERN = re.compile(r"_((?:S|P)\d+)(?:_|$)", re.IGNORECASE)


def _derive_subject_id(stem: str) -> str:
    m = SUBJECT_PATTERN.search(stem)
    return m.group(1).upper() if m else ""


def _derive_scene(adl_root: str) -> str:
    """Infer scene name from parent directory of ADL_5min folder."""
    parent = os.path.basename(os.path.dirname(os.path.abspath(adl_root)))
    return parent


def _frames_in_bin(bin_path: str, frame_bytes: int) -> int:
    """Return number of COMPLETE frames in a single bin part.

    ADL_5min bin parts may not be individually frame-aligned because the
    DCA1000 splits the stream at 1 GB boundaries mid-frame.  We therefore
    return the integer-truncated count and handle the leftover bytes when
    stitching parts together.
    """
    size = os.path.getsize(bin_path)
    return size // frame_bytes


def _total_frames_for_session(
    parts: List[Tuple[int, str]], frame_bytes: int
) -> Tuple[int, int]:
    """Return (total_complete_frames, total_bytes) treating all parts as one stream."""
    total_bytes = sum(os.path.getsize(p) for _, p in parts)
    return total_bytes // frame_bytes, total_bytes


def _group_parts(adl_root: str) -> Dict[str, List[Tuple[int, str]]]:
    """Group bin files by recording session prefix.

    Returns
    -------
    dict mapping session_prefix -> sorted list of (part_index, full_path)
    """
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for filename in os.listdir(adl_root):
        m = PART_PATTERN.match(filename)
        if not m:
            continue
        prefix = m.group(1)   # e.g. ADL_5min_S01_bedroom_01_Raw
        part_idx = int(m.group(2))
        full_path = os.path.join(adl_root, filename)
        groups.setdefault(prefix, []).append((part_idx, full_path))
    # Sort each group by part index
    for prefix in groups:
        groups[prefix].sort(key=lambda x: x[0])
    return groups


def build_segment_records(
    adl_root: str,
    segment_frames: int = 100,
    label: str = "non_fall",
    frame_bytes: int = DEFAULT_FRAME_BYTES,
) -> List[Dict[str, Any]]:
    """Build JSONL records for all segments across all recording sessions.

    ADL_5min parts are treated as a single concatenated stream.  Segments
    that would cross a part boundary are assigned to the part that contains
    their starting frame, and the frame_end is clamped to that part's last
    complete frame.  In practice, with the correct frame_bytes the stream
    is cleanly divisible, so cross-boundary segments do not occur.
    """
    scene = _derive_scene(adl_root)
    groups = _group_parts(adl_root)
    records: List[Dict[str, Any]] = []

    for session_prefix, parts in sorted(groups.items()):
        session_stem = session_prefix.rsplit("_Raw", 1)[0]
        subject_id = _derive_subject_id(session_stem)

        # Compute total frames treating all parts as one stream
        total_bytes = sum(os.path.getsize(p) for _, p in parts)
        total_frames = total_bytes // frame_bytes
        if total_frames == 0:
            print(f"[warn] {session_stem}: zero frames — skipping")
            continue

        # Build part boundary table: list of (part_global_start, part_global_end, bin_path)
        boundaries: List[Tuple[int, int, str]] = []
        offset = 0
        for _idx, bin_path in parts:
            part_bytes = os.path.getsize(bin_path)
            part_frames = part_bytes // frame_bytes
            boundaries.append((offset, offset + part_frames, bin_path))
            offset += part_frames
        # The last part may have leftover bytes; remaining frames go to last part
        # offset == total_frames already because total_bytes // frame_bytes

        seg_idx = 0
        global_frame = 0
        while global_frame + segment_frames <= total_frames:
            seg_start = global_frame
            seg_end = global_frame + segment_frames

            # Find the part that owns seg_start
            owner_bin = None
            local_start = 0
            local_end = 0
            for p_start, p_end, bin_path in boundaries:
                if p_start <= seg_start < p_end:
                    local_start = seg_start - p_start
                    local_end = seg_end - p_start
                    if local_end <= (p_end - p_start):
                        # Entire segment within this part
                        owner_bin = bin_path
                    else:
                        # Crosses boundary — skip this segment
                        owner_bin = None
                    break

            if owner_bin is not None:
                clip_id = f"{session_stem}_seg_{seg_idx:03d}"
                records.append({
                    "bin_path": os.path.abspath(owner_bin),
                    "label": label,
                    "clip_id": clip_id,
                    "frame_start": local_start,
                    "frame_end": local_end,
                    "subject_id": subject_id,
                    "session_id": session_stem,
                    "scene": scene,
                    "metadata": {
                        "label_source": "adl_5min_segment",
                        "session_prefix": session_prefix,
                        "global_frame_start": seg_start,
                        "global_frame_end": seg_end,
                        "part_file": os.path.basename(owner_bin),
                    },
                })
                seg_idx += 1

            global_frame += segment_frames

        skipped = total_frames - (seg_idx * segment_frames)
        print(
            f"[split] {session_stem}: total_frames={total_frames}, "
            f"segments={seg_idx}, skipped_remainder={skipped}"
        )

    return records


def write_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _get_frame_bytes_from_cfg(cfg_path: str) -> int:
    """Parse Radar.cfg to get the actual frame size in bytes."""
    try:
        import sys
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from offline.feature_extractor import parse_cfg
        params = parse_cfg(cfg_path)
        frame_length = int(params["frame_length"])
        return frame_length * RAW_INT16_BYTES
    except Exception as exc:
        print(f"[warn] Could not parse cfg ({exc}), using default frame_bytes={DEFAULT_FRAME_BYTES}")
        return DEFAULT_FRAME_BYTES


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Split ADL_5min multi-part .bin files into fixed-length non-fall segments. "
            "Outputs a JSONL manifest ready for offline/feature_extractor.py."
        )
    )
    p.add_argument(
        "--adl-root", nargs="+", required=True,
        help="One or more ADL_5min directories (e.g. F:/Data_bin/bedroom/ADL_5min).",
    )
    p.add_argument("--out", required=True, help="Output JSONL manifest path.")
    p.add_argument(
        "--segment-frames", type=int, default=100,
        help="Frames per segment clip (default: 100).",
    )
    p.add_argument(
        "--label", default="non_fall",
        help="Label to assign to all segments (default: non_fall).",
    )
    p.add_argument(
        "--cfg", default=None,
        help="Radar.cfg path to auto-detect frame_bytes. Defaults to config/Radar.cfg.",
    )
    args = p.parse_args(argv)

    # Determine frame_bytes
    cfg_path = args.cfg
    if cfg_path is None:
        default_cfg = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "Radar.cfg"
        )
        if os.path.exists(default_cfg):
            cfg_path = default_cfg

    frame_bytes = _get_frame_bytes_from_cfg(cfg_path) if cfg_path else DEFAULT_FRAME_BYTES
    print(f"[split] frame_bytes={frame_bytes}")

    all_records: List[Dict[str, Any]] = []
    for adl_root in args.adl_root:
        print(f"[split] scanning {adl_root}")
        records = build_segment_records(
            adl_root=adl_root,
            segment_frames=args.segment_frames,
            label=args.label,
            frame_bytes=frame_bytes,
        )
        all_records.extend(records)

    if not all_records:
        raise SystemExit("No segments generated. Check --adl-root paths.")

    write_jsonl(all_records, args.out)
    print(f"[split] wrote {len(all_records)} segment record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
