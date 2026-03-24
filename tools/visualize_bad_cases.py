"""
RACA v2 bad-case visualization and heuristic analysis utility.

This script reads the failed sample list exported by ``offline_infer.py`` and
renders RD/RA/RE feature summaries for manual inspection.

Usage::

    python tools/visualize_bad_cases.py \
        --input bad_samples.txt \
        --out-dir bad_case_plots
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from statistics import median
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np


INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

PATTERN_ORDER = [
    "missed_fall",
    "adl_false_alarm",
    "sit_false_alarm",
    "walk_false_alarm",
    "other",
]

PATTERN_TITLES = {
    "missed_fall": "漏检跌倒",
    "adl_false_alarm": "ADL_5min 高置信误报",
    "sit_false_alarm": "sit / sit_to_chair / sit_to_floor 过渡类误报",
    "walk_false_alarm": "walk 误报",
    "other": "其他错例",
}

PATTERN_NOTES = {
    "missed_fall": (
        "经验归纳：从 RD/RA/RE/RT/DT/AT 汇总图看，这类样本往往只在少数帧里出现短促能量突变，"
        "RT/DT 轨迹偏弱或持续时间偏短，累积图中的主能量团也更分散，所以模型更容易把它们压成 non-fall。"
    ),
    "adl_false_alarm": (
        "经验归纳：这类 ADL 窗口通常带有连续的人体活动和明显的速度变化，RT/DT 图里常出现较长时段的活动带，"
        "如果其中恰好夹杂快速起坐、转身或弯腰，模型容易把持续动作窗口误当成 fall。"
    ),
    "sit_false_alarm": (
        "经验归纳：坐下、坐椅子和坐地动作在 RT/DT 图上经常带有一次明显的向下过渡和短时速度峰值，"
        "累积 RA/RE 图也常保留紧凑的人体能量团，视觉上和真实跌倒的后半段很接近，因此最容易形成高置信误报。"
    ),
    "walk_false_alarm": (
        "经验归纳：walk 误报通常出现在包含突然加减速、转向或停止的窗口里，DT 图会有更宽的速度展开，"
        "但 RT 图未必呈现持续稳定的平移，所以模型可能把单个异常片段放大成 fall。"
    ),
    "other": (
        "经验归纳：这一组不属于当前的高频模式，建议优先结合对应图片逐个排查，再决定是否需要细分成新的错误桶。"
    ),
}


def load_sample(sample_dir: str):
    """Load one extracted sample directory."""
    try:
        rd = np.load(os.path.join(sample_dir, "RD.npy"))  # (T, Range, Doppler)
        ra = np.load(os.path.join(sample_dir, "RA.npy"))  # (T, Angle, Range)
        re = np.load(os.path.join(sample_dir, "RE.npy"))  # (T, Angle, Range)

        meta_path = os.path.join(sample_dir, "meta.json")
        meta = {}
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        return rd, ra, re, meta
    except Exception as exc:
        print(f"[Error] Failed to load sample {sample_dir}: {exc}")
        return None, None, None, None


def _normalize_path(path: str) -> str:
    return os.path.abspath(path)


def _safe_filename(name: str) -> str:
    cleaned = INVALID_FILENAME_CHARS.sub("_", str(name)).strip()
    return cleaned or "sample"


def plot_radar_features(rd, ra, re, meta, save_path: str):
    """Render a 2x3 summary plot for one failed sample."""
    true_label = meta.get("label", "Unknown")
    clip_id = meta.get("clip_id", os.path.basename(os.path.dirname(save_path)))

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        f"Bad Case Analysis | Clip: {clip_id} | True Label: {true_label}",
        fontsize=16,
    )

    # Spatial accumulation views.
    rd_acc = np.max(rd, axis=0)
    ra_acc = np.max(ra, axis=0).T
    re_acc = np.max(re, axis=0).T

    ax = axes[0, 0]
    im = ax.imshow(rd_acc, aspect="auto", origin="lower", cmap="jet")
    ax.set_title("Accumulated RD Map")
    ax.set_ylabel("Range Bins")
    ax.set_xlabel("Doppler Bins")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(ra_acc, aspect="auto", origin="lower", cmap="jet")
    ax.set_title("Accumulated RA Map (Azimuth)")
    ax.set_ylabel("Range Bins")
    ax.set_xlabel("Angle Bins")
    fig.colorbar(im, ax=ax)

    ax = axes[0, 2]
    im = ax.imshow(re_acc, aspect="auto", origin="lower", cmap="jet")
    ax.set_title("Accumulated RE Map (Elevation)")
    ax.set_ylabel("Range Bins")
    ax.set_xlabel("Angle Bins")
    fig.colorbar(im, ax=ax)

    # Temporal projections.
    rt_map = np.max(rd, axis=2).T
    dt_map = np.max(rd, axis=1).T
    at_map = np.max(ra, axis=2).T

    ax = axes[1, 0]
    im = ax.imshow(rt_map, aspect="auto", origin="lower", cmap="jet")
    ax.set_title("Range-Time (RT) Map")
    ax.set_ylabel("Range Bins")
    ax.set_xlabel("Time Frames")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(dt_map, aspect="auto", origin="lower", cmap="jet")
    ax.set_title("Doppler-Time (DT) Map")
    ax.set_ylabel("Doppler Bins")
    ax.set_xlabel("Time Frames")
    fig.colorbar(im, ax=ax)

    ax = axes[1, 2]
    im = ax.imshow(at_map, aspect="auto", origin="lower", cmap="jet")
    ax.set_title("Azimuth-Time (AT) Map")
    ax.set_ylabel("Angle Bins")
    ax.set_xlabel("Time Frames")
    fig.colorbar(im, ax=ax)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def read_sample_paths(input_path: str) -> list[str]:
    with open(input_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def visualize_samples(sample_paths: Iterable[str], out_dir: str, limit: int = 0) -> dict[str, str]:
    sample_paths = [_normalize_path(path) for path in sample_paths if str(path).strip()]
    if limit > 0:
        sample_paths = sample_paths[:limit]

    if not sample_paths:
        print("[Info] No failed samples to visualize.")
        return {}

    os.makedirs(out_dir, exist_ok=True)

    plot_map: dict[str, str] = {}
    total = len(sample_paths)
    for i, sample_dir in enumerate(sample_paths, 1):
        print(f"[{i}/{total}] Processing: {os.path.basename(sample_dir)}", end="", flush=True)

        rd, ra, re, meta = load_sample(sample_dir)
        if rd is None:
            print(" -> skipped (incomplete data)")
            continue

        clip_id = meta.get("clip_id", f"sample_{i}")
        save_filename = f"{_safe_filename(clip_id)}.png"
        save_path = os.path.abspath(os.path.join(out_dir, save_filename))

        plot_radar_features(rd, ra, re, meta, save_path)
        plot_map[_normalize_path(sample_dir)] = save_path
        print(" -> done")

    print(f"\n[Done] Generated {len(plot_map)} plot(s) in: {out_dir}")
    return plot_map


def visualize_bad_case_file(input_path: str, out_dir: str, limit: int = 0) -> dict[str, str]:
    if not os.path.exists(input_path):
        print(f"[Error] Input file not found: {input_path}")
        return {}

    sample_paths = read_sample_paths(input_path)
    total = len(sample_paths)
    if total == 0:
        print("[Info] Input file is empty, nothing to visualize.")
        return {}

    if limit > 0:
        print(f"[Info] Found {total} failed sample(s); visualizing the first {min(limit, total)}.")
    else:
        print(f"[Info] Found {total} failed sample(s); generating plots...")

    return visualize_samples(sample_paths, out_dir=out_dir, limit=limit)


def _is_true_fall(label: str) -> bool:
    return str(label).lower().strip() in ("fall", "falls")


def _categorize_failed_record(record: Mapping[str, object]) -> str:
    true_label = str(record.get("true", "")).lower().strip()
    pred_label = str(record.get("pred", "")).lower().strip()
    sample_name = os.path.basename(str(record.get("path", ""))).lower()
    parent_name = os.path.basename(os.path.dirname(str(record.get("path", "")))).lower()

    if _is_true_fall(true_label) and pred_label != "fall":
        return "missed_fall"

    if pred_label == "fall":
        if "adl_5min" in sample_name:
            return "adl_false_alarm"
        if parent_name == "sit" or true_label.startswith("sit") or sample_name.startswith("sit"):
            return "sit_false_alarm"
        if parent_name == "walk" or true_label == "walk" or sample_name.startswith("walk"):
            return "walk_false_alarm"

    return "other"


def _prob_stats(records: list[Mapping[str, object]]) -> tuple[float, float, float]:
    probs = [float(record["prob"]) for record in records]
    return min(probs), float(median(probs)), max(probs)


def _format_prob_stats(records: list[Mapping[str, object]]) -> str:
    prob_min, prob_med, prob_max = _prob_stats(records)
    return f"min={prob_min:.1%}, median={prob_med:.1%}, max={prob_max:.1%}"


def _record_sort_key(record: Mapping[str, object], category: str) -> tuple[float, str]:
    prob = float(record["prob"])
    path = str(record["path"])
    if category == "missed_fall":
        return (prob, path)
    return (-prob, path)


def _format_plot_link(record: Mapping[str, object], plot_map: Mapping[str, str], report_path: str) -> str:
    plot_path = plot_map.get(_normalize_path(str(record["path"])))
    if not plot_path:
        return ""

    report_dir = os.path.dirname(os.path.abspath(report_path)) if report_path else os.getcwd()
    rel_path = os.path.relpath(plot_path, report_dir).replace(os.sep, "/")
    return f" | 图: [{os.path.basename(plot_path)}]({rel_path})"


def _group_failed_records(
    failed_records: Iterable[Mapping[str, object]],
) -> dict[str, list[Mapping[str, object]]]:
    grouped = {key: [] for key in PATTERN_ORDER}
    for record in failed_records:
        grouped[_categorize_failed_record(record)].append(record)
    return grouped


def build_bad_case_report(
    failed_records: Iterable[Mapping[str, object]],
    plot_map: Mapping[str, str] | None = None,
    report_path: str = "",
) -> str:
    records = list(failed_records)
    plot_map = {_normalize_path(k): os.path.abspath(v) for k, v in (plot_map or {}).items()}
    grouped = _group_failed_records(records)

    lines = [
        "# 错例模式分析报告",
        "",
        f"共识别出 **{len(records)}** 个错例。",
        "",
        "> 说明：以下结论是基于 RD/RA/RE/RT/DT/AT 汇总图的经验归纳，用于辅助排查，不应视为模型因果解释。",
        "",
        "## 总览",
        "",
    ]

    for category in PATTERN_ORDER:
        bucket = grouped[category]
        if not bucket:
            continue
        lines.append(f"- {PATTERN_TITLES[category]}: {len(bucket)} 个，{_format_prob_stats(bucket)}")

    for category in PATTERN_ORDER:
        bucket = grouped[category]
        if not bucket:
            continue

        lines.extend(
            [
                "",
                f"## {PATTERN_TITLES[category]}",
                "",
                f"- 数量: {len(bucket)}",
                f"- 概率统计: {_format_prob_stats(bucket)}",
                f"- 经验归纳: {PATTERN_NOTES[category]}",
                "- 代表样本:",
            ]
        )

        top_records = sorted(bucket, key=lambda item: _record_sort_key(item, category))[:3]
        for record in top_records:
            plot_link = _format_plot_link(record, plot_map, report_path)
            sample_name = os.path.basename(str(record["path"]))
            lines.append(
                f"  - `{sample_name}` | true=`{record['true']}` pred=`{record['pred']}` "
                f"| prob=`{float(record['prob']):.1%}`{plot_link} | path=`{record['path']}`"
            )

    return "\n".join(lines) + "\n"


def analyze_bad_cases(
    failed_records: Iterable[Mapping[str, object]],
    plot_map: Mapping[str, str] | None = None,
    report_path: str = "",
) -> str:
    records = list(failed_records)
    if not records:
        print("[Analysis] 没有错例可分析。")
        return ""

    plot_map = plot_map or {}
    grouped = _group_failed_records(records)

    print("\n[Analysis] 错例模式摘要")
    for category in PATTERN_ORDER:
        bucket = grouped[category]
        if not bucket:
            continue
        print(f"  - {PATTERN_TITLES[category]}: {len(bucket)} 个, {_format_prob_stats(bucket)}")

    report_text = build_bad_case_report(records, plot_map=plot_map, report_path=report_path)

    if report_path:
        report_dir = os.path.dirname(os.path.abspath(report_path))
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[Analysis] Markdown 报告已保存到: {report_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser(
        description="Visualize failed samples exported by offline_infer.py."
    )
    parser.add_argument("--input", required=True, help="Path to bad_samples.txt")
    parser.add_argument(
        "--out-dir",
        default="bad_case_plots",
        help="Directory where visualization images will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to visualize (0 means all).",
    )
    args = parser.parse_args()

    plot_map = visualize_bad_case_file(
        input_path=args.input,
        out_dir=args.out_dir,
        limit=args.limit,
    )
    if not plot_map and not os.path.exists(args.input):
        sys.exit(1)


if __name__ == "__main__":
    main()
