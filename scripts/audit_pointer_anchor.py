"""量「光标锚点用食指尖还是掌心」——离线对比，不动实机。

为什么需要这个脚本（2026-10-23 用户拍板「按顺序自己做」的第三步）：

他复报过两件事，其实是**同一个根因**——光标锚在食指尖上：

1. 磁吸释放瞬间光标"跳"回随手位置（食指尖比掌心多出来的那一段位移）；
2. 捏合比从 1.42 掉到 0.90 这段，光标还跟着食指走（捏合时食指本来就在动）。

根治办法是把锚点从"食指尖（landmark 8）"换成"掌心（0/5/9/13/17 均值）"。
但这是一次**改所有 MOVE 映射坐标**的改动，不许拿实机当试验田 → 先量。

⚠️ **能量什么、不能量什么**（说清楚，免得把结论用过头）：

* ``gesture_data`` 里存的特征是**减手腕 + 除手长**之后的 40 维，也就是
  **平移与整体缩放都已被消掉**。所以这里量的是"手的形状/姿态抖动"，
  不包含手臂的整体晃动。两个锚点共享同一份整体晃动，所以**两者之比**
  仍然有效，但绝对数值会**低估**真实光标抖动。
* 因此本脚本只报**比值**（掌心 / 食指），不硬换算成屏幕 px——手长在
  画面里占多少像素没有存下来，换算出来是假精度。

用法::

    python scripts/audit_pointer_anchor.py                 # 默认读 samples.jsonl
    python scripts/audit_pointer_anchor.py --with-pending   # 连回流样本一起量
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

#: 同一段内的最大帧间隔（纳秒）。超过就认为中间断过，不拿跨段的位移当抖动。
SEGMENT_GAP_NS = 300_000_000

#: 参与统计的最短段长度（帧）。太短的段算"位移"没有意义。
MIN_SEGMENT_FRAMES = 8

PALM_INDICES = (0, 5, 9, 13, 17)
INDEX_TIP = 8


def load_samples(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            features = record.get("features")
            if not isinstance(features, list) or len(features) != 40:
                continue
            records.append(record)
    return records


def to_points(features: list[float]) -> list[tuple[float, float]]:
    """40 维特征还原成 21 个二维点（0 号点是手腕，恒为原点，不在特征里）。"""

    points = [(0.0, 0.0)]
    for index in range(20):
        points.append((float(features[index * 2]), float(features[index * 2 + 1])))
    return points


def anchors(points: list[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    """返回 ``(食指尖, 掌心)`` 两个二维坐标（手长为单位的相对坐标）。"""

    tip = points[INDEX_TIP]
    palm = (
        sum(points[index][0] for index in PALM_INDICES) / len(PALM_INDICES),
        sum(points[index][1] for index in PALM_INDICES) / len(PALM_INDICES),
    )
    return tip, palm


def segments(records: list[dict]) -> list[list[dict]]:
    ordered = sorted(records, key=lambda r: int(r.get("captured_ns", 0)))
    result: list[list[dict]] = []
    current: list[dict] = []
    last_ns: int | None = None
    for record in ordered:
        now = int(record.get("captured_ns", 0))
        if last_ns is not None and now - last_ns > SEGMENT_GAP_NS:
            if current:
                result.append(current)
            current = []
        current.append(record)
        last_ns = now
    if current:
        result.append(current)
    return result


def _hypot(ax: float, ay: float, bx: float, by: float) -> float:
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def measure(records: list[dict]) -> dict:
    """返回逐帧抖动与段内位移的统计（单位：手长）。"""

    frame_jitter_tip: list[float] = []
    frame_jitter_palm: list[float] = []
    spread_tip: list[float] = []
    spread_palm: list[float] = []

    for segment in segments(records):
        if len(segment) < MIN_SEGMENT_FRAMES:
            continue
        tip_positions: list[tuple[float, float]] = []
        palm_positions: list[tuple[float, float]] = []
        for record in segment:
            tip, palm = anchors(to_points(record["features"]))
            tip_positions.append(tip)
            palm_positions.append(palm)
        for previous, current in zip(tip_positions, tip_positions[1:]):
            frame_jitter_tip.append(_hypot(previous[0], previous[1], current[0], current[1]))
        for previous, current in zip(palm_positions, palm_positions[1:]):
            frame_jitter_palm.append(_hypot(previous[0], previous[1], current[0], current[1]))
        # 段内离首帧的最大偏移：等价于"这一段里光标会被带跑多远"。
        base_tip = tip_positions[0]
        base_palm = palm_positions[0]
        spread_tip.append(
            max(_hypot(base_tip[0], base_tip[1], x, y) for x, y in tip_positions)
        )
        spread_palm.append(
            max(_hypot(base_palm[0], base_palm[1], x, y) for x, y in palm_positions)
        )

    def med(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    return {
        "frames": len(frame_jitter_tip),
        "jitter_tip": med(frame_jitter_tip),
        "jitter_palm": med(frame_jitter_palm),
        "spread_tip": med(spread_tip),
        "spread_palm": med(spread_palm),
        "segments": len(spread_tip),
    }


def print_table(name: str, per_label: dict[str, dict]) -> None:
    print(f"\n-- {name} --")
    header = (
        f"  {'标签':<8}{'段数':>5}{'帧对':>7}"
        f"{'逐帧抖动 食指':>15}{'掌心':>9}{'掌心/食指':>11}"
        f"{'段内位移 食指':>15}{'掌心':>9}{'掌心/食指':>11}"
    )
    print(header)
    for label in sorted(per_label):
        row = per_label[label]
        if not row["segments"]:
            continue
        jitter_ratio = row["jitter_palm"] / row["jitter_tip"] if row["jitter_tip"] else 0.0
        spread_ratio = (
            row["spread_palm"] / row["spread_tip"] if row["spread_tip"] else 0.0
        )
        print(
            f"  {label:<8}{row['segments']:>5}{row['frames']:>7}"
            f"{row['jitter_tip']:>15.4f}{row['jitter_palm']:>9.4f}{jitter_ratio:>11.0%}"
            f"{row['spread_tip']:>15.4f}{row['spread_palm']:>9.4f}{spread_ratio:>11.0%}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        default=str(PROJECT / "gesture_data" / "samples.jsonl"),
        help="细分标注样本库（默认 gesture_data/samples.jsonl）",
    )
    parser.add_argument(
        "--with-pending",
        action="store_true",
        help="把 pending.jsonl（拒识回流）也算进来：它的帧量比标注库大 7 倍",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples)
    records = load_samples(samples_path)
    if not records:
        print(f"没有可用样本：{samples_path}")
        return 1

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("label", "?")), []).append(record)

    per_label = {label: measure(items) for label, items in grouped.items()}
    print(f"样本库 {samples_path.name}：{len(records)} 帧 / {len(grouped)} 类")
    print_table("细分标注库（samples.jsonl）", per_label)

    total = measure(records)
    print(
        f"\n  合计：{total['segments']} 段 / {total['frames']} 帧对"
        f"  逐帧抖动 掌心 {total['jitter_palm'] / total['jitter_tip']:.0%} of 食指"
        f"  段内位移 掌心 {total['spread_palm'] / total['spread_tip']:.0%} of 食指"
    )

    if args.with_pending:
        pending_path = samples_path.parent / "pending.jsonl"
        pending = load_samples(pending_path)
        if pending:
            pool = {label: [] for label in grouped}
            for label in grouped:
                pool[label] = grouped[label]
            # pending 是拒识帧（多为真实使用中的模糊姿态），单独量一张表，
            # 不混进标注库——否则"标注库很干净"这个对照就没了。
            print_table(
                f"拒识回流（pending.jsonl，{len(pending)} 帧，整体）",
                {"unknown": measure(pending)},
            )
        else:
            print(f"\n  没有读到 {pending_path}")

    print(
        "\n  读法：比值越小 = 换成掌心锚点后越稳。"
        "\n  ⚠️ 特征已减去整体位移，所以这些数是**形状抖动**，不含手臂晃动；"
        "\n     两个锚点共享那份晃动，比值有效，绝对值会低估真实抖动。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
