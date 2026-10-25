"""诊断个人手势样本库：逐帧找出污染训练集的样本。

用法::

    python scripts/diagnose_gestures.py            # 按实际使用的粗分类分析
    python scripts/diagnose_gestures.py --raw      # 看采集时的细分标签

输出四块：
1. 每类类内距离分布——散得越开越难和其他类分开；
2. 跨类最小距离——和类内分布一比就知道有没有重叠；
3. 每类首尾的"异类帧"——采集窗口边界上的过渡帧，标错了标签；
4. 时间序列突变点——同一阶段内特征突跳，通常是换手/关键点飘了。

默认按 **point / open / fist** 粗分类分析，因为那才是识别时真正使用的
粒度；移动/停住/捏合三者在单帧距离上没有边界（实测类间最近 0.37 小于
类内 p90 0.69），按细分看会得到一堆"冲突"的错觉。
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from handmouse.gesture_data import (  # noqa: E402
    LABEL_NAMES,
    LABELS,
    GestureDataset,
    GestureSample,
    default_dataset_path,
)
from handmouse.gesture_model import (  # noqa: E402
    COARSE_MAP,
    COARSE_NAMES,
    _distance,
)


def _name(label: str) -> str:
    return LABEL_NAMES.get(label) or COARSE_NAMES.get(label, label)


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(int(len(ordered) * ratio), len(ordered) - 1)
    return ordered[position]


def _intra_nearest(samples) -> list[float]:
    result: list[float] = []
    for index, sample in enumerate(samples):
        nearest = min(
            (
                _distance(sample.features, other.features)
                for other_index, other in enumerate(samples)
                if other_index != index
            ),
            default=None,
        )
        if nearest is not None:
            result.append(nearest)
    return result


def _cross_nearest(left, right) -> float:
    return min(
        _distance(first.features, second.features)
        for first in left
        for second in right
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="诊断手势样本库")
    parser.add_argument("--path", default="", help="样本库路径")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="按采集时的细分标签分析（默认按 point/open/fist 粗分类）",
    )
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else default_dataset_path()
    dataset = GestureDataset.load(path)
    if not len(dataset):
        print(f"[错误] {path} 里没有样本")
        return 1

    if args.raw:
        analysis = dataset
        order = LABELS
    else:
        analysis = GestureDataset(
            GestureSample(
                label=COARSE_MAP.get(sample.label, sample.label),
                features=sample.features,
                captured_ns=sample.captured_ns,
                source=sample.source,
            )
            for sample in dataset
        )
        order = tuple(COARSE_MAP.get(label, label) for label in LABELS)
        order = tuple(dict.fromkeys(order))

    print(f"样本库：{path}")
    print(f"总样本：{len(dataset)}　分析粒度：{'细分标签' if args.raw else '粗分类'}")
    print(f"批次：{sorted({s.source for s in dataset})}\n")

    present = [label for label in order if len(analysis.for_label(label)) >= 3]

    print("=" * 64)
    print("1) 类内距离（同类最近邻距离。越大说明这个手势做得越不稳定）")
    print("=" * 64)
    intra_map: dict[str, list[float]] = {}
    for label in present:
        intra = _intra_nearest(analysis.for_label(label))
        intra_map[label] = intra
        print(
            f"  {_name(label):<24} {len(intra) + 1:>4} 帧"
            f"　中位 {statistics.median(intra):.3f}"
            f"　p90 {_percentile(intra, 0.90):.3f}"
            f"　最大 {max(intra):.3f}",
        )

    print()
    print("=" * 64)
    print("2) 跨类最小距离（小于类内 p90 就是重叠区）")
    print("=" * 64)
    overlaps: list[tuple[str, str, float]] = []
    for first_index, left_label in enumerate(present):
        for right_label in present[first_index + 1 :]:
            distance = _cross_nearest(
                analysis.for_label(left_label),
                analysis.for_label(right_label),
            )
            ceiling = max(
                _percentile(intra_map.get(left_label, [0.0]), 0.90),
                _percentile(intra_map.get(right_label, [0.0]), 0.90),
            )
            flag = "　← 重叠" if distance < ceiling else ""
            print(
                f"  {_name(left_label):<22} ↔ {_name(right_label):<22}"
                f" 最近 {distance:.3f}{flag}",
            )
            if flag:
                overlaps.append((left_label, right_label, distance))

    print()
    print("=" * 64)
    print("3) 采集窗口边界的过渡帧（最近邻明显更像别的类 = 可能标错）")
    print("=" * 64)
    suspects = 0
    for label in present:
        samples = sorted(analysis.for_label(label), key=lambda s: s.captured_ns)
        typical = statistics.median(intra_map.get(label, [1.0]))
        foreign: list[tuple[int, str, float]] = []
        for index, sample in enumerate(samples):
            best_label = ""
            best_distance = float("inf")
            for other in analysis:
                if other.label == label:
                    continue
                distance = _distance(sample.features, other.features)
                if distance < best_distance:
                    best_distance = distance
                    best_label = other.label
            if best_distance < typical * 0.8:
                foreign.append((index + 1, best_label, best_distance))
        suspects += len(foreign)
        if not foreign:
            print(f"  {_name(label):<24} 干净")
            continue
        head = sum(1 for item in foreign if item[0] <= 5)
        tail = sum(1 for item in foreign if item[0] > len(samples) - 5)
        print(
            f"  {_name(label):<24} {len(foreign)}/{len(samples)} 帧更像别的类"
            f"（开头 {head}、结尾 {tail}、中间 {len(foreign) - head - tail}）",
        )
        for position, best_label, best_distance in foreign[:5]:
            print(
                f"      第 {position:>3} 帧 → 更像「{_name(best_label)}」"
                f" 距离 {best_distance:.3f}",
            )

    print()
    print("=" * 64)
    print("4) 同一阶段内的特征突跳（相邻帧中位的 4 倍以上，多为关键点飘了）")
    print("=" * 64)
    for label in present:
        samples = sorted(analysis.for_label(label), key=lambda s: s.captured_ns)
        steps = [
            _distance(samples[index].features, samples[index + 1].features)
            for index in range(len(samples) - 1)
        ]
        if not steps:
            continue
        typical = statistics.median(steps)
        spikes = [
            (index + 1, step)
            for index, step in enumerate(steps)
            if step > max(typical * 4.0, 0.6)
        ]
        if not spikes:
            print(f"  {_name(label):<24} 平稳（相邻帧中位 {typical:.3f}）")
            continue
        print(
            f"  {_name(label):<24} {len(spikes)} 处突跳"
            f"（相邻帧中位 {typical:.3f}）",
        )
        for position, step in spikes[:4]:
            print(f"      第 {position}→{position + 1} 帧 跳变 {step:.3f}")

    print()
    print("=" * 64)
    print("结论")
    print("=" * 64)
    if overlaps:
        for left, right, distance in overlaps:
            print(f"  仍重叠：{_name(left)} ↔ {_name(right)}（最近 {distance:.3f}）")
        print("  → 靠调样本救不了，要么把手势做得更夸张，要么交给时序逻辑区分。")
    else:
        print("  各分类之间没有重叠，样本库可用。")
    if suspects:
        print(f"  可疑过渡帧 {suspects} 帧：检查采集窗口边界是否录进了上一个手势。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
