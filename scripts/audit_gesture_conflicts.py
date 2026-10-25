"""离线回放：用真实样本检测「只伸食指被判成五指张开」的误报率。

判据全部 import 自 handmouse.gestures（不复制），所以测的就是线上代码。
40 维特征只含 x/y（相对手腕、除以手长），但
  * ``_angle`` 只看二维夹角      → 对尺度不变
  * ``_is_five_open`` 等只看比值 → 尺度在分子分母抵消
所以这些判据可以在归一化特征上精确复现，无需原始 21 点。

用法:
    python scripts/audit_gesture_conflicts.py [samples.jsonl ...]
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from handmouse.contracts import Landmark  # noqa: E402
from handmouse.gestures import (  # noqa: E402
    _finger_extensions,
    _is_five_open,
    _is_open_palm,
    _thumb_spread_value,
)


def points_from_features(features: list[float]) -> tuple[Landmark, ...]:
    """40 维 → 21 点（0 号点固定为原点，全部 z=0）。"""

    pts = [Landmark(x=0.0, y=0.0, z=0.0)]
    for i in range(20):
        pts.append(Landmark(x=features[2 * i], y=features[2 * i + 1], z=0.0))
    return tuple(pts)


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or [
        ROOT / "gesture_data" / "samples.jsonl",
        ROOT / "gesture_data" / "pending.jsonl",
    ]
    for path in paths:
        if not path.exists():
            print(f"[skip] {path} 不存在")
            continue
        frames: dict[str, list[dict]] = defaultdict(list)
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                feats = rec.get("features")
                if not isinstance(feats, list) or len(feats) != 40:
                    continue
                pts = points_from_features([float(v) for v in feats])
                label = str(rec.get("label", "?"))
                ext = _finger_extensions(pts)
                frames[label].append(
                    {
                        "five": _is_five_open(pts),
                        "open_palm": _is_open_palm(pts),
                        "ext": ext,
                        "thumb47": _palm_ratio(pts),
                        "thumb45": _thumb_spread_value(pts),
                    }
                )

        print(f"\n{'=' * 78}\n{path.name}  共 {sum(len(v) for v in frames.values())} 帧")
        print(f"{'label':<10}{'帧数':>6}  {'five_open误报':>13}  {'open_palm':>10}  "
              f"{'四指全伸':>8}  {'拇指47中位':>10}  {'拇指45中位':>10}")
        for label in sorted(frames, key=lambda k: -len(frames[k])):
            rows = frames[label]
            n = len(rows)
            five = sum(r["five"] for r in rows) / n
            op = sum(r["open_palm"] for r in rows) / n
            all_ext = sum(all(r["ext"]) for r in rows) / n
            print(
                f"{label:<10}{n:>6}  {five * 100:>12.1f}%  {op * 100:>9.1f}%  "
                f"{all_ext * 100:>7.1f}%  "
                f"{median([r['thumb47'] for r in rows]):>10.2f}  "
                f"{median([r['thumb45'] for r in rows]):>10.2f}"
            )

        print(f"\n-- 各 label 的单指伸直率（'只伸食指'应只有食指高）--")
        print(f"{'label':<10}{'食指':>8}{'中指':>8}{'无名':>8}{'小指':>8}")
        for label in sorted(frames, key=lambda k: -len(frames[k])):
            rows = frames[label]
            n = len(rows)
            cols = [sum(r["ext"][i] for r in rows) / n * 100 for i in range(4)]
            print(f"{label:<10}" + "".join(f"{c:>7.1f}%" for c in cols))

        # 关键冲突证据：只伸食指的帧里，有多少被判成五指张开
        move = frames.get("move") or frames.get("point") or []
        if move:
            bad = [r for r in move if r["five"]]
            print(
                f"\n🔴 '只伸食指'帧 {len(move)} 中判成 five_open（→待机冻光标）: "
                f"{len(bad)} = {len(bad) / len(move) * 100:.1f}%"
            )
            if bad:
                print(
                    "   这些误报帧的拇指47(4→17/掌宽) 分位: "
                    f"p10={pct([r['thumb47'] for r in bad], 0.1):.2f} "
                    f"中位={median([r['thumb47'] for r in bad]):.2f} "
                    f"p90={pct([r['thumb47'] for r in bad], 0.9):.2f} "
                    f"(门槛 0.90)"
                )
                print(
                    "   同一批帧的拇指45(4→5/掌宽) 分位: "
                    f"p10={pct([r['thumb45'] for r in bad], 0.1):.2f} "
                    f"中位={median([r['thumb45'] for r in bad]):.2f} "
                    f"p90={pct([r['thumb45'] for r in bad], 0.9):.2f}"
                )
        # 反向：真五指张开是否也在
        for label in ("open", "fist", "pinch", "stop"):
            rows = frames.get(label)
            if rows:
                five = sum(r["five"] for r in rows) / len(rows)
                print(f"   对照 {label}: five_open 命中 {five * 100:.1f}%")
    return 0


def _palm_ratio(pts: tuple[Landmark, ...]) -> float:
    """_is_five_open 实际用的拇指外展比 = dist(4,17)/掌宽。"""

    from math import hypot

    width = max(hypot(pts[5].x - pts[17].x, pts[5].y - pts[17].y), 1e-4)
    return hypot(pts[4].x - pts[17].x, pts[4].y - pts[17].y) / width


if __name__ == "__main__":
    raise SystemExit(main())
