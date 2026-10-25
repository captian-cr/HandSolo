"""判别实验：哪种「五指张开」判据能把你自己的手分开。

背景：`_is_five_open` = 四指二维夹角全 ≥135° + 拇指 4→17/掌宽 ≥0.9。
2026-10-22 用户实机截图：只伸食指 → HUD 报「五指张开（待机）」，光标冻结。

本脚本用 gesture_data 里用户真实样本，比较三族判据在
move(只伸食指) / stop(食指+拇指张开) / swipe(双指) / open(五指张开) 上的假阳/真阳：

  A 角度族   ：现役 all(_finger_extensions) + 拇指 4→17/掌宽 ≥0.9
  B 长度族   ：四指 |MCP→tip| / 掌宽 全部 ≥ T
  C 相对族   ：四指 reach 的 min/max ≥ R（各指一样长才算全张开）

用法: python scripts/audit_five_open.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from math import hypot
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from handmouse.contracts import Landmark  # noqa: E402
from handmouse.gestures import _finger_extensions, _is_five_open  # noqa: E402

FINGERS = ((5, 8), (9, 12), (13, 16), (17, 20))


def load(path: Path) -> dict[str, list[tuple[Landmark, ...]]]:
    out: dict[str, list[tuple[Landmark, ...]]] = defaultdict(list)
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
            pts = [Landmark(0.0, 0.0, 0.0)]
            pts += [Landmark(float(feats[2 * i]), float(feats[2 * i + 1]), 0.0) for i in range(20)]
            out[str(rec.get("label"))].append(tuple(pts))
    return out


def reaches(pts: tuple[Landmark, ...]) -> list[float]:
    width = max(hypot(pts[5].x - pts[17].x, pts[5].y - pts[17].y), 1e-4)
    return [hypot(pts[a].x - pts[b].x, pts[a].y - pts[b].y) / width for a, b in FINGERS]


def q(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))]


def main() -> int:
    data = load(ROOT / "gesture_data" / "samples.jsonl")
    labels = [l for l in ("move", "stop", "swipe", "pinch", "fist", "open") if data.get(l)]
    print(f"样本: " + "  ".join(f"{l}={len(data[l])}" for l in labels))

    print("\n【A 现役角度族】四指 reach 分布（÷掌宽；伸直一般 ≥1.3）")
    print(f"{'label':<8}{'食指':>16}{'中指':>16}{'无名':>16}{'小指':>16}")
    print(f"{'':8}" + "  p10/中位/p90 " * 4)
    for label in labels:
        cols = []
        for i in range(4):
            vals = [reaches(p)[i] for p in data[label]]
            cols.append(f"{q(vals, .1):.2f}/{median(vals):.2f}/{q(vals, .9):.2f}")
        print(f"{label:<8}" + "".join(f"{c:>16}" for c in cols))

    print("\n【判据对照】各类被判为「五指张开」的比例")
    header = f"{'label':<8}{'现役A':>8}" + "".join(f"{f'reach≥{t}':>12}" for t in (1.0, 1.1, 1.2, 1.3)) \
        + "".join(f"{f'min/max≥{r}':>14}" for r in (0.6, 0.7, 0.75, 0.8))
    print(header)
    for label in labels:
        rows = data[label]
        n = len(rows)
        line = f"{label:<8}{sum(_is_five_open(p) for p in rows) / n * 100:>7.1f}%"
        for t in (1.0, 1.1, 1.2, 1.3):
            hit = sum(all(r >= t for r in reaches(p)) for p in rows) / n * 100
            line += f"{hit:>11.1f}%"
        for r in (0.6, 0.7, 0.75, 0.8):
            def ok(p: tuple[Landmark, ...]) -> bool:
                v = reaches(p)
                return min(v) / max(v) >= r
            hit = sum(ok(p) for p in rows) / n * 100
            line += f"{hit:>13.1f}%"
        print(line)

    print("\n【组合判据】四指 reach≥T 且 min/max≥R 且 拇指4→17/掌宽≥0.9")
    print(f"{'label':<8}" + "".join(f"{f'T{t}/R{r}':>12}" for t in (1.1, 1.2, 1.3) for r in (0.7, 0.75)))
    for label in labels:
        rows = data[label]
        n = len(rows)
        line = f"{label:<8}"
        for t in (1.1, 1.2, 1.3):
            for r in (0.7, 0.75):
                def ok(p: tuple[Landmark, ...]) -> bool:
                    v = reaches(p)
                    if min(v) < t or min(v) / max(v) < r:
                        return False
                    width = max(hypot(p[5].x - p[17].x, p[5].y - p[17].y), 1e-4)
                    return hypot(p[4].x - p[17].x, p[4].y - p[17].y) / width >= 0.9
                line += f"{sum(ok(p) for p in rows) / n * 100:>11.1f}%"
        print(line)

    # 交叉：只伸食指被误抓的比例 vs 真五指张开被漏掉的比例
    print("\n【取舍曲线】用 T 当唯一长度门槛（含拇指闸），看两头的代价")
    print(f"{'T':>5}{'move误抓':>12}{'stop误抓':>12}{'swipe误抓':>12}{'open漏抓':>12}")
    for t in (0.9, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4):
        def hit(p: tuple[Landmark, ...], t: float = t) -> bool:
            if not all(r >= t for r in reaches(p)):
                return False
            width = max(hypot(p[5].x - p[17].x, p[5].y - p[17].y), 1e-4)
            return hypot(p[4].x - p[17].x, p[4].y - p[17].y) / width >= 0.9
        fp_move = sum(hit(p) for p in data["move"]) / len(data["move"]) * 100
        fp_stop = sum(hit(p) for p in data.get("stop", [])) / max(1, len(data.get("stop", []))) * 100
        fp_swipe = sum(hit(p) for p in data["swipe"]) / len(data["swipe"]) * 100
        miss_open = 100 - sum(hit(p) for p in data["open"]) / len(data["open"]) * 100
        print(f"{t:>5.2f}{fp_move:>11.1f}%{fp_stop:>11.1f}%{fp_swipe:>11.1f}%{miss_open:>11.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
