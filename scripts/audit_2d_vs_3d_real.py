"""2D 判据 vs 3D 判据：在**带米制 3D 的真实采集样本**上逐项对比。

用 ``gesture_data/samples.jsonl`` 里带 ``world`` 字段的样本：同一次采集同时
存了归一化 2D 特征和米制 3D 世界坐标，因此能在**同一批帧**上让两套尺子
各判一次，直接比"召回 / 误触"。

三个候选判据（都是"食指伸出 且 中指伸出 且（无名指 或 小指没伸出）"）：

* ``2D 距离``：现行主判据，指尖到掌心距离 ÷ 本帧最长
* ``2D 伸直``：逐指伸直度（MCP→TIP ÷ 三节骨长），量在归一化图像坐标上
* ``3D 伸直``：同一个量，量在米制 3D 世界坐标上

运行：python scripts/audit_2d_vs_3d_real.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from handmouse.contracts import Landmark  # noqa: E402
from handmouse.gestures import (  # noqa: E402
    _FINGER_CHAINS,
    _is_two_finger,
    _is_two_finger_hold,
    _two_finger_diag,
)

SAMPLES = ROOT / "gesture_data" / "samples.jsonl"
LABELS = ("swipe", "move", "stop", "pinch", "open", "fist")
CONFIRM_MS = 150.0


def to_2d(features: list[float]) -> tuple[Landmark, ...] | None:
    if len(features) != 40:
        return None
    points = [Landmark(0.0, 0.0, 0.0)]
    for i in range(20):
        points.append(Landmark(float(features[i * 2]), float(features[i * 2 + 1]), 0.0))
    return tuple(points)


def to_3d(world: list[float]) -> tuple[Landmark, ...] | None:
    if len(world) != 63:
        return None
    return tuple(
        Landmark(float(world[i * 3]), float(world[i * 3 + 1]), float(world[i * 3 + 2]))
        for i in range(21)
    )


def d2(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def d3(a: Landmark, b: Landmark) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def straight(points, chain, dist) -> float:
    mcp, pip, dip, tip = chain
    path = dist(points[mcp], points[pip]) + dist(points[pip], points[dip]) + dist(points[dip], points[tip])
    return dist(points[mcp], points[tip]) / path if path > 1e-9 else 0.0


def load():
    seq: dict[tuple[str, str], list] = defaultdict(list)
    with SAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            label = raw.get("label")
            if label not in LABELS:
                continue
            p2 = to_2d(raw.get("features") or [])
            p3 = to_3d(raw.get("world") or [])
            if p2 is None or p3 is None:
                continue
            seq[(label, raw.get("source", "?"))].append(
                (int(raw.get("captured_ns", 0)), p2, p3)
            )
    for key in seq:
        seq[key].sort(key=lambda item: item[0])
    return seq


def feats2(points) -> dict[str, float]:
    return {n: straight(points, ch, d2) for n, ch in _FINGER_CHAINS.items()}


def feats3(points) -> dict[str, float]:
    return {n: straight(points, ch, d3) for n, ch in _FINGER_CHAINS.items()}


def make_rule(kind: str, ext: float = 0.90, curl: float = 0.90):
    if kind == "2d-distance":
        return lambda p2, p3, _f2, _f3: _is_two_finger(_two_finger_diag(p2))
    if kind == "2d-straight":
        return lambda p2, p3, f2, _f3: (
            f2["index"] >= ext
            and f2["middle"] >= ext
            and (f2["ring"] <= curl or f2["pinky"] <= curl)
        )
    if kind == "3d-straight":
        return lambda p2, p3, _f2, f3: (
            f3["index"] >= ext
            and f3["middle"] >= ext
            and (f3["ring"] <= curl or f3["pinky"] <= curl)
        )
    if kind == "3d-or-2d":
        r3 = make_rule("3d-straight", ext, curl)
        return lambda p2, p3, f2, f3: r3(p2, p3, f2, f3) or _is_two_finger(_two_finger_diag(p2))
    if kind == "3d-only-middle":
        # 2D 距离判据 + 「3D 说中指确实蜷着就否决」
        def rule(p2, p3, _f2, f3):
            if not _is_two_finger(_two_finger_diag(p2)):
                return False
            return f3["middle"] >= ext

        return rule
    raise ValueError(kind)


def simulate(frames, rule, hold_ms: int = 450, confirm_ms: float = CONFIRM_MS):
    hold_ns = hold_ms * 1_000_000
    confirm_ns = confirm_ms * 1_000_000
    active = False
    last_seen = None
    hit_prev = None
    hit_cur = None
    count = 0
    entries = 0
    was_active = False
    for ts, p2, p3 in frames:
        f2 = feats2(p2)
        f3 = feats3(p3)
        hit = rule(p2, p3, f2, f3)
        if hit:
            hit_prev = hit_cur
            hit_cur = ts
        confirmed = hit_prev is not None and hit_cur is not None and hit_cur - hit_prev <= confirm_ns
        if hit and (confirmed or active):
            last_seen = ts
            active = True
        elif (
            active
            and last_seen is not None
            and ts - last_seen <= hold_ns
            and _is_two_finger_hold(_two_finger_diag(p2))
        ):
            active = True
        else:
            active = False
            last_seen = None
        if active:
            count += 1
            if not was_active:
                entries += 1
        was_active = active
    return count, len(frames), entries


def evaluate(seq, rule):
    per = {}
    for label in LABELS:
        tot = hit = 0
        for (lb, _src), frames in seq.items():
            if lb != label or len(frames) < 5:
                continue
            h, n, _e = simulate(frames, rule)
            tot += n
            hit += h
        per[label] = hit / tot * 100 if tot else 0.0
    return per


def main() -> None:
    seq = load()
    total = sum(len(v) for v in seq.values())
    print(f"带 3D 的样本会话 {len(seq)} 个，总帧 {total}")

    print(f"\n{'判据':<26}{'swipe':>8}" + "".join(f"{l:>8}" for l in LABELS[1:]) + f"{'最大误触':>9}")
    candidates = [
        ("2D 距离（现行主判据）", make_rule("2d-distance")),
        ("2D 伸直 ext.90/curl.90", make_rule("2d-straight", 0.90, 0.90)),
        ("2D 伸直 ext.85/curl.90", make_rule("2d-straight", 0.85, 0.90)),
        ("3D 伸直 ext.90/curl.90", make_rule("3d-straight", 0.90, 0.90)),
        ("3D 伸直 ext.88/curl.92", make_rule("3d-straight", 0.88, 0.92)),
        ("3D 伸直 ext.85/curl.92", make_rule("3d-straight", 0.85, 0.92)),
        ("3D ∪ 2D距离", make_rule("3d-or-2d", 0.88, 0.92)),
        ("2D距离 + 3D中指否决(.88)", make_rule("3d-only-middle", 0.88)),
        ("2D距离 + 3D中指否决(.85)", make_rule("3d-only-middle", 0.85)),
    ]
    for name, rule in candidates:
        per = evaluate(seq, rule)
        worst = max(per[lb] for lb in LABELS[1:])
        flag = "  <<<" if worst < 1.0 else ""
        print(
            f"{name:<26}{per['swipe']:>7.1f}%"
            + "".join(f"{per[l]:7.1f}%" for l in LABELS[1:])
            + f"{worst:>8.1f}%{flag}"
        )

    # 逐指数值对照：2D 与 3D 在"该伸直"和"该蜷"上的分离度
    print("\n=== 逐指分离度（中位数）===")
    for label in ("swipe", "move"):
        print(f"  [{label}]")
        for n in ("index", "middle", "ring", "pinky"):
            v2: list[float] = []
            v3: list[float] = []
            for (lb, _src), frames in seq.items():
                if lb != label:
                    continue
                for _ts, p2, p3 in frames:
                    v2.append(feats2(p2)[n])
                    v3.append(feats3(p3)[n])
            v2.sort()
            v3.sort()
            m2 = v2[len(v2) // 2] if v2 else 0.0
            m3 = v3[len(v3) // 2] if v3 else 0.0
            print(f"    {n:<7} 2D 中位 {m2:.3f}   3D 中位 {m3:.3f}")


if __name__ == "__main__":
    main()
