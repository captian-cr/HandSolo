"""用用户自己的真实样本离线审计「双指滑动」几何判据。

背景（2026-10-22 第四次复报）：用户正对镜头持双指时双指永远不触发。
根因是判据里的绝对阈值 ``reach >= 1.3``（= |MCP→tip| ÷ **掌宽**）——
掌宽在正对镜头时是满宽（约 8cm），伸直的食指比例只有 ~1.0；
只有把手侧过来、掌宽被透视压扁到 4~5cm 时才够 1.3。
所以这条判据本质是「侧手专用」，正手必失效。

本脚本把判据改成**以手长为尺**（|腕→中指 MCP|，旋转下最稳定的长度基准），
在真实样本上做网格搜索，找出兼顾召回与误报的门限。

特征文件里存的是 (x, y)，已按手长归一化并以手腕为原点，
所以任意两点距离除以 ``hypot(f[16], f[17])`` 就是「以手长为尺」的真实比值，
不需要重新采集。

运行：python scripts/audit_swipe_geom.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "gesture_data" / "samples.jsonl"
LABELS = ("move", "stop", "pinch", "open", "fist", "swipe")


def _pt(f, index: int) -> tuple[float, float]:
    """feature 下标 → 关键点坐标；0 号点（手腕）恒为原点。"""
    if index == 0:
        return (0.0, 0.0)
    return (f[index * 2 - 2], f[index * 2 - 1])


def _d(f, a: int, b: int) -> float:
    ax, ay = _pt(f, a)
    bx, by = _pt(f, b)
    return math.hypot(ax - bx, ay - by)


def _angle(f, a: int, v: int, b: int) -> float:
    ax, ay = _pt(f, a)
    vx, vy = _pt(f, v)
    bx, by = _pt(f, b)
    u1 = (ax - vx, ay - vy)
    u2 = (bx - vx, by - vy)
    n1 = math.hypot(*u1)
    n2 = math.hypot(*u2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    c = (u1[0] * u2[0] + u1[1] * u2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def feats(f) -> dict[str, float]:
    hl = math.hypot(f[16], f[17]) or 1e-6  # |腕→中指MCP|
    pw = _d(f, 5, 17) or 1e-6  # 掌宽
    palm = _center(f)
    out = {
        "hl": hl,
        "pw": pw,
        "frontal": pw / hl,  # 掌宽/手长：正手≈1.0，侧手≈0.5
        # —— 以掌宽为尺（现行判据用的）——
        "r_i_pw": _d(f, 5, 8) / pw,
        "r_m_pw": _d(f, 9, 12) / pw,
        # —— 以手长为尺（候选）——
        "r_i": _d(f, 5, 8) / hl,
        "r_m": _d(f, 9, 12) / hl,
        "r_r": _d(f, 13, 16) / hl,
        "r_p": _d(f, 17, 20) / hl,
        # 指尖到掌心（含手腕的 5 点质心）：弯曲时指尖回到掌心附近
        "t_i": _dc(f, 8, palm) / hl,
        "t_m": _dc(f, 12, palm) / hl,
        "t_r": _dc(f, 16, palm) / hl,
        "t_p": _dc(f, 20, palm) / hl,
        # 两指尖间距：双指并拢才靠得近，单指/捏合都远
        "gap": _d(f, 8, 12) / hl,
        "gap_pw": _d(f, 8, 12) / pw,
        # 关节角
        "a_i": _angle(f, 5, 6, 8),
        "a_m": _angle(f, 9, 10, 12),
        "a_r": _angle(f, 13, 14, 16),
        "a_p": _angle(f, 17, 18, 20),
    }
    out["ring_vs_i"] = out["r_r"] / max(out["r_i"], 1e-6)
    out["pinky_vs_i"] = out["r_p"] / max(out["r_i"], 1e-6)
    out["d_i_m"] = abs(out["r_i"] - out["r_m"])
    return out


def _center(f):
    ids = (0, 5, 9, 13, 17)
    xs = sum(_pt(f, i)[0] for i in ids) / 5.0
    ys = sum(_pt(f, i)[1] for i in ids) / 5.0
    return (xs, ys)


def _dc(f, idx, p) -> float:
    x, y = _pt(f, idx)
    return math.hypot(x - p[0], y - p[1])


def load() -> dict[str, list[dict[str, float]]]:
    data: dict[str, list[dict[str, float]]] = defaultdict(list)
    with SAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            label = raw.get("label")
            fs = raw.get("features")
            if label in LABELS and isinstance(fs, list) and len(fs) == 40:
                data[label].append(feats([float(v) for v in fs]))
    return data


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def rule_current(x: dict[str, float]) -> bool:
    """现行判据（以掌宽为尺的绝对 1.3）。"""
    return (
        x["r_i_pw"] >= 1.3
        and x["r_m_pw"] >= 1.3
        and x["a_m"] >= 135.0
        and abs(x["r_i_pw"] - x["r_m_pw"]) <= 1.2
        and x["r_r"] <= x["r_i"] * 0.75
        and x["r_p"] <= x["r_i"] * 0.75
    )


def main() -> None:
    data = load()
    total = sum(len(v) for v in data.values())
    print(f"样本总数 {total}")
    for label in LABELS:
        print(f"  {label:6s} {len(data[label]):5d}")

    print("\n=== 关键特征分位数（p5 / p50 / p95）===")
    keys = ["frontal", "r_i_pw", "r_i", "r_m", "r_r", "r_p", "t_i", "t_m", "t_r", "t_p", "gap", "a_m", "a_r", "a_p"]
    print(f"{'特征':10s}" + "".join(f"{lb:>22s}" for lb in LABELS))
    for key in keys:
        row = f"{key:10s}"
        for label in LABELS:
            vals = [x[key] for x in data[label]]
            row += f"{pct(vals, .05):7.2f}/{pct(vals, .5):5.2f}/{pct(vals, .95):5.2f}"
        print(row)

    print("\n=== 现行判据 ===")
    _score(data, rule_current, "current(reach>=1.3/palm)")

    print("\n=== 网格搜索：以手长为尺 ===")
    best = []
    for t_i in (1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3):
        for t_m in (0.9, 1.0, 1.05, 1.1, 1.15, 1.2):
            for t_r in (0.9, 1.0, 1.05, 1.1, 1.15):
                for gap in (1.1, 1.3, 1.5, 1.7, 1.9, 2.1):
                    rule = _make(t_i, t_m, t_r, gap)
                    rec, fp = _score(data, rule, None)
                    best.append((rec, -fp, t_i, t_m, t_r, gap))
    best.sort(reverse=True)
    print("\ntop 15（按召回排序）:")
    for rec, negfp, t_i, t_m, t_r, gap in best[:15]:
        print(f"  r_i>={t_i} r_m>={t_m} r_r<={t_r} gap<={gap}  → 召回 {rec*100:5.1f}%  误报 {-negfp*100:5.2f}%")


def _make(t_i, t_m, t_r, gap):
    def rule(x: dict[str, float]) -> bool:
        return (
            x["r_i"] >= t_i
            and x["r_m"] >= t_m
            and x["gap"] <= gap
            and x["r_r"] <= t_r
            and x["r_p"] <= t_r
        )

    return rule


def _score(data, rule, name):
    hits = 0
    tot = 0
    fp = 0
    ntot = 0
    per = {}
    for label in LABELS:
        n = sum(1 for x in data[label] if rule(x))
        per[label] = n / max(len(data[label]), 1)
        if label == "swipe":
            hits, tot = n, len(data[label])
        else:
            fp += n
            ntot += len(data[label])
    rec = hits / max(tot, 1)
    fpr = fp / max(ntot, 1)
    if name:
        print(f"{name}: 召回 {rec*100:.1f}%  非双指误报 {fpr*100:.2f}%")
        print("  各类通过率: " + "  ".join(f"{lb} {per[lb]*100:.1f}%" for lb in LABELS))
    return rec, fpr


if __name__ == "__main__":
    main()
