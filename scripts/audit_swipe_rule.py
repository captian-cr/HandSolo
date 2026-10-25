"""第三版审计：不依赖任何"长度基准"的纯指间比值 + 贪心规则搜索。

前两版结论：
  * 以掌宽为尺（现行）召回 36.3%
  * 以手长为尺 召回 24.8%（更差——swipe 类内 r_i 跨 3.7 倍）
  * 以"本帧最长指尖距离"为尺 召回 65.2% / 误报 1.32%

本版彻底不用外部基准：所有特征都是**同一帧内两个距离相除**，
分子分母共享同一个投影缩放系数，姿态变化直接约掉。
重点候选：
  * ``rm_ri = d(9,12) / d(5,8)``    中指伸展 / 食指伸展 —— 两根手指"一样直"才接近 1
  * ``rr_ri = d(13,16) / d(5,8)``   无名指 / 食指
  * ``vcos``                       食指、中指方向向量夹角余弦（并拢且同向才接近 1）
  * ``par`` = d(8,12) / d(5,9)      两指尖距 / 两指根距 —— 并排平行时约 1~2，中指蜷起时很大

再做贪心规则搜索（每轮挑一条"单特征阈值"把 FP 压下去 / 把召回拉上来），
输出可读的规则链，用于写进 gestures.py。

运行：python scripts/audit_swipe_rule.py
"""

from __future__ import annotations

import itertools
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


def _pt(f, i):
    return (0.0, 0.0) if i == 0 else (f[i * 2 - 2], f[i * 2 - 1])


def _d(f, a, b):
    ax, ay = _pt(f, a)
    bx, by = _pt(f, b)
    return math.hypot(ax - bx, ay - by) or 1e-6


def _ang(f, a, v, b):
    ax, ay = _pt(f, a)
    vx, vy = _pt(f, v)
    bx, by = _pt(f, b)
    u1 = (ax - vx, ay - vy)
    u2 = (bx - vx, by - vy)
    n1 = math.hypot(*u1) or 1e-6
    n2 = math.hypot(*u2) or 1e-6
    c = (u1[0] * u2[0] + u1[1] * u2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def _cosdir(f, a, b, c, d):
    """向量 ab 与 cd 的夹角余弦。"""
    ax, ay = _pt(f, a)
    bx, by = _pt(f, b)
    cx, cy = _pt(f, c)
    dx, dy = _pt(f, d)
    u = (bx - ax, by - ay)
    v = (dx - cx, dy - cy)
    nu = math.hypot(*u) or 1e-6
    nv = math.hypot(*v) or 1e-6
    return (u[0] * v[0] + u[1] * v[1]) / (nu * nv)


def feats(f):
    d58, d912, d1316, d1720 = _d(f, 5, 8), _d(f, 9, 12), _d(f, 13, 16), _d(f, 17, 20)
    ids = (0, 5, 9, 13, 17)
    cx = sum(_pt(f, i)[0] for i in ids) / 5.0
    cy = sum(_pt(f, i)[1] for i in ids) / 5.0
    t = {k: math.hypot(_pt(f, i)[0] - cx, _pt(f, i)[1] - cy) for k, i in (("i", 8), ("m", 12), ("r", 16), ("p", 20))}
    s = max(max(t.values()), 1e-6)
    return {
        # —— 纯指间比值 ——
        "rm_ri": d912 / d58,
        "rr_ri": d1316 / d58,
        "rp_ri": d1720 / d58,
        "rr_rm": d1316 / d912,
        "rp_rm": d1720 / d912,
        "par": _d(f, 8, 12) / _d(f, 5, 9),
        "vcos": _cosdir(f, 5, 8, 9, 12),
        "vcos_rm": _cosdir(f, 9, 12, 13, 16),
        # —— 前两版验证过有信息的 ——
        "ti": t["i"] / s,
        "tm": t["m"] / s,
        "tr": t["r"] / s,
        "tp": t["p"] / s,
        "a_m": _ang(f, 9, 10, 12),
        "a_r": _ang(f, 13, 14, 16),
        "a_i": _ang(f, 5, 6, 8),
        "a_p": _ang(f, 17, 18, 20),
    }


def load():
    data = defaultdict(list)
    with SAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("label") in LABELS and isinstance(raw.get("features"), list) and len(raw["features"]) == 40:
                data[raw["label"]].append(feats([float(v) for v in raw["features"]]))
    return data


def pct(v, q):
    if not v:
        return float("nan")
    o = sorted(v)
    k = (len(o) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(o) - 1)
    return o[lo] + (o[hi] - o[lo]) * (k - lo)


def main():
    data = load()
    keys = list(data["swipe"][0].keys())
    print(f"样本 {sum(len(v) for v in data.values())}")
    print("\n=== 分位数（p5/p25/p50/p75/p95）===")
    for key in keys:
        row = f"{key:8s}"
        for label in LABELS:
            v = [x[key] for x in data[label]]
            row += f" {label[:4]:>4s} " + " ".join(f"{pct(v,q):6.2f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95))
        print(row)

    # ---- 贪心规则搜索：从"全通过"开始，逐轮加条件 ----
    print("\n=== 贪心规则（目标：召回 >= 90%，同时误报最小）===")
    for target in (0.90, 0.85, 0.80):
        rule = _greedy(data, keys, target)
        if rule:
            rec, fp = _eval(data, rule, verbose=True)
            print(f"  目标 {target*100:.0f}% → 召回 {rec*100:.1f}%  误报 {fp*100:.2f}%  规则: {rule}")


def _cond(x, key, sign, thr):
    return x[key] >= thr if sign == ">=" else x[key] <= thr


def _eval(data, rule, verbose=False):
    per = {}
    for lb in LABELS:
        per[lb] = sum(1 for x in data[lb] if all(_cond(x, k, s, t) for k, s, t in rule)) / max(len(data[lb]), 1)
    rec = per["swipe"]
    n = sum(len(data[lb]) for lb in LABELS if lb != "swipe") or 1
    fp = sum(per[lb] * len(data[lb]) for lb in LABELS if lb != "swipe") / n
    if verbose:
        print("    各类: " + "  ".join(f"{lb} {per[lb]*100:5.1f}%" for lb in LABELS))
    return rec, fp


def _greedy(data, keys, target):
    """每轮在所有 (key, sign, 分位阈值) 中挑：在保持召回 >= target 的前提下误报最低。"""
    cands = []
    for key in keys:
        vals = sorted(x[key] for lb in LABELS for x in data[lb])
        for q in (0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95):
            thr = vals[min(int(q * (len(vals) - 1)), len(vals) - 1)]
            cands.append((key, ">=", thr))
            cands.append((key, "<=", thr))

    rule = []
    best_rec, best_fp = 1.0, 1.0
    for _ in range(6):
        pick = None
        for cand in cands:
            trial = rule + [cand]
            rec, fp = _eval(data, trial)
            if rec < target:
                continue
            if pick is None or fp < pick[2] - 1e-9:
                pick = (cand, rec, fp)
        if pick is None:
            return None
        rule.append(pick[0])
        best_rec, best_fp = pick[1], pick[2]
        print(f"    + {pick[0][0]} {pick[0][1]} {pick[0][2]:.3f}  → 召回 {best_rec*100:.1f}%  误报 {best_fp*100:.2f}%")
        if best_fp <= 0.005:
            break
    return rule


if __name__ == "__main__":
    main()
