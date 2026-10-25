"""第二版审计：完全标度不变的「指间相对」判据。

第一版结论（已跑过）：把尺子从「掌宽」换成「手长」并不能救回召回——
swipe 类内 ``r_i`` 的 p5~p95 是 0.41~1.51，跨度 3.7 倍。
原因：单目二维投影下，手指长度本身随手掌朝向剧烈变化，
**任何"绝对长度 ÷ 某个基准"的阈值都只能覆盖一部分姿态**。

所以换判别维度：不比"多长"，只比"谁比谁长"。
以四指指尖到掌心距离 ``t_i/t_m/t_r/t_p`` 为基础，取本帧最大值做基准：

    s = max(t_i, t_m, t_r, t_p)

* 只伸食指：``t_m/s ≈ 0.33``
* 五指张开：四指都 ≈ 0.85~1.0
* 双指：``t_i/s ≈ 1``、``t_m/s ≈ 1``，而 ``t_r/s``、``t_p/s`` 明显小

分子分母同乘同除一个投影系数，**姿态压不扁这个比值**。

运行：python scripts/audit_swipe_rel.py
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


def _pt(f, index):
    if index == 0:
        return (0.0, 0.0)
    return (f[index * 2 - 2], f[index * 2 - 1])


def _angle(f, a, v, b):
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


def feats(f):
    ids = (0, 5, 9, 13, 17)
    cx = sum(_pt(f, i)[0] for i in ids) / 5.0
    cy = sum(_pt(f, i)[1] for i in ids) / 5.0
    t = {}
    for name, idx in (("i", 8), ("m", 12), ("r", 16), ("p", 20)):
        x, y = _pt(f, idx)
        t[name] = math.hypot(x - cx, y - cy)
    s = max(max(t.values()), 1e-6)
    hl = math.hypot(f[16], f[17]) or 1e-6
    return {
        "s": s / hl,  # 本帧"最长指尖距离 ÷ 手长"，反映手离镜头/朝向的整体缩放
        "ti": t["i"] / s,
        "tm": t["m"] / s,
        "tr": t["r"] / s,
        "tp": t["p"] / s,
        "a_i": _angle(f, 5, 6, 8),
        "a_m": _angle(f, 9, 10, 12),
        "a_r": _angle(f, 13, 14, 16),
        "a_p": _angle(f, 17, 18, 20),
        "gap": math.hypot(_pt(f, 8)[0] - _pt(f, 12)[0], _pt(f, 8)[1] - _pt(f, 12)[1]) / hl,
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


def pct(vals, q):
    if not vals:
        return float("nan")
    o = sorted(vals)
    k = (len(o) - 1) * q
    lo = int(math.floor(k))
    hi = min(lo + 1, len(o) - 1)
    return o[lo] + (o[hi] - o[lo]) * (k - lo)


def main():
    data = load()
    print(f"样本总数 {sum(len(v) for v in data.values())}")
    keys = ["ti", "tm", "tr", "tp", "a_i", "a_m", "a_r", "a_p", "gap"]
    print("\n=== 相对特征分位数（p5 / p25 / p50 / p75 / p95）===")
    print(f"{'特征':6s}" + "".join(f"{lb:>30s}" for lb in LABELS))
    for key in keys:
        row = f"{key:6s}"
        for label in LABELS:
            v = [x[key] for x in data[label]]
            row += f"{pct(v,.05):6.2f}{pct(v,.25):6.2f}{pct(v,.5):6.2f}{pct(v,.75):6.2f}{pct(v,.95):6.2f}"
        print(row)

    print("\n=== 网格搜索（ti/tm 高，tr/tp 低；可选加中指关节角）===")
    results = []
    for ti in (0.60, 0.70, 0.75, 0.80, 0.85, 0.90):
        for tm in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            for tro in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
                for use_ang in (False, True):
                    rule = _make(ti, tm, tro, use_ang)
                    rec, fpr = _score(data, rule, None)
                    results.append((rec, -fpr, ti, tm, tro, use_ang))
    results.sort(reverse=True)
    for rec, nf, ti, tm, tro, ang in results[:20]:
        print(f"  ti>={ti} tm>={tm} tr,tp<={tro} ang={int(ang)}  → 召回 {rec*100:5.1f}%  误报 {-nf*100:5.2f}%")

    print("\n=== 误报/召回平衡最好的（要求非双指误报 <= 1.5%）===")
    ok = [r for r in results if -r[1] <= 0.015]
    ok.sort(reverse=True)
    for rec, nf, ti, tm, tro, ang in ok[:10]:
        print(f"  ti>={ti} tm>={tm} tr,tp<={tro} ang={int(ang)}  → 召回 {rec*100:5.1f}%  误报 {-nf*100:2.2f}%")
    if ok:
        rec, nf, ti, tm, tro, ang = ok[0]
        _score(data, _make(ti, tm, tro, ang), "BEST")


def _make(ti, tm, tro, use_ang):
    def rule(x):
        if not (x["ti"] >= ti and x["tm"] >= tm and x["tr"] <= tro and x["tp"] <= tro):
            return False
        if use_ang and x["a_m"] < 135.0:
            return False
        return True

    return rule


def _score(data, rule, name):
    per = {}
    for label in LABELS:
        per[label] = sum(1 for x in data[label] if rule(x)) / max(len(data[label]), 1)
    rec = per["swipe"]
    nf = sum(len(data[lb]) for lb in LABELS if lb != "swipe") or 1
    fpr = sum(per[lb] * len(data[lb]) for lb in LABELS if lb != "swipe") / nf
    if name:
        print(f"{name}: 召回 {rec*100:.1f}%  非双指误报 {fpr*100:.2f}%")
        print("  各类通过率: " + "  ".join(f"{lb} {per[lb]*100:.1f}%" for lb in LABELS))
    return rec, fpr


if __name__ == "__main__":
    main()
