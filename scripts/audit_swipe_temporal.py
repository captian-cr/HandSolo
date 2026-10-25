"""第四版审计：按真实帧序做**时序仿真**，看"进入判据 + 持有窗口"合起来的实际可用度。

前三版只统计"单帧判据命中率"，但用户感觉到的是**连续可用时长**：
只要判据抖动一两帧就掉出 SWIPE，光标就会瞬移回食指尖（"乱飞"）。
真实的落点指标是：在一次真实的双指导下，系统有多少比例的帧处于 SWIPE 状态。

样本里带 ``captured_ns``，可以按 ``source`` 还原原始帧序，所以这里能做真时序仿真：
  * 进入：单帧判据成立
  * 持有：判据失败后，只要距上次成立不超过 hold_ms 就继续算 SWIPE
  * 输出：swipe 会话里"处于 SWIPE 的帧占比"（可用度）、move 会话里的误报时长占比

运行：python scripts/audit_swipe_temporal.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_swipe_rel import feats  # noqa: E402

SAMPLES = ROOT / "gesture_data" / "samples.jsonl"
LABELS = ("move", "stop", "pinch", "open", "fist", "swipe")


def load_sequences():
    seq = defaultdict(list)
    with SAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("label") in LABELS and isinstance(raw.get("features"), list) and len(raw["features"]) == 40:
                seq[(raw.get("label"), raw.get("source", "?"))].append(
                    (int(raw.get("captured_ns", 0)), feats([float(v) for v in raw["features"]]))
                )
    for key in seq:
        seq[key].sort(key=lambda item: item[0])
    return seq


def rule_rel(x):
    """候选判据（标度不变）：食指、中指都相对最长为 0.7，无名指小指 <=0.6。"""
    return x["ti"] >= 0.7 and x["tm"] >= 0.7 and x["tr"] <= 0.6 and x["tp"] <= 0.6


def rule_palm_width(x):
    """现行判据：reach>=1.3 ÷ 掌宽 + 中指关节角 + 相对收拢。"""
    return (
        x["r_i_pw"] >= 1.3
        and x["r_m_pw"] >= 1.3
        and x["a_m"] >= 135.0
        and abs(x["r_i_pw"] - x["r_m_pw"]) <= 1.2
        and x["rr_ri"] <= 0.75
        and x["rp_ri"] <= 0.75
    )


def simulate(seq, rule, hold_ms):
    """返回 (处于 SWIPE 的帧数, 总帧数, 进入次数)。"""
    held = 0
    last_hit = None
    entries = 0
    prev = False
    hold_ns = hold_ms * 1_000_000
    for ts, x in seq:
        hit = rule(x)
        if hit:
            last_hit = ts
        active = bool(hit) or (last_hit is not None and ts - last_hit <= hold_ns)
        if active:
            held += 1
            if not prev:
                entries += 1
        prev = active
    return held, len(seq), entries


def main():
    seqs = load_sequences()

    # 现行口径的特征名不一样，单独补算
    def load_pw():
        out = defaultdict(list)
        with SAMPLES.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if raw.get("label") not in LABELS or not isinstance(raw.get("features"), list) or len(raw["features"]) != 40:
                    continue
                f = [float(v) for v in raw["features"]]
                pw = math.hypot(f[8] - f[32], f[9] - f[33]) or 1e-6

                def d(a, b):
                    pa = (0.0, 0.0) if a == 0 else (f[a * 2 - 2], f[a * 2 - 1])
                    pb = (0.0, 0.0) if b == 0 else (f[b * 2 - 2], f[b * 2 - 1])
                    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])

                def pt(i):
                    return (0.0, 0.0) if i == 0 else (f[i * 2 - 2], f[i * 2 - 1])

                def ang(a, v, b):
                    ax, ay = pt(a)
                    vx, vy = pt(v)
                    bx, by = pt(b)
                    u1 = (ax - vx, ay - vy)
                    u2 = (bx - vx, by - vy)
                    n1 = math.hypot(*u1) or 1e-6
                    n2 = math.hypot(*u2) or 1e-6
                    c = (u1[0] * u2[0] + u1[1] * u2[1]) / (n1 * n2)
                    return math.degrees(math.acos(max(-1.0, min(1.0, c))))

                m = {
                    "r_i_pw": d(5, 8) / pw,
                    "r_m_pw": d(9, 12) / pw,
                    "rr_ri": d(13, 16) / d(5, 8),
                    "rp_ri": d(17, 20) / d(5, 8),
                    "a_m": ang(9, 10, 12),
                }
                out[(raw["label"], raw.get("source", "?"))].append((int(raw.get("captured_ns", 0)), m))
        for k in out:
            out[k].sort(key=lambda i: i[0])
        return out

    pw_seqs = load_pw()

    for name, getter, rule in (
        ("现行 (reach>=1.3 ÷ 掌宽)", pw_seqs, rule_palm_width),
        ("候选 (标度不变相对判据)", seqs, rule_rel),
    ):
        print(f"\n=== {name} ===")
        for hold in (0, 150, 250, 400, 600):
            print(f"  hold={hold:4d}ms")
            for label in ("swipe", "move", "stop", "pinch"):
                tot_f = tot_n = 0
                entries = 0
                for (lb, src), s in getter.items():
                    if lb != label or len(s) < 5:
                        continue
                    h, n, e = simulate(s, rule, hold)
                    tot_f += h
                    tot_n += n
                    entries += e
                if tot_n:
                    print(f"    {label:6s} 处于 SWIPE 帧占比 {tot_f/tot_n*100:5.1f}%   进入次数 {entries:3d} / {tot_n:5d} 帧")


if __name__ == "__main__":
    main()
