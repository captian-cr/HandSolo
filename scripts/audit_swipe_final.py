"""双指滑动的**最终**离线验收（判据直接 import 自 ``handmouse.gestures``，不会漂移）。

用途：改完判据以后，用用户自己的真实样本 + 按 ``captured_ns`` 还原的原始帧序，
量化"他在实机上到底能不能用"，而不是只看单测过没过。

三个指标：
  * **可用度**：在一次真实双指导下，处于 SWIPE 状态的帧占比（越高越好）
  * **误报**：在一次真实"只伸食指/停住/捏合/张开"下，被误判成 SWIPE 的帧占比
  * **进入次数**：一次连续动作里被切成几段（>1 说明中途掉出去过，光标会瞬移）

运行：python scripts/audit_swipe_final.py
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
    _is_two_finger_hold,
    _palm_width,
    _two_finger_diag,
    _two_finger_hit,
)

SAMPLES = ROOT / "gesture_data" / "samples.jsonl"
LABELS = ("swipe", "move", "stop", "pinch", "open", "fist")
#: 与 ``gestures.py`` 里的 ``_SWIPE_CONFIRM_MS`` 保持一致。
CONFIRM_MS = 150.0


def to_points(features: list[float]) -> tuple[Landmark, ...] | None:
    """40 维特征还原成 21 个关键点（z 丢弃——所有判据都是二维的）。"""

    if len(features) != 40:
        return None
    points = [Landmark(x=0.0, y=0.0, z=0.0)]
    for index in range(20):
        points.append(Landmark(x=float(features[index * 2]), y=float(features[index * 2 + 1]), z=0.0))
    return tuple(points)


def old_rule(points: tuple[Landmark, ...]) -> bool:
    """2026-10-22 第五次修订前的判据（保留作对照基线，冻结不再改）。"""

    width = _palm_width(points)

    def dist2d(a: int, b: int) -> float:
        return math.hypot(points[a].x - points[b].x, points[a].y - points[b].y)

    def angle(a: int, v: int, b: int) -> float:
        u1 = (points[a].x - points[v].x, points[a].y - points[v].y)
        u2 = (points[b].x - points[v].x, points[b].y - points[v].y)
        n1 = math.hypot(*u1) or 1e-6
        n2 = math.hypot(*u2) or 1e-6
        c = (u1[0] * u2[0] + u1[1] * u2[1]) / (n1 * n2)
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    reach_index = dist2d(5, 8) / width
    reach_middle = dist2d(9, 12) / width
    return (
        reach_index >= 1.3
        and reach_middle >= 1.3
        and angle(9, 10, 12) >= 135.0
        and abs(reach_index - reach_middle) <= 1.2
        and dist2d(13, 16) / width <= reach_index * 0.75
        and dist2d(17, 20) / width <= reach_index * 0.75
    )


def load() -> dict[tuple[str, str], list[tuple[int, list[float]]]]:
    seq: dict[tuple[str, str], list[tuple[int, list[float]]]] = defaultdict(list)
    with SAMPLES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            label = raw.get("label")
            feats = raw.get("features")
            if label in LABELS and isinstance(feats, list) and len(feats) == 40:
                seq[(label, raw.get("source", "?"))].append((int(raw.get("captured_ns", 0)), feats))
    for key in seq:
        seq[key].sort(key=lambda item: item[0])
    return seq


def simulate(frames, enter, hold, hold_ms, confirm_ms=CONFIRM_MS):
    """复刻 pipeline 的时序状态机，返回 (active 帧数, 总帧数, 进入次数)。"""

    hold_ns = hold_ms * 1_000_000
    confirm_ns = confirm_ms * 1_000_000
    active = False
    last_seen = None
    hit_prev = None
    hit_cur = None
    count = 0
    entries = 0
    was_active = False
    for ts, points in frames:
        diag = _two_finger_diag(points)
        # ``enter`` 收两个参数：主判据只看 ``diag``，逐指并联判据还要原始点。
        two_finger = enter(points, diag)
        if two_finger:
            hit_prev = hit_cur
            hit_cur = ts
        confirmed = hit_prev is not None and hit_cur is not None and hit_cur - hit_prev <= confirm_ns
        if two_finger and (confirmed or active):
            last_seen = ts
            active = True
        elif active and last_seen is not None and ts - last_seen <= hold_ns and hold(diag):
            active = True
        else:
            active = False
            last_seen = None
        if active:
            count += 1
            if count == 1 or not was_active:
                entries += 1
        was_active = active
    return count, len(frames), entries


def run(seq, enter, hold, hold_ms):
    per = {}
    entries = {}
    for label in LABELS:
        tot = hit = ent = 0
        for (lb, _src), frames in seq.items():
            if lb != label or len(frames) < 5:
                continue
            pts = [(ts, to_points(feats)) for ts, feats in frames]
            pts = [(ts, p) for ts, p in pts if p is not None]
            h, n, e = simulate(pts, enter, hold, hold_ms)
            tot += n
            hit += h
            ent += e
        per[label] = hit / tot if tot else 0.0
        entries[label] = ent
    return per, entries


def main():
    seq = load()
    print(f"样本会话 {len(seq)} 个，总样本 {sum(len(v) for v in seq.values())}")

    def old_only(frames):
        """旧判据：单帧成立即算 SWIPE，无确认、无持有（对它最有利的口径）。"""

        tot = hit = ent = 0
        prev = False
        for _ts, feats in frames:
            pts = to_points(feats)
            if pts is None:
                continue
            tot += 1
            active = old_rule(pts)
            if active:
                hit += 1
            if active and not prev:
                ent += 1
            prev = active
        return hit, tot, ent

    rows = []
    per = {}
    ent = {}
    for label in LABELS:
        tot = hit = e_sum = 0
        for (lb, _src), frames in seq.items():
            if lb != label or len(frames) < 5:
                continue
            h, n, e = old_only(frames)
            hit += h
            tot += n
            e_sum += e
        per[label] = hit / tot if tot else 0.0
        ent[label] = e_sum
    rows.append(("旧判据 ÷掌宽 reach>=1.3（无确认/无持有）", per, ent))

    for hold_ms in (250, 350, 400, 450, 600):
        per, ent = run(seq, _two_finger_hit, _is_two_finger_hold, hold_ms)
        rows.append((f"新判据 确认150ms 持有{hold_ms}ms", per, ent))

    for name, per, ent in rows:
        print(f"\n=== {name} ===")
        print("  帧占比: " + "  ".join(f"{lb} {per[lb]*100:5.1f}%" for lb in LABELS))
        print("  进入次数: " + "  ".join(f"{lb} {ent[lb]}" for lb in LABELS))

    # ---- 逐会话分布：别让平均数掩盖"某几次录制依然完全用不了" ----
    from handmouse.config import AppConfig

    hold_ms = AppConfig().swipe_hold_ms
    print(f"\n=== 逐会话分布（swipe，持有 {hold_ms}ms = 当前配置）===")
    print(f"  {'会话':<28} {'帧数':>5} {'可用度':>7} {'进入':>4}")
    shares = []
    for (lb, src), frames in sorted(seq.items(), key=lambda kv: kv[0][1]):
        if lb != "swipe" or len(frames) < 5:
            continue
        pts = [(ts, to_points(f)) for ts, f in frames]
        pts = [(ts, p) for ts, p in pts if p is not None]
        h, n, e = simulate(pts, _two_finger_hit, _is_two_finger_hold, hold_ms)
        share = h / n if n else 0.0
        shares.append((share, src, n, e))
        print(f"  {src[:28]:<28} {n:>5} {share*100:>6.1f}% {e:>4}")
    if shares:
        shares.sort()
        worst = shares[0]
        best = shares[-1]
        med = shares[len(shares) // 2][0]
        print(
            f"\n  最差 {worst[0]*100:.1f}% ({worst[1]})  "
            f"中位 {med*100:.1f}%  最好 {best[0]*100:.1f}%"
        )
        bad = [s for s in shares if s[0] < 0.5]
        print(f"  可用度 <50% 的会话：{len(bad)}/{len(shares)}")


if __name__ == "__main__":
    main()
