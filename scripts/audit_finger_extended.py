"""「手指伸出来了吗」的**逐指**判据离线验收。

设计动机（2026-10-22 用户提出）：
    不要再"拿不同手指互相比长度"。那种判据里必然有一个**跨手指**的比例，
    换个手腕角度各指的投影缩放就不一样了，阈值永远调不完。
    改成**每根手指跟自己比**：这根手指的骨链是直的、而且朝外伸出去。

两个量都是**单指内部归一化**，不含任何跨手指比较：

* ``s``（伸直度）= 〚MCP→TIP〛 ÷ (〚MCP→PIP〛+〚PIP→DIP〛+〚DIP→TIP〛)
  只量 PIP/DIP 有没有弯。伸直 ≈ 1.0。对均匀投影缩放**严格不变**。
* ``a``（指根角，度）= 掌指关节排中心→MCP 的方向 与 MCP→PIP 的夹角。
  补上 ``s`` 抓不到的"MCP 关节把整根手指折向掌心"——那种情况
  三节自身是直的（``s``≈1）却趴在掌心里，投影上正是"只伸食指"的形状。

一个指头「伸出来了」= ``s >= S_MIN`` 且 ``a >= A_MIN``。
双指滑动的完整判据 = 食指伸出 且 中指伸出 且（无名指 或 小指 没伸出）。

运行：python scripts/audit_finger_extended.py
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
from handmouse.gestures import _is_two_finger_hold, _two_finger_diag  # noqa: E402

SAMPLES = ROOT / "gesture_data" / "samples.jsonl"
LABELS = ("swipe", "move", "stop", "pinch", "open", "fist")
FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
KNUCKLES = (5, 9, 13, 17)
CONFIRM_MS = 150.0


def to_points(features: list[float]) -> tuple[Landmark, ...] | None:
    if len(features) != 40:
        return None
    points = [Landmark(0.0, 0.0, 0.0)]
    for index in range(20):
        points.append(Landmark(float(features[index * 2]), float(features[index * 2 + 1]), 0.0))
    return tuple(points)


def dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def angle(a: Landmark, vertex: Landmark, b: Landmark) -> float:
    u1 = (a.x - vertex.x, a.y - vertex.y)
    u2 = (b.x - vertex.x, b.y - vertex.y)
    n1 = math.hypot(*u1) or 1e-9
    n2 = math.hypot(*u2) or 1e-9
    cosine = (u1[0] * u2[0] + u1[1] * u2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def finger_feats(points: tuple[Landmark, ...]) -> dict[str, tuple[float, float]]:
    """返回 ``{finger: (s, a)}``，两个量都只用到这一根手指自己的点。"""

    cx = sum(points[i].x for i in KNUCKLES) / len(KNUCKLES)
    cy = sum(points[i].y for i in KNUCKLES) / len(KNUCKLES)
    knuckle_center = Landmark(cx, cy, 0.0)

    out: dict[str, tuple[float, float]] = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        path = dist(points[mcp], points[pip]) + dist(points[pip], points[dip]) + dist(points[dip], points[tip])
        straight = dist(points[mcp], points[tip]) / path if path > 1e-6 else 0.0
        root = angle(knuckle_center, points[mcp], points[pip])
        out[name] = (straight, root)
    return out


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
            points = to_points(raw.get("features") or [])
            if points is None:
                continue
            seq[(label, raw.get("source", "?"))].append(
                (
                    int(raw.get("captured_ns", 0)),
                    finger_feats(points),
                    _two_finger_diag(points),
                )
            )
    for key in seq:
        seq[key].sort(key=lambda item: item[0])
    return seq


def simulate(frames, enter, hold_ms: int = 450, confirm_ms: float = CONFIRM_MS):
    hold_ns = hold_ms * 1_000_000
    confirm_ns = confirm_ms * 1_000_000
    active = False
    last_seen = None
    hit_prev = None
    hit_cur = None
    count = 0
    entries = 0
    was_active = False
    for ts, feats, diag in frames:
        hit = enter(feats, diag)
        if hit:
            hit_prev = hit_cur
            hit_cur = ts
        confirmed = hit_prev is not None and hit_cur is not None and hit_cur - hit_prev <= confirm_ns
        if hit and (confirmed or active):
            last_seen = ts
            active = True
        elif active and last_seen is not None and ts - last_seen <= hold_ns and _is_two_finger_hold(diag):
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


def evaluate(seq, enter):
    per = {}
    entries = {}
    for label in LABELS:
        tot = hit = ent = 0
        for (lb, _src), frames in seq.items():
            if lb != label or len(frames) < 5:
                continue
            h, n, e = simulate(frames, enter)
            tot += n
            hit += h
            ent += e
        per[label] = hit / tot * 100 if tot else 0.0
        entries[label] = ent
    return per, entries


def main():
    seq = load()
    print(f"样本会话 {len(seq)} 个，总样本 {sum(len(v) for v in seq.values())}")

    def make(s_min: float, a_min: float, curl: float):
        def enter(feats, _diag):
            ix_s, ix_a = feats["index"]
            md_s, md_a = feats["middle"]
            rg_s, _ = feats["ring"]
            pk_s, _ = feats["pinky"]
            return (
                ix_s >= s_min
                and ix_a >= a_min
                and md_s >= s_min
                and md_a >= a_min
                and (rg_s <= curl or pk_s <= curl)
            )

        return enter

    print(f"\n{'S_MIN':>6}{'A_MIN':>7}{'CURL':>6}{'swipe':>8}" + "".join(f"{l:>8}" for l in LABELS[1:]) + f"{'入场':>6}")
    best = None
    for s_min in (0.85, 0.88, 0.90, 0.93):
        for a_min in (55, 65, 75, 85, 95):
            for curl in (0.90, 0.95):
                per, ent = evaluate(seq, make(s_min, a_min, curl))
                others = max(per[lb] for lb in LABELS[1:])
                flag = " <<<" if others < 1.0 else ""
                if others < 1.0 and (best is None or per["swipe"] > best[0]):
                    best = (per["swipe"], s_min, a_min, curl)
                print(
                    f"{s_min:>6.2f}{a_min:>7}{curl:>6.2f}{per['swipe']:>7.1f}%"
                    + "".join(f"{per[l]:7.1f}%" for l in LABELS[1:])
                    + f"{ent['swipe']:>6}{flag}"
                )

    print("\n=== 当前线上判据（指尖到掌心 ÷ 本帧最长）对照 ===")
    from handmouse.gestures import _is_two_finger

    per, ent = evaluate(seq, lambda _f, diag: _is_two_finger(diag))
    print("  帧占比: " + "  ".join(f"{lb} {per[lb]:5.1f}%" for lb in LABELS))
    print("  进入次数: " + "  ".join(f"{lb} {ent[lb]}" for lb in LABELS))

    print("\n=== 并联：线上判据 ∪ 逐指伸直（两把尺子各抓各的盲区）===")
    best_union = None
    for s_min in (0.88, 0.90, 0.93):
        for a_min in (55, 75, 95):
            for curl in (0.90, 0.95):
                fingers = make(s_min, a_min, curl)

                def union(feats, diag, fingers=fingers):
                    return _is_two_finger(diag) or fingers(feats, diag)

                per, ent = evaluate(seq, union)
                others = max(per[lb] for lb in LABELS[1:])
                flag = " <<<" if others < 1.0 else ""
                if others < 1.0 and (best_union is None or per["swipe"] > best_union[0]):
                    best_union = (per["swipe"], s_min, a_min, curl)
                print(
                    f"   S={s_min:.2f} A={a_min:>3} CURL={curl:.2f}"
                    f"{per['swipe']:>7.1f}%"
                    + "".join(f"{per[l]:7.1f}%" for l in LABELS[1:])
                    + flag
                )

    if best:
        print(f"\n逐指判据零误报下最优：S_MIN={best[1]} A_MIN={best[2]} CURL={best[3]} → 双指可用度 {best[0]:.1f}%")
    if best_union:
        print(
            f"并联后零误报下最优：S_MIN={best_union[1]} A_MIN={best_union[2]} "
            f"CURL={best_union[3]} → 双指可用度 {best_union[0]:.1f}%"
        )


if __name__ == "__main__":
    main()
