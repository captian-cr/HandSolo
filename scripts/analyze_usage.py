"""分析日常使用记录，找出「哪里反复试都没成」。

配合 `handmouse.usage_log`：在 `config.json` 里打开

    "usage_log_enabled": true

正常用一段时间（越长越准），然后运行::

    cd "<项目根目录>"
    python -m scripts.analyze_usage

默认分析 ``logs/`` 里最新的那个 ``usage-*.jsonl``，也可以传路径。

输出四块
--------
1. **反复点击热点** —— 同一小片区域里点了 3 次以上。正常点一次就够，
   反复点说明那一下没被系统接住（点击丢了 / 落点偏了）。
2. **长时间逗留热点** —— 光标在同一小片区域磨了很久（来回微调）。
   这是"瞄不准"的直接证据。
3. **捏了却没点** —— 捏合比已经低到 engage 以下持续若干帧，但整段没产生点击。
   这是"捏合没被识别"的证据，也是最影响手感的一类。
4. **覆盖率** —— 屏幕上哪些格子从没去过（可能是够不到，也可能是压根没用到）。

同时把热图存成 PNG（`logs/usage-heatmap-*.png`），一眼看分布。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"

GRID = 40          # 热图网格：40 x 40
CLICK_RADIUS = 0.035   # 点击聚类的归一化半径（约屏幕 3.5%）
CLICK_MIN = 3          # 同簇至少几次点击才算"反复"
DWELL_MIN_SECONDS = 1.2  # 同一格累计停留超过这么久算"磨"


def _latest_log() -> Path | None:
    files = sorted(LOG_DIR.glob("usage-*.jsonl"))
    return files[-1] if files else None


def _cell(x: float, y: float) -> tuple[int, int]:
    return (
        min(GRID - 1, max(0, int(x * GRID))),
        min(GRID - 1, max(0, int(y * GRID))),
    )


def _load(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pct(values: list[float]) -> str:
    return f"{statistics.median(values):.3f}" if values else "--"


def analyze(rows: list[dict], path: Path) -> int:
    if len(rows) < 50:
        print(f"记录只有 {len(rows)} 行，太短了。多用一会儿再分析。")
        return 1

    duration = rows[-1]["t"] - rows[0]["t"]
    print("=" * 68)
    print(f"记录文件：{path.name}")
    print(f"时长 {duration / 60:.1f} 分钟，{len(rows)} 帧")
    print("=" * 68)

    clicks = [r for r in rows if "c" in r and any(
        k.startswith("left_") for k in r["c"]
    )]
    print(f"\n点击事件 {len(clicks)} 次")

    # ---- 1. 反复点击热点 ----
    print("\n[1] 反复点击热点（同一小片区域点了 >= "
          f"{CLICK_MIN} 次 —— 通常说明前几次没点中）")
    clusters: list[list[dict]] = []
    for row in clicks:
        for cluster in clusters:
            cx = statistics.mean(r["x"] for r in cluster)
            cy = statistics.mean(r["y"] for r in cluster)
            if abs(cx - row["x"]) <= CLICK_RADIUS and abs(cy - row["y"]) <= CLICK_RADIUS:
                cluster.append(row)
                break
        else:
            clusters.append([row])
    clusters.sort(key=len, reverse=True)
    hot = [c for c in clusters if len(c) >= CLICK_MIN]
    if not hot:
        print("    没有（点击都很分散，说明点得比较顺）")
    for cluster in hot[:8]:
        cx = statistics.mean(r["x"] for r in cluster)
        cy = statistics.mean(r["y"] for r in cluster)
        span = (cluster[-1]["t"] - cluster[0]["t"]) if len(cluster) > 1 else 0.0
        print(
            f"    位置 ({cx:.2f}, {cy:.2f})  点击 {len(cluster)} 次"
            f"  跨度 {span:.1f}s",
        )

    # ---- 2. 长时间逗留热点 ----
    print(f"\n[2] 长时间逗留热点（同格累计 > {DWELL_MIN_SECONDS}s —— 来回微调）")
    dwell: dict[tuple[int, int], list[float]] = defaultdict(list)
    for prev, cur in zip(rows, rows[1:]):
        dt = cur["t"] - prev["t"]
        if dt <= 0 or dt > 0.5:      # 跳变/停顿不计
            continue
        dwell[_cell(cur["x"], cur["y"])].append(dt)
    dwell_ranked = sorted(
        ((cell, sum(times)) for cell, times in dwell.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    shown = 0
    for (gx, gy), seconds in dwell_ranked:
        if seconds < DWELL_MIN_SECONDS or shown >= 8:
            break
        print(
            f"    格子 ({gx / GRID:.2f}, {gy / GRID:.2f})  累计停留 {seconds:.1f}s",
        )
        shown += 1
    if shown == 0:
        print("    没有明显停留热点")

    # ---- 3. 捏了却没点 ----
    print("\n[3] 「捏了却没产生点击」的片段（捏合比持续低于阈值但没出点击）")
    engaged: list[dict] = []
    segments: list[list[dict]] = []
    for row in rows:
        low = row["r"] <= 0.42
        clicked = "c" in row and any(k.startswith("left_") for k in row["c"])
        if low:
            engaged.append(row)
            if clicked:
                # 点了就结算这一段，不算问题
                engaged = []
        elif engaged:
            if len(engaged) >= 4:          # 至少 4 帧（约 0.15s）
                segments.append(engaged)
            engaged = []
    if engaged and len(engaged) >= 4:
        segments.append(engaged)
    if not segments:
        print("    没有（每次捏合都产生了点击）")
    else:
        total = sum(len(s) for s in segments)
        print(
            f"    {len(segments)} 段，共 {total} 帧"
            f"（占全部帧的 {total / len(rows) * 100:.1f}%）",
        )
        for seg in segments[:8]:
            xs = [r["x"] for r in seg]
            ys = [r["y"] for r in seg]
            print(
                f"    位置约 ({statistics.mean(xs):.2f}, "
                f"{statistics.mean(ys):.2f})  "
                f"持续 {len(seg)} 帧  "
                f"最低捏合比 {min(r['r'] for r in seg):.2f}",
            )

    # ---- 4. 覆盖率 ----
    visited = {_cell(r["x"], r["y"]) for r in rows}
    total_cells = GRID * GRID
    print(
        f"\n[4] 屏幕覆盖率：{len(visited) / total_cells * 100:.0f}%"
        f"（{len(visited)}/{total_cells} 格）",
    )
    print("    从没去过的格子集中在外围时，多半是「够不到」，而不是没用。")

    # ---- 热图 ----
    heat = np.zeros((GRID, GRID), dtype=np.float32)
    for row in rows:
        gx, gy = _cell(row["x"], row["y"])
        heat[gy, gx] += 1
    heat = (heat / max(heat.max(), 1.0) * 255).astype(np.uint8)
    image = cv2.resize(heat, (800, 800), interpolation=cv2.INTER_NEAREST)
    image = cv2.applyColorMap(image, cv2.COLORMAP_TURBO)
    for cluster in clusters:
        cx = int(statistics.mean(r["x"] for r in cluster) * 799)
        cy = int(statistics.mean(r["y"] for r in cluster) * 799)
        color = (0, 0, 255) if len(cluster) >= CLICK_MIN else (255, 255, 255)
        cv2.circle(image, (cx, cy), 6, color, 2, cv2.LINE_AA)
    out = LOG_DIR / f"usage-heatmap-{path.stem.split('-', 1)[-1]}.png"
    cv2.imwrite(str(out), image)
    print(f"\n热图已保存：{out}")
    print("（红圈 = 反复点击热点；白圈 = 普通点击）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="分析日常使用记录")
    parser.add_argument("path", nargs="?", help="usage-*.jsonl 路径，省略则取最新")
    args = parser.parse_args()

    path = Path(args.path) if args.path else _latest_log()
    if path is None or not path.exists():
        print("找不到使用记录。先打开 config.json 的 usage_log_enabled 再用一段时间。")
        return 2
    return analyze(_load(path), path)


if __name__ == "__main__":
    raise SystemExit(main())
