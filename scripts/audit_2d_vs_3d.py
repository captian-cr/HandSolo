"""2D 判据 vs 3D 判据：同一批真实画面上量"手指伸直度"的差异。

要验证的命题（这是"换个角度又失灵"的物理来源）：

    MediaPipe 的 ``hand_landmarks`` 是**归一化图像坐标**，x 除以画面宽、
    y 除以画面高。两个方向比例不同（4:3 画面里水平被拉伸 4/3），所以
    **同一根伸直的手指，在画面里横着指和竖着指，2D 量出来的伸直度不同**。
    而 ``hand_world_landmarks`` 是米制 3D 骨架，与朝向无关。

做法：对每张真实采集帧各算一遍
  * ``s2d`` = 2D 伸直度（用归一化 x/y，与线上判据同源）
  * ``s3d`` = 3D 伸直度（用 world 坐标的 x/y/z）
然后看：
  1. 两者不一致的帧占比；
  2. 不一致是否随"手指在画面里的朝向"变化（这是宽高比失真的指纹）；
  3. 把 2D 坐标按宽高比纠正后，不一致是否消失（证明成因）。

运行：python scripts/audit_2d_vs_3d.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from handmouse.gestures import _FINGER_CHAINS  # noqa: E402
from handmouse.vision import default_model_path  # noqa: E402


def straight(pts: list[tuple[float, float, float]], chain) -> float:
    mcp, pip, dip, tip = chain

    def d(a, b):
        return math.dist(pts[a], pts[b])

    path = d(mcp, pip) + d(pip, dip) + d(dip, tip)
    return d(mcp, tip) / path if path > 1e-9 else 0.0


def collect_frames(limit_dirs: int) -> list[Path]:
    frames: list[Path] = []
    for folder in sorted((ROOT / "logs").glob("frames-*")):
        images = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png")
        )
        frames.extend(images)
    return frames


def read_image(path: Path):
    """``cv2.imread`` 在 Windows 中文路径下会静默返回 None，改字节 + imdecode。"""

    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=int, default=2, help="每隔几帧取一张")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张（0=全部）")
    args = parser.parse_args()

    model_buffer = default_model_path().read_bytes()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_buffer),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    paths = collect_frames(0)[:: max(1, args.stride)]
    if args.limit:
        paths = paths[: args.limit]
    print(f"待处理帧 {len(paths)} 张")

    rows = []
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for i, path in enumerate(paths):
            image_bgr = read_image(path)
            if image_bgr is None:
                continue
            h, w = image_bgr.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not result.hand_landmarks or not result.hand_world_landmarks:
                continue
            norm = result.hand_landmarks[0]
            world = result.hand_world_landmarks[0]
            if len(norm) != 21 or len(world) != 21:
                continue

            pts2d = [(float(p.x), float(p.y), 0.0) for p in norm]
            # 宽高比纠正：把 x 乘回宽/高，让两个方向落在同一尺度上。
            # 注意归一化坐标 x∈[0,1] 对应 w 像素、y∈[0,1] 对应 h 像素。
            pts2d_fixed = [(float(p.x) * w, float(p.y) * h, 0.0) for p in norm]
            pts3d = [(float(p.x), float(p.y), float(p.z)) for p in world]

            for name, chain in _FINGER_CHAINS.items():
                s2d = straight(pts2d, chain)
                s2d_fixed = straight(pts2d_fixed, chain)
                s3d = straight(pts3d, chain)
                # 手指在画面里的朝向（度）：MCP→TIP 与水平轴夹角，0=横指、90=竖指
                mcp, _pip, _dip, tip = chain
                dx = pts2d[tip][0] - pts2d[mcp][0]
                dy = pts2d[tip][1] - pts2d[mcp][1]
                orientation = abs(math.degrees(math.atan2(dy, dx)))
                if orientation > 90:
                    orientation = 180 - orientation
                rows.append((name, s2d, s2d_fixed, s3d, orientation, w, h))

            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(paths)} ...")

    if not rows:
        print("没有取到任何手部帧")
        return

    print(f"\n有效手指样本 {len(rows)} 条")
    array = np.array([r[1:] for r in rows], dtype=float)
    s2d, s2d_fixed, s3d, orient = array[:, 0], array[:, 1], array[:, 2], array[:, 3]
    print(f"画面尺寸示例 {int(array[0, 4])}x{int(array[0, 5])}")

    # 只看"3D 认为伸直"的样本：这是判据真正要做对的那批
    straight_3d = s3d >= 0.97
    print(f"\n3D 判定「伸直」（s3d>=0.97）的样本 {int(straight_3d.sum())} 条")
    if straight_3d.sum() > 20:
        within = straight_3d & (s2d >= 0.97)
        print(f"  2D 也判伸直 {int(within.sum())} 条 → 一致率 {within.sum() / straight_3d.sum() * 100:.1f}%")

        # 按画面朝向分桶，看不一致是否依赖朝向（宽高比失真的指纹）
        print("\n  按手指在画面里的朝向分桶（0°=横指，90°=竖指）：")
        print(f"    {'朝向桶':<10}{'样本':>6}{'2D 伸直率':>12}{'2D纠正后':>12}{'平均偏差':>10}")
        for lo in range(0, 90, 15):
            mask = straight_3d & (orient >= lo) & (orient < lo + 15)
            if mask.sum() < 10:
                continue
            rate = (s2d[mask] >= 0.97).mean() * 100
            rate_fixed = (s2d_fixed[mask] >= 0.97).mean() * 100
            bias = (s2d[mask] - s3d[mask]).mean()
            print(
                f"    {lo:>2}-{lo + 15:<7}{int(mask.sum()):>6}{rate:>11.1f}%"
                f"{rate_fixed:>11.1f}%{bias:>+10.3f}"
            )

        print("\n  全部伸直样本上：")
        print(f"    2D 原始    平均偏差 {np.mean(s2d[straight_3d] - s3d[straight_3d]):+.4f}  "
              f"标准差 {np.std(s2d[straight_3d] - s3d[straight_3d]):.4f}")
        print(f"    2D 纠正后  平均偏差 {np.mean(s2d_fixed[straight_3d] - s3d[straight_3d]):+.4f}  "
              f"标准差 {np.std(s2d_fixed[straight_3d] - s3d[straight_3d]):.4f}")

    print("\n各指 s3d 分布（3D 伸直度）：")
    for name in _FINGER_CHAINS:
        values = np.array([r[3] for r in rows if r[0] == name])
        if values.size == 0:
            continue
        q = np.percentile(values, [5, 25, 50, 75, 95])
        print(
            f"  {name:<7} p5={q[0]:.3f} p25={q[1]:.3f} med={q[2]:.3f} "
            f"p75={q[3]:.3f} p95={q[4]:.3f}"
        )


if __name__ == "__main__":
    main()
