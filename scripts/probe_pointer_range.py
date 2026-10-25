"""实测「手指能到哪」与「向下时被误判成什么」——带实时预览。

背景（2026-10-20 用户反馈）：
1. 手指指到视频边缘，光标到不了屏幕边缘；
2. 手指向下移动光标不跟，下面四分之一很难指到，非常不稳定。

v2 修正了 v1 的两个测量缺陷：
* v1 用 ``RunningMode.IMAGE``，每帧重新做手掌检测、没有时序跟踪，检出手的
  能力远低于实际应用。**应用用的是 LIVE_STREAM**，这里改成一致。
* v1 没有预览窗口，用户看不到自己的手在不在画面里、骨架画得准不准，
  只能在"盲操作"下做动作 —— 那是 v1 失手率高达 99% 的主因。

用法（在项目根目录下执行，照抄这两行，别改）::

    cd "<项目根目录>"
    python -m scripts.probe_pointer_range

运行时会出现一个预览窗：摄像头画面 + 骨架 + 检出手状态 + 映射后光标位置。
先确认骨架跟着你的食指，再跟着控制台的阶段提示做动作。
按 ``q`` 或 ``ESC`` 可随时中止。
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import mediapipe as mp  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from handmouse.config import ConfigStore  # noqa: E402
from handmouse.debug_view import draw_skeleton  # noqa: E402
from handmouse.filters import map_activity_region  # noqa: E402
from handmouse.gestures import (  # noqa: E402
    _finger_extensions,
    _is_fist,
    _palm_width,
)
from handmouse.vision import (  # noqa: E402
    default_model_path,
    prepare_recognition_frame,
)

WINDOW = "pointer range probe"

# 分阶段动作提示：(持续秒数, 提示文字)
STAGES = [
    (5.0, "只伸食指，手放画面正中"),
    (5.0, "慢慢往上，手尽量抬高"),
    (4.0, "回到正中"),
    (6.0, "慢慢往下，到手腕快出画面为止"),
    (4.0, "回到正中"),
    (5.0, "慢慢往左，到画面左边缘"),
    (4.0, "回到正中"),
    (5.0, "慢慢往右，到画面右边缘"),
    (3.0, "结束，保持不动"),
]

MINI_W, MINI_H = 160, 90


def _draw_mini_screen(view: np.ndarray, mapped: tuple[float, float] | None) -> None:
    """右下角画一个缩小的屏幕示意图，标出映射后的光标位置。"""

    h, w = view.shape[:2]
    x0 = w - MINI_W - 12
    y0 = h - MINI_H - 12
    cv2.rectangle(view, (x0, y0), (x0 + MINI_W, y0 + MINI_H), (60, 60, 60), -1)
    cv2.rectangle(view, (x0, y0), (x0 + MINI_W, y0 + MINI_H), (180, 180, 180), 1)
    cv2.line(
        view,
        (x0 + MINI_W // 2, y0),
        (x0 + MINI_W // 2, y0 + MINI_H),
        (90, 90, 90),
        1,
    )
    cv2.line(
        view,
        (x0, y0 + MINI_H // 2),
        (x0 + MINI_W, y0 + MINI_H // 2),
        (90, 90, 90),
        1,
    )
    if mapped is not None:
        cx = int(x0 + mapped[0] * MINI_W)
        cy = int(y0 + mapped[1] * MINI_H)
        cx = max(x0 + 2, min(x0 + MINI_W - 2, cx))
        cy = max(y0 + 2, min(y0 + MINI_H - 2, cy))
        cv2.circle(view, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(
        view, "SCREEN", (x0 + 6, y0 + 14),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA,
    )


def main() -> int:
    config = ConfigStore().load()
    model_path = default_model_path()
    if not model_path.exists():
        print(f"缺少模型文件：{model_path}")
        return 2

    latest: dict = {"result": None, "at": 0}

    def on_result(result, _output_image, timestamp_ms) -> None:
        latest["result"] = result
        latest["at"] = timestamp_ms

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=config.min_detection_confidence,
        min_hand_presence_confidence=config.min_presence_confidence,
        min_tracking_confidence=config.min_tracking_confidence,
        result_callback=on_result,
    )

    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.camera_index, backend)
    if not cap.isOpened():
        print(f"打不开摄像头 {config.camera_index}")
        return 2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.camera_height)
    cap.set(cv2.CAP_PROP_FPS, config.camera_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"摄像头 {config.camera_index}  @ {config.camera_width}x{config.camera_height}")
    print(
        f"映射：margin_x={config.pointer_margin_x} margin_y={config.pointer_margin_y} "
        f"zoom={config.recognition_zoom}",
    )
    print()
    print("=" * 66)
    print("会出现一个预览窗口。先看窗口里骨架是否跟着你的食指，再开始动作。")
    print("按 q 或 ESC 可随时中止。")
    print("=" * 66)
    for index in range(3, 0, -1):
        print(f"  {index}...", end="", flush=True)
        time.sleep(1.0)
    print(" 开始！\n")

    records: list[dict] = []
    stage_lost: list[tuple[int, float]] = []
    last_ts = 0
    aborted = False

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for stage_index, (duration, hint) in enumerate(STAGES):
            print(f"[阶段 {stage_index + 1}] {hint}")
            stage_start = time.monotonic()
            frames = 0
            lost = 0
            while time.monotonic() - stage_start < duration:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                frames += 1
                if config.mirror:
                    frame = cv2.flip(frame, 1)
                prepared, transform = prepare_recognition_frame(
                    frame, config.recognition_zoom, config.inference_width,
                )
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB),
                )
                ts_ms = max(int(time.monotonic() * 1000), last_ts + 1)
                last_ts = ts_ms
                landmarker.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms,
                )

                view = frame.copy()
                result = latest["result"]
                entry: dict = {
                    "stage": stage_index,
                    "t": time.monotonic() - stage_start,
                    "detected": False,
                }
                if result is not None and result.hand_landmarks:
                    points = [transform.restore(p) for p in result.hand_landmarks[0]]
                    draw_skeleton(view, points)
                    tip = points[8]
                    palm = max(_palm_width(points), 1e-4)
                    pinch = float(
                        np.hypot(points[4].x - tip.x, points[4].y - tip.y)
                    ) / palm
                    mapped = map_activity_region(
                        tip.x, tip.y,
                        config.pointer_margin_x, config.pointer_margin_y,
                    )
                    entry.update(
                        {
                            "detected": True,
                            "tip_x": tip.x,
                            "tip_y": tip.y,
                            "palm": palm,
                            "pinch": pinch,
                            "fist": bool(_is_fist(points)),
                            "index_ext": bool(_finger_extensions(points)[0]),
                            "map_x": mapped[0],
                            "map_y": mapped[1],
                        },
                    )
                    _draw_mini_screen(view, mapped)
                    status = (
                        f"tip=({tip.x:.2f},{tip.y:.2f}) pinch={pinch:.2f} "
                        f"-> screen=({mapped[0]:.2f},{mapped[1]:.2f})"
                    )
                    color = (0, 255, 0)
                    label = "HAND OK"
                else:
                    _draw_mini_screen(view, None)
                    status = "no hand"
                    color = (0, 0, 255)
                    label = "NO HAND"
                    lost += 1
                records.append(entry)

                cv2.rectangle(view, (0, 0), (view.shape[1], 56), (24, 24, 24), -1)
                cv2.putText(
                    view, f"[{stage_index + 1}/{len(STAGES)}] {hint}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    1, cv2.LINE_AA,
                )
                cv2.putText(
                    view, f"{label}   {status}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                    1, cv2.LINE_AA,
                )
                cv2.imshow(WINDOW, view)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                    aborted = True
                    break
            lost_ratio = lost / max(frames, 1)
            stage_lost.append((stage_index, lost_ratio))
            print(
                f"           采集 {frames} 帧，未检到 {lost} 帧"
                f"（{lost_ratio * 100:.0f}%）\n",
            )
            if aborted:
                break

    cap.release()
    cv2.destroyWindow(WINDOW)
    if aborted:
        print("已中止。")
    return _report(config, records, stage_lost)


def _report(config, records: list[dict], stage_lost) -> int:
    print("=" * 66)
    print("诊断结果")
    print("=" * 66)

    detected = [r for r in records if r["detected"]]
    total = len(records)
    if not detected:
        print(f"共 {total} 帧，一帧都没检到手。先确认预览窗里能看到骨架。")
        return 1

    print(
        f"\n[0] 整体检出手比例：{len(detected)}/{total} = "
        f"{len(detected) / total * 100:.0f}%",
    )

    ys = [r["tip_y"] for r in detected]
    xs = [r["tip_x"] for r in detected]
    print("\n[1] 食指尖实际到达的画面范围（0=上/左，1=下/右）")
    print(f"    y: {min(ys):.3f} ~ {max(ys):.3f}")
    print(f"    x: {min(xs):.3f} ~ {max(xs):.3f}")

    mx, my = config.pointer_margin_x, config.pointer_margin_y
    checks = [
        ("屏幕下边缘", "y", ">=", 1.0 - my, max(ys)),
        ("屏幕上边缘", "y", "<=", my, min(ys)),
        ("屏幕右边缘", "x", ">=", 1.0 - mx, max(xs)),
        ("屏幕左边缘", "x", "<=", mx, min(xs)),
    ]
    print("\n[2] 当前配置要求 vs 实际能到")
    for name, axis, op, threshold, actual in checks:
        ok = actual >= threshold if op == ">=" else actual <= threshold
        print(
            f"    {name}: 需要 {axis} {op} {threshold:.3f}，"
            f"实际 {axis}={actual:.3f}  {'够得到' if ok else '够不到 ✗'}",
        )

    top = max(0.0, min(0.40, min(ys)))
    bottom = max(0.0, min(0.40, 1.0 - min(max(ys), 1.0)))
    left = max(0.0, min(0.40, min(xs)))
    right = max(0.0, min(0.40, 1.0 - min(max(xs), 1.0)))
    print("\n[3] 按实测范围建议的非对称内缩")
    print(
        f"    纵向：margin_top={top:.3f}  margin_bottom={bottom:.3f}"
        f"   （当前对称 margin_y={my}）",
    )
    print(
        f"    横向：margin_left={left:.3f}  margin_right={right:.3f}"
        f"   （当前对称 margin_x={mx}）",
    )

    print("\n[4] 各阶段：未检到率 / 误判率")
    for stage_index, lost_ratio in stage_lost:
        rows = [r for r in records if r["stage"] == stage_index and r["detected"]]
        head = f"    阶段{stage_index + 1}: 未检到{lost_ratio * 100:3.0f}%"
        if not rows:
            print(f"{head}  （该阶段一帧都没检到）")
            continue
        fists = sum(1 for r in rows if r["fist"])
        pinches = sum(1 for r in rows if r["pinch"] <= config.pinch_engage)
        curled = sum(1 for r in rows if not r["index_ext"])
        print(
            f"{head}  握拳{fists / len(rows) * 100:3.0f}%"
            f"  捏合{pinches / len(rows) * 100:3.0f}%"
            f"  食指判弯{curled / len(rows) * 100:3.0f}%",
        )

    low = [r for r in detected if r["tip_y"] >= 0.6]
    if low:
        fists = sum(1 for r in low if r["fist"])
        pinches = sum(1 for r in low if r["pinch"] <= config.pinch_engage)
        vals = [r["pinch"] for r in low]
        print(f"\n[5] 指尖在画面下半部（y>=0.6）共 {len(low)} 帧：")
        print(
            f"    判成握拳 {fists / len(low) * 100:.1f}%   "
            f"判成捏合 {pinches / len(low) * 100:.1f}%",
        )
        print(
            f"    捏合比 最小{min(vals):.2f} 中位{statistics.median(vals):.2f} "
            f"最大{max(vals):.2f}（阈值 {config.pinch_engage}）",
        )

    print("\n把上面这段（含 [0]~[5]）整段贴回来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
