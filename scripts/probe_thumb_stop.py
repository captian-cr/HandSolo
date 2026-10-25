"""定位「向下移动光标卡住/颤动」——量拇指张开度与停住锁存。

怀疑对象（`gestures.py:_allow_cursor_move`）：

    spread = _thumb_spread_value(points)      # dist(拇指尖4, 食指MCP5)/掌宽
    if spread >= thumb_stop_threshold:        # 用户配置 = 0.76
        self._thumb_stop_latched = True       # → 该帧起 MOVE 被全部丢弃
    return not self._thumb_stop_latched

即：拇指张开度一旦越过阈值，光标**冻结**；回落到
``thumb_resume_threshold``(0.72) 以下才恢复。若向下指时这个值正好在
0.76/0.72 之间来回穿，就表现为「卡住 + 颤动」。

本脚本逐帧记录 tip_y / 拇指张开度 / 捏合比，并对多个候选阈值
分别统计"会被判成停住"的比例，据此决定该把阈值调到多少。

用法::

    cd "<项目根目录>"
    python -m scripts.probe_thumb_stop

约 25 秒，跟着提示做动作。按 q / ESC 中止。
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
from handmouse.gestures import (  # noqa: E402
    _distance_2d,
    _finger_extensions,
    _is_fist,
    _palm_width,
    _thumb_spread_value,
)
from handmouse.vision import (  # noqa: E402
    default_model_path,
    prepare_recognition_frame,
)

WINDOW = "thumb stop probe"
CANDIDATES = (0.76, 0.85, 0.95, 1.05, 1.15)

STAGES = [
    (5.0, "只伸食指、拇指收起，手放正中"),
    (7.0, "保持手势，慢慢往下移，一直移到手快出画面"),
    (3.0, "停在最下面别动"),
    (5.0, "慢慢往上回到正中"),
    (3.0, "停在正中别动"),
]


def main() -> int:
    config = ConfigStore().load()
    model_path = default_model_path()
    if not model_path.exists():
        print(f"缺少模型文件：{model_path}")
        return 2

    latest: dict = {"result": None}

    def on_result(result, _img, _ts) -> None:
        latest["result"] = result

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

    print(
        f"当前 thumb_stop_enabled={config.thumb_stop_enabled} "
        f"threshold={config.thumb_stop_threshold} "
        f"resume={config.thumb_resume_threshold}",
    )
    print("=" * 64)
    print("先看预览窗里骨架是否跟手，然后跟着提示做动作。约 25 秒。")
    print("=" * 64)
    for i in range(3, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1.0)
    print(" 开始！\n")

    records: list[dict] = []
    last_ts = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for stage_index, (duration, hint) in enumerate(STAGES):
            print(f"[阶段 {stage_index + 1}] {hint}")
            stage_start = time.monotonic()
            while time.monotonic() - stage_start < duration:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if config.mirror:
                    frame = cv2.flip(frame, 1)
                prepared, transform = prepare_recognition_frame(
                    frame, config.recognition_zoom, config.inference_width,
                )
                rgb = np.ascontiguousarray(
                    cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB),
                )
                ts = max(int(time.monotonic() * 1000), last_ts + 1)
                last_ts = ts
                landmarker.detect_async(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts,
                )

                view = frame.copy()
                result = latest["result"]
                info = "no hand"
                color = (0, 0, 255)
                if result is not None and result.hand_landmarks:
                    pts = [transform.restore(p) for p in result.hand_landmarks[0]]
                    draw_skeleton(view, pts)
                    tip = pts[8]
                    palm = max(_palm_width(pts), 1e-4)
                    spread = _thumb_spread_value(pts)
                    pinch = _distance_2d(pts[4], pts[8]) / palm
                    records.append(
                        {
                            "stage": stage_index,
                            "tip_y": tip.y,
                            "spread": spread,
                            "pinch": pinch,
                            "fist": bool(_is_fist(pts)),
                            "index_ext": bool(_finger_extensions(pts)[0]),
                        },
                    )
                    stop_now = spread >= config.thumb_stop_threshold
                    color = (0, 0, 255) if stop_now else (0, 255, 0)
                    info = (
                        f"y={tip.y:.2f} spread={spread:.2f} "
                        f"pinch={pinch:.2f} "
                        f"{'STOP(冻结!)' if stop_now else 'move OK'}"
                    )
                cv2.rectangle(view, (0, 0), (view.shape[1], 56), (24, 24, 24), -1)
                cv2.putText(
                    view, f"[{stage_index + 1}/{len(STAGES)}] {hint}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (255, 255, 255), 1, cv2.LINE_AA,
                )
                cv2.putText(
                    view, info, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA,
                )
                cv2.imshow(WINDOW, view)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                    cap.release()
                    cv2.destroyWindow(WINDOW)
                    return _report(config, records)

    cap.release()
    cv2.destroyWindow(WINDOW)
    return _report(config, records)


def _report(config, records: list[dict]) -> int:
    print("\n" + "=" * 64)
    print("诊断结果")
    print("=" * 64)
    if not records:
        print("一帧都没检到手。")
        return 1

    print(f"有效帧 {len(records)}")
    ys = [r["tip_y"] for r in records]
    print(f"指尖 y 范围: {min(ys):.3f} ~ {max(ys):.3f}")

    low = [r for r in records if r["tip_y"] >= 0.55]
    print(f"\n画面下半部（y>=0.55）共 {len(low)} 帧")
    if not low:
        print("  没有下半部数据，重跑一次并把手移到下面。")
        return 0

    spreads = [r["spread"] for r in low]
    print(
        f"  拇指张开度 spread: 最小{min(spreads):.2f} "
        f"中位{statistics.median(spreads):.2f} 最大{max(spreads):.2f}",
    )

    print("\n  各候选阈值下「被判成停住」的比例：")
    for th in CANDIDATES:
        hit = sum(1 for s in spreads if s >= th)
        mark = "  <- 当前" if abs(th - config.thumb_stop_threshold) < 1e-6 else ""
        print(
            f"    threshold={th:.2f}: 冻结 {hit / len(spreads) * 100:5.1f}%"
            f"{mark}",
        )

    # 阈值穿越次数：频繁穿越 = 光标一顿一顿
    cur = config.thumb_stop_threshold
    crosses = 0
    prev = None
    for s in spreads:
        state = s >= cur
        if prev is not None and state != prev:
            crosses += 1
        prev = state
    print(f"\n  以当前阈值 {cur} 判定的状态翻转次数: {crosses} 次")
    print(
        "  （翻转多 = 冻结/恢复交替 = 用户感觉「颤动」；"
        f"共 {len(spreads)} 帧）",
    )

    fists = sum(1 for r in low if r["fist"])
    curled = sum(1 for r in low if not r["index_ext"])
    pinches = sum(1 for r in low if r["pinch"] <= config.pinch_engage)
    print(f"\n  下半部其它误判：握拳 {fists / len(low) * 100:.1f}%  "
          f"食指判弯 {curled / len(low) * 100:.1f}%  "
          f"捏合 {pinches / len(low) * 100:.1f}%")

    print("\n把上面整段贴回来。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
