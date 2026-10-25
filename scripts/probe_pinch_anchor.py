"""为「捏合趋向 → 自动冻结光标」采集判据数据。

背景：用户 2026-10-21 提出——拇指张开手动停住已经关闭，那么
**如果能提前识别"要捏合了"，就在捏合前把光标钉住**，就不需要手动停住。

代码里已有锚定机制（`gestures.py:_pinch_anchor_active` / `cursor_frozen`），
但它当前必须靠 `_thumb_stop_latched` 触发，而该功能已关闭 → 锚定失效。

难点：**「手指朝下」和「要捏合」都会让 dist(拇指尖4, 食指尖8) 变小**
（`gestures.py:1466` 注释原话：手指向下指时食指尖自然勾向拇指）。
只靠捏合比无法区分，会误冻结。

所以本脚本一次性采集**全部原始量**并落 CSV，供离线挑选判据：
  d48  拇指尖↔食指尖      d45 拇指尖↔食指MCP
  d58  食指MCP↔食指尖     palm_w 掌宽(5→17)   hand_len 手长(腕0→中指MCP9)
  pinch=t48/palm_w  spread=d45/palm_w  以及指尖位移速度

用法::

    cd "<项目根目录>"
    python -m scripts.probe_pinch_anchor

约 21 秒。运行后 CSV 落在 ``logs/pinch-anchor-<时间戳>.csv``。
"""

from __future__ import annotations

import csv
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
)
from handmouse.vision import (  # noqa: E402
    default_model_path,
    prepare_recognition_frame,
)

WINDOW = "pinch anchor probe"

STAGES = [
    (4.0, "normal", "食指指着屏幕，左右慢慢移动"),
    (5.0, "down", "慢慢往下移到底（就是会卡住的那个动作）"),
    (3.0, "hold", "保持在下面，别动"),
    (9.0, "pinch", "停住不动，做 2~3 次捏合：慢慢靠近→捏住→松开"),
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

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    csv_path = log_dir / f"pinch-anchor-{time.strftime('%Y%m%d-%H%M%S')}.csv"

    print("=" * 68)
    print("先确认预览窗里骨架跟手，然后按提示做动作。约 21 秒。")
    print("=" * 68)
    for i in range(3, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1.0)
    print(" 开始！\n")

    rows: list[dict] = []
    prev_ts: int | None = None
    prev_xy: tuple[float, float] | None = None
    last_ts = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        for stage_index, (duration, tag, hint) in enumerate(STAGES):
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
                    hand_len = max(_distance_2d(pts[0], pts[9]), 1e-4)
                    d48 = _distance_2d(pts[4], pts[8])
                    d45 = _distance_2d(pts[4], pts[5])
                    d58 = _distance_2d(pts[5], pts[8])
                    pinch = d48 / palm
                    spread = d45 / palm
                    speed = 0.0
                    if prev_xy is not None and prev_ts is not None:
                        dt = max((ts - prev_ts) / 1000.0, 1e-3)
                        speed = (
                            np.hypot(tip.x - prev_xy[0], tip.y - prev_xy[1]) / dt
                        )
                    prev_xy = (tip.x, tip.y)
                    prev_ts = ts
                    rows.append(
                        {
                            "stage": tag,
                            "t": round(time.monotonic() - stage_start, 3),
                            "tip_x": round(tip.x, 4),
                            "tip_y": round(tip.y, 4),
                            "pinch": round(pinch, 4),
                            "spread": round(spread, 4),
                            "d48": round(d48, 5),
                            "d45": round(d45, 5),
                            "d58": round(d58, 5),
                            "palm_w": round(palm, 5),
                            "hand_len": round(hand_len, 5),
                            "speed": round(float(speed), 4),
                            "index_ext": int(bool(_finger_extensions(pts)[0])),
                            "fist": int(bool(_is_fist(pts))),
                        },
                    )
                    color = (0, 255, 0) if pinch > config.pinch_engage else (0, 200, 255)
                    info = (
                        f"y={tip.y:.2f} pinch={pinch:.2f} "
                        f"spread={spread:.2f} v={speed:.2f}"
                    )
                cv2.rectangle(view, (0, 0), (view.shape[1], 56), (24, 24, 24), -1)
                cv2.putText(
                    view, f"[{stage_index + 1}/{len(STAGES)}] {hint}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
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
                    return _save(rows, csv_path)

    cap.release()
    cv2.destroyWindow(WINDOW)
    return _save(rows, csv_path)


def _save(rows: list[dict], csv_path: Path) -> int:
    if not rows:
        print("一帧都没检到手。")
        return 1
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n采集 {len(rows)} 帧")
    for tag in ("normal", "down", "hold", "pinch"):
        sub = [r for r in rows if r["stage"] == tag]
        if not sub:
            continue
        pins = sorted(r["pinch"] for r in sub)
        spds = sorted(r["speed"] for r in sub)
        print(
            f"  {tag:<7} {len(sub):4d} 帧  "
            f"pinch 中位 {pins[len(pins) // 2]:.2f} (min {pins[0]:.2f})  "
            f"speed 中位 {spds[len(spds) // 2]:.2f}",
        )
    print(f"\nCSV 已写入：{csv_path}")
    print("把这个文件路径告诉我，我来挑判据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
