"""点击标定：用你的实际点击反推最合适的灵敏度（margin）。

用户 2026-10-21 提出：在屏幕各处依次冒出圆点，我控制光标点中它，
**指头为了走到那个位置所移动的距离**就是标定依据，重复多次后由数据算出速率。

原理
----
引擎的映射是线性的（`filters.map_activity_region`）：

    screen = (tip - margin) / (1 - 2·margin)

把它看成一次线性回归 ``screen = a·tip + b``，用「捏合瞬间的 tip 坐标」当自变量、
「目标点的屏幕坐标」当因变量做最小二乘，就能反解出：

    margin = -b / a          （= 你自然到达的最小 tip 位置）
    手行程 = 1/a

也就是说：**回归出来的 margin 就是"你的手实际用到的范围"**，把它写进配置，
手不需要再抬到画面边缘，屏幕四角却都能到。

v2（2026-10-21）为什么改成**不经过手势引擎**
--------------------------------------------
v1 用 `GesturePipeline` 驱动，为了省掉"进入门"姿势把关卡关了 —— 但那会切换成
"握拳=切换暂停"的旧语义，而**捏合姿势本身就像握拳**，于是用户一捏就暂停，
表现为「能识别食指但捏合没反应」。此外引擎还有约 6% 的捏合误判。

v2 改为自采集：只做「相机 → MediaPipe → 映射 → 画准星」，捏合由本脚本
**直接按捏合比判**（跌破 pinch_engage 记一次、回升过 pinch_release 算完成），
不碰状态机，因此不会被暂停/闸门/误判干扰。

用法::

    cd "<项目根目录>"
    python -m scripts.calibrate_gain

安全说明：本脚本**不接管系统鼠标**，光标是画在标定窗口里的十字准星。
按 ``q``/``ESC`` 可随时中止。
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
from handmouse.gestures import _distance_2d, _palm_width  # noqa: E402
from handmouse.vision import (  # noqa: E402
    default_model_path,
    prepare_recognition_frame,
)

WINDOW = "gain calibration"
WIN_W, WIN_H = 1280, 720

#: 9 个目标点（归一化屏幕坐标）：四角 + 四边中点 + 正中。
TARGETS = [
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
]

ROUNDS = 2
#: 换目标后的缓冲时间：避免上一次捏合的"松开"被算成这次点击。
SETTLE_SECONDS = 0.8


def _canvas() -> np.ndarray:
    img = np.full((WIN_H, WIN_W, 3), 24, dtype=np.uint8)
    cv2.rectangle(img, (2, 2), (WIN_W - 3, WIN_H - 3), (55, 55, 55), 1)
    return img


def _to_px(nx: float, ny: float) -> tuple[int, int]:
    return int(nx * (WIN_W - 1)), int(ny * (WIN_H - 1))


def _draw_target(img, nx, ny, index, total, radius=22) -> None:
    cx, cy = _to_px(nx, ny)
    cv2.circle(img, (cx, cy), radius, (70, 70, 70), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), radius, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, (0, 220, 255), -1, cv2.LINE_AA)
    cv2.putText(
        img, f"{index + 1}/{total}", (cx - 18, cy - radius - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA,
    )


def _draw_cursor(img, nx, ny, pinching: bool) -> None:
    cx, cy = _to_px(nx, ny)
    color = (0, 120, 255) if pinching else (0, 255, 120)
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 26, 2, cv2.LINE_AA)


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """返回 (a, b)，使 y ≈ a·x + b。"""

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 1.0, 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    a = sxy / sxx
    return a, mean_y - a * mean_x


def _r_squared(
    xs: list[float],
    ys: list[float],
    a: float,
    b: float,
) -> float:
    """拟合优度：1 = 完美线性；越低说明手的位置和目标越不对应。"""

    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot <= 1e-12:
        return 0.0
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    return max(0.0, 1.0 - ss_res / ss_tot)


class PinchTracker:
    """与真实应用同构的「捏合趋向锚定 + 完成判定」。

    与 `gestures._pinch_approach_detected` 一样，用**逼近阈值**（默认 0.70）
    提前冻住光标，避免捏合动作把手带偏；真正算"捏了"仍要求跌破
    `pinch_engage`，松开过 `pinch_release` 才算完成一次点击。

    ⚠️ 释放条件必须用 ``ratio > anchor_ratio``（退回原处 = 放弃），
    **不能**用 ``>= pinch_release``：0.55 低于锚定阈值 0.70，
    那样会"一锚定就立刻释放"，锚定形同失效。
    """

    def __init__(self, anchor_ratio: float, engage: float, release: float) -> None:
        self.anchor_ratio = float(anchor_ratio)
        self.engage = float(engage)
        self.release = float(release)
        self.anchored = False
        self.pinched = False
        self.tip = None
        self.cursor = None

    def reset(self) -> None:
        self.anchored = False
        self.pinched = False
        self.tip = None
        self.cursor = None

    def update(self, ratio: float, tip, cursor) -> str | None:
        """推进一帧；返回 ``start`` / ``complete`` / ``cancel`` / ``None``。"""

        if not self.anchored and ratio <= self.anchor_ratio:
            self.anchored = True
            self.pinched = False
            self.tip = tip
            self.cursor = cursor
            return "start"
        if self.anchored and ratio <= self.engage:
            self.pinched = True
            return None
        if self.anchored and self.pinched and ratio >= self.release:
            self.anchored = False
            self.pinched = False
            return "complete"
        if self.anchored and not self.pinched and ratio > self.anchor_ratio:
            self.anchored = False
            self.tip = None
            return "cancel"
        return None


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

    print("=" * 70)
    print("点击标定 v2：不用刻意对准，捏的位置就是数据。")
    print(f"共 {len(TARGETS) * ROUNDS} 个点。")
    print("看到圆点后，用你觉得最舒服的幅度把光标往那个方向移，")
    print("觉得到位了就捏一下 —— 不用刻意让准星压住圆点。")
    print("按 q / ESC 中止。")
    print("=" * 70)
    for i in range(3, 0, -1):
        print(f"  {i}...", end="", flush=True)
        time.sleep(1.0)
    print(" 开始！\n")

    samples: list[dict] = []
    rejected = 0
    last_ts = 0
    cursor = (0.5, 0.5)
    aborted = False

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            for rnd in range(ROUNDS):
                for index, (tx, ty) in enumerate(TARGETS):
                    label = f"{rnd + 1}/{ROUNDS} · {index + 1}/{len(TARGETS)}"
                    settle_until = time.monotonic() + SETTLE_SECONDS
                    # 捏合趋向锚定（与应用内一致）：捏合比逼近就冻住准星。
                    tracker = PinchTracker(
                        float(getattr(config, "pinch_anchor_ratio", 0.70)),
                        config.pinch_engage,
                        config.pinch_release,
                    )
                    done = False
                    no_hand_since = time.monotonic()

                    while not done:
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

                        img = _canvas()
                        result = latest["result"]
                        has_hand = bool(result is not None and result.hand_landmarks)
                        if has_hand:
                            no_hand_since = time.monotonic()
                            points = [
                                transform.restore(p)
                                for p in result.hand_landmarks[0]
                            ]
                            draw_skeleton(img, points)
                            tip = points[8]
                            palm = max(_palm_width(points), 1e-4)
                            ratio = _distance_2d(points[4], points[8]) / palm
                            mapped = map_activity_region(
                                tip.x, tip.y,
                                config.pointer_margin_x, config.pointer_margin_y,
                            )
                            # 与真实应用同构的两段式捏合（见 PinchTracker）。
                            # 锚定期间准星不动，所以记录的位置就是应用里
                            # 点击会落到的位置 —— 这正是"捏合趋向"要保证的事。
                            event = tracker.update(ratio, tip, cursor)
                            if event == "complete":
                                if time.monotonic() >= settle_until:
                                    miss = float(
                                        np.hypot(
                                            tracker.cursor[0] - tx,
                                            tracker.cursor[1] - ty,
                                        )
                                    )
                                    # 🔴 不按"离目标太远"拒绝样本：
                                    # 增益还没调准时准星本来就到不了目标，
                                    # 而这些"偏得很远"的点恰恰是最需要的拟合数据。
                                    # 拒绝它们会死锁（v2 实测：41 次全被拒、0 成功）。
                                    samples.append(
                                        {
                                            "round": rnd,
                                            "index": index,
                                            "target_x": tx,
                                            "target_y": ty,
                                            "tip_x": tracker.tip.x,
                                            "tip_y": tracker.tip.y,
                                            "cursor_x": tracker.cursor[0],
                                            "cursor_y": tracker.cursor[1],
                                            "error": miss,
                                            "pinch": ratio,
                                        },
                                    )
                                    done = True
                                tracker.reset()
                            # 可见准星始终跟随手 —— 标定时一旦冻住就没法瞄准了。
                            # 记录点仍取"锚定那一刻"，与应用里点击实际落点一致。
                            cursor = mapped
                            live = (
                                f"捏合比 {ratio:.2f}   "
                                f"距目标 {np.hypot(cursor[0] - tx, cursor[1] - ty):.2f}"
                                f"   {'已锚定' if tracker.anchored else ''}"
                            )

                        _draw_target(img, tx, ty, index, len(TARGETS))
                        _draw_cursor(img, cursor[0], cursor[1], tracker.anchored)
                        hint = (
                            f"{label}   朝圆点方向移到舒服的位置 → 捏一下（不用对齐准星）"
                            if has_hand
                            else "没检到手 —— 把手放进画面（食指伸出）"
                        )
                        if has_hand:
                            cv2.putText(
                                img, live, (16, WIN_H - 46),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (150, 150, 150), 1, cv2.LINE_AA,
                            )
                        cv2.putText(
                            img, hint, (16, WIN_H - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                            (200, 200, 200) if has_hand else (0, 120, 255),
                            1, cv2.LINE_AA,
                        )
                        cv2.putText(
                            img,
                            f"已完成 {len(samples)}/{len(TARGETS) * ROUNDS}",
                            (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (150, 150, 150), 1, cv2.LINE_AA,
                        )
                        cv2.imshow(WINDOW, img)
                        if (cv2.waitKey(1) & 0xFF) in (27, ord("q"), ord("Q")):
                            aborted = True
                            break
                        if time.monotonic() - no_hand_since > 25.0:
                            print("连续 25 秒没检到手，已中止。")
                            aborted = True
                            break
                    if aborted:
                        break
                    print(f"  {label}  目标({tx:.2f},{ty:.2f})  已完成 {len(samples)}")
                if aborted:
                    break
    finally:
        cap.release()
        cv2.destroyWindow(WINDOW)

    if aborted:
        print("已中止。")
    return _report(config, samples, rejected)


def _report(config, samples: list[dict], rejected: int = 0) -> int:
    print("\n" + "=" * 70)
    print("标定结果")
    print("=" * 70)
    if len(samples) < 5:
        print(
            f"只采到 {len(samples)} 个点（丢弃 {rejected}），样本太少无法拟合"
            "（至少 5 个）。",
        )
        return 1

    print(f"有效点击 {len(samples)} 次   丢弃误触 {rejected} 次\n")

    tip_xs = [s["tip_x"] for s in samples]
    tip_ys = [s["tip_y"] for s in samples]
    print("[1] 你实际用到的指尖范围（画面归一化）")
    print(f"    x: {min(tip_xs):.3f} ~ {max(tip_xs):.3f}")
    print(f"    y: {min(tip_ys):.3f} ~ {max(tip_ys):.3f}")

    ax, bx = _least_squares(tip_xs, [s["target_x"] for s in samples])
    ay, by = _least_squares(tip_ys, [s["target_y"] for s in samples])

    print("\n[2] 最小二乘拟合  screen = a·tip + b")
    r2_x = _r_squared(tip_xs, [s["target_x"] for s in samples], ax, bx)
    r2_y = _r_squared(tip_ys, [s["target_y"] for s in samples], ay, by)
    print(f"    x: a={ax:.3f}  b={bx:.3f}   R²={r2_x:.3f}")
    print(f"    y: a={ay:.3f}  b={by:.3f}   R²={r2_y:.3f}")
    print("    （R² 越接近 1，说明你捏合时的指尖位置和目标位置对应得越一致）")

    if abs(ax) < 0.3 or abs(ay) < 0.3:
        print("\n拟合退化（a 太小）：指尖几乎没有随目标位置移动。")
        print("可能是捏合时手基本没动。重跑一次，每个点都朝它的方向移一段。")
        return 1
    if min(r2_x, r2_y) < 0.5:
        print(
            "\n⚠️ 拟合质量偏低（R² < 0.5）：说明同一方向的点，你手到位的位置差别较大。"
            "\n   建议加做几轮，或每个点都用「同样舒服的幅度」去够。",
        )

    # ---- 两种推导，分开算，别混用 ----
    # 映射 t = (tip - lo)/(1 - lo - hi)  →  a = 1/(1-lo-hi), b = -lo/(1-lo-hi)
    # 反解：lo = -b/a，hi = 1 - 1/a - lo      （四边独立，允许上下/左右不同）
    # 旧版把 "lo = -b/a" 和 "行程 = 1-2·lo" 混着用，两者在数据有偏移时互相矛盾，
    # 曾算出"margin 0.37 但行程 25%"这种自相矛盾的结果。
    def from_fit(a: float, b: float) -> tuple[float, float]:
        lo = -b / a
        hi = 1.0 - 1.0 / a - lo
        return lo, hi

    fx_lo, fx_hi = from_fit(ax, bx)
    fy_lo, fy_hi = from_fit(ay, by)

    # 另一路：直接拿实测最小/最大值铺满屏幕（R² 低时更可靠）
    mx_lo, mx_hi = min(tip_xs), 1.0 - max(tip_xs)
    my_lo, my_hi = min(tip_ys), 1.0 - max(tip_ys)

    print("\n[3] 候选内缩（两条路，R² 低时用「实测极值」那条）")
    print(
        f"    回归法   x: 左{fx_lo:+.2f} 右{fx_hi:+.2f}   "
        f"y: 上{fy_lo:+.2f} 下{fy_hi:+.2f}",
    )
    print(
        f"    极值法   x: 左{mx_lo:.2f} 右{mx_hi:.2f}   "
        f"y: 上{my_lo:.2f} 下{my_hi:.2f}",
    )
    use_extremes = min(r2_x, r2_y) < 0.5
    rec_x = (mx_lo, mx_hi) if use_extremes else (fx_lo, fx_hi)
    rec_y = (my_lo, my_hi) if use_extremes else (fy_lo, fy_hi)
    print(
        f"    → 推荐用「{'实测极值' if use_extremes else '回归'}」法："
        f"左{rec_x[0]:.2f} 右{rec_x[1]:.2f} 上{rec_y[0]:.2f} 下{rec_y[1]:.2f}",
    )
    print(
        f"      手行程：横 {1 - rec_x[0] - rec_x[1]:.0%}  "
        f"纵 {1 - rec_y[0] - rec_y[1]:.0%}",
    )
    cur_l, cur_t, cur_r, cur_b = config.effective_margins()
    print(
        f"      当前生效：左{cur_l:.2f} 上{cur_t:.2f} 右{cur_r:.2f} 下{cur_b:.2f}"
        f"（手行程 横{1 - cur_l - cur_r:.0%} 纵{1 - cur_t - cur_b:.0%}）",
    )
    print(
        "      注意：内缩是「手要移动多少」的反比 —— 内缩越大、手行程越短、"
        "越省力，但也越放大抖动。",
    )

    errors = [s["error"] for s in samples]
    print("\n[4] 当前速率下，你捏合时光标离圆点有多远")
    print(
        f"    中位 {statistics.median(errors):.4f}   "
        f"均值 {statistics.mean(errors):.4f}   "
        f"最大 {max(errors):.4f}",
    )
    print("    （1.0 = 整个屏幕宽。这不是你没瞄准，而是当前 margin 不匹配。）")

    print("\n[5] 分区域精度 —— 看哪个角最难")
    zones = {
        "左上": lambda s: s["target_x"] < 0.35 and s["target_y"] < 0.35,
        "右上": lambda s: s["target_x"] > 0.65 and s["target_y"] < 0.35,
        "中心": lambda s: 0.35 <= s["target_x"] <= 0.65
        and 0.35 <= s["target_y"] <= 0.65,
        "左下": lambda s: s["target_x"] < 0.35 and s["target_y"] > 0.65,
        "右下": lambda s: s["target_x"] > 0.65 and s["target_y"] > 0.65,
        "上边": lambda s: 0.35 < s["target_x"] < 0.65 and s["target_y"] < 0.35,
        "下边": lambda s: 0.35 < s["target_x"] < 0.65 and s["target_y"] > 0.65,
        "左边": lambda s: s["target_x"] < 0.35 and 0.35 <= s["target_y"] <= 0.65,
        "右边": lambda s: s["target_x"] > 0.65 and 0.35 <= s["target_y"] <= 0.65,
    }
    for name, rule in zones.items():
        sub = [s["error"] for s in samples if rule(s)]
        if sub:
            print(f"    {name}: n={len(sub):2d}  中位 {statistics.median(sub):.4f}")

    print("\n把上面整段贴回来，我按拟合值改进配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
