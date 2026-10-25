"""首次使用向导：先标点，再把手势逐个练一遍，最后记住你的习惯。

这是给**下载用户**的第一课。作者自己的阈值是按他的手标定的，换一个人直接
用就是"勉强能用"；所以新用户第一次启动必须完成三件事：

1. **标点**：屏幕（预览窗）上画一个十字，用户把手移过去、用最舒服的姿势捏合。
   **捏合发生的位置就是他的"舒适点"**——它决定光标的活动范围，而不是让用户
   去理解 `pointer_margin_*` 是什么。
2. **量速率**：让手指像平时那样来回动一会儿，量出他习惯的移动速度分布。
3. **过手势**：七种手势各做一遍（准备 3 秒 + 练习/录制 6 秒），边做边看识别
   结果对不对——等于让他把每个动作都适应一遍，同时把样本录进他的样本库。

全程不动鼠标（这里不接 `GestureEngine`），也不写任何系统输入。

产物：
- `gesture_data/`：他的个人样本库（`collect.run_collect` 负责采，这里负责收尾）
- `config.json`：`pointer_margin_*`（由舒适点算出）、`onboarding_done=True`
- `gesture_data/onboarding.json`：舒适点、习惯速率、每种手势录了多少帧

🔴 **舒适点只用来定活动范围，不改捏合阈值**。捏合阈值由样本库在采集结束时
算出的拒识阈值 + `calibration` 的规则决定；拿"一次捏合"去改 `pinch_engage`
会把阈值钉死在某个偶然的姿势上（作者早期踩过：捏合比在同一个人的不同角度
上能差 3 倍）。
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from handmouse import APP_NAME
from handmouse.collect import PRACTICE_STEPS, _blank_frame, _open_workers
from handmouse.config import AppConfig, ConfigStore
from handmouse.contracts import DetectionPacket, FramePacket
from handmouse.debug_view import GRAY, GREEN, RED, WHITE, YELLOW, draw_hud, draw_skeleton
from handmouse.gesture_data import GestureDataset, default_dataset_path
from handmouse.gestures import _aspect_scale, _distance_2d, _isotropic, _palm_width
from handmouse.latest_slot import LatestSlot

#: 标定点。**每个点代表"屏幕上的哪个位置"**，不是"手要放哪儿"。
#:
#: 🔴 用户第一版反馈（原话）：「你给的标点就一次，没有其他位置的标点了，
#: 而且要写清楚不是要移动到标点的位置，而是移到你舒服的位置」。
#: 所以现在：① 标 3 个点（中央 / 左上 / 右下），用三点圈出的范围反推活动区；
#: ② 十字的位置**只表示屏幕上的相对方位**，屏幕上的文字明说"不是让你把手放这儿"。
@dataclass(frozen=True)
class _Target:
    label: str
    #: 十字画在预览窗里的位置——代表"这个点在屏幕上的相对方位"。
    mark: tuple[float, float]
    hint: str


CALIBRATION_TARGETS: tuple[_Target, ...] = (
    _Target("屏幕中央", (0.50, 0.50), "手放在你最舒服、最想长期待着的位置"),
    _Target("屏幕左上", (0.30, 0.32), "比刚才往左上挪一点（不用到画面边）"),
    _Target("屏幕右下", (0.70, 0.68), "比刚才往右下挪一点（不用到画面边）"),
)

#: 三点圈出的范围再放大这么多当活动区。
#: 取 1.25 而不是 1.0：用户捏合时不会正好落在边界上，留一点余量，
#: 否则"够到屏幕边"要靠把手指伸到极限，手感很别扭。
_SPREAD_GAIN = 1.25

#: 三点彼此相距都小于这个值时，认为"没提供范围信息"，退回单点逻辑。
_SPREAD_MIN = 0.08

#: 活动区半宽/半高（单点标定时用）。默认配置的内缩 0.2/0.16 对应
#: 半宽 0.30、半高 0.34，这里保持一致，换的只是"中心在哪"。
_HALF_X = 0.30
_HALF_Y = 0.34

#: 舒适点离画面正中多近就不改配置（避免把噪声当结论）。
_MARGIN_DEADZONE = 0.06

#: 活动区半宽/半高的下限。低于这个值等于"手几乎不用动光标就跑满全屏"，
#: 手感会变成一碰就飞，所以贴边的舒适点宁可让区域偏一点，也不能缩到这么小。
_MIN_HALF = 0.12

#: 单阶段最短/最长时间。
_CALIBRATE_TIMEOUT_S = 60.0
_SPEED_SECONDS = 7.0

_WINDOW = f"{APP_NAME} - Setup"
_ABORT_KEYS = (27, ord("q"), ord("Q"))

#: 用户在标点时捏合多少帧才算确认（3 帧 ≈ 0.1s，压掉单帧抖动）。
_PINCH_FRAMES = 3


@dataclass(frozen=True)
class ComfortPoint:
    """用户捏合时手指所在的位置。"""

    x: float
    y: float
    pinch_ratio: float
    samples: int


@dataclass(frozen=True)
class SpeedProfile:
    """他习惯的手部移动速度（归一化画面宽度/秒）。

    单位：1.0 = 每秒移动一个画面宽。
    """

    p50: float
    p90: float
    samples: int

    @property
    def label(self) -> str:
        """粗分类，给人看一眼用的。

        🔴 **边界是暂定值，只有作者一个人的数据（2026-10-25 实测 p50=0.029 /
        p90=0.090）**。第一版按 p50 划 0.8/1.6，结果真实数据全部落在
        "偏慢"的下限之外 27 倍 —— 说明那组边界是凭空拍的，不是量出来的。
        现在改成按 **p90**（p50 被"站着看提示语"的近静止帧拉低）划
        0.10 / 0.30 / 0.80，只求"能区分快慢"，**不要拿它做任何自动调节**。
        """

        if self.p90 < 0.10:
            return "很稳（移动幅度很小）"
        if self.p90 < 0.30:
            return "偏慢"
        if self.p90 < 0.80:
            return "中等"
        return "偏快"


def _effective_half(center: float, half: float) -> float:
    """把半宽/半高收缩到"以 ``center`` 为中心还能放得下"的尺寸。

    🔴 这是**必须**的一步，不是保险：舒适点若靠边（比如 y=0.72），
    半高 0.34 会让活动区伸到画面外（1.06），下边只能夹到 0，
    结果"活动区中心"就偏离了舒适点 —— 用户会觉得自己瞄准的位置不对。
    收缩到 ``min(half, center, 1-center)`` 才能保持中心精确落在舒适点上。
    """

    return max(_MIN_HALF, min(half, center, 1.0 - center))


def margins_from_comfort(
    cx: float,
    cy: float,
    *,
    half_x: float = _HALF_X,
    half_y: float = _HALF_Y,
) -> tuple[float, float, float, float] | None:
    """把**单个**舒适点换算成四边内缩 ``(left, top, right, bottom)``。

    ``None`` 表示"离正中太近，不值得改"——这种情况下保持配置里的原值，
    免得把一次偶然的偏斜写进配置。多点标定走 `margins_from_points`。
    """

    if abs(cx - 0.5) <= _MARGIN_DEADZONE and abs(cy - 0.5) <= _MARGIN_DEADZONE:
        return None
    return _box_to_margins(cx, cy, _effective_half(cx, half_x), _effective_half(cy, half_y))


def _box_to_margins(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
) -> tuple[float, float, float, float]:
    left = max(0.0, min(0.35, cx - hx))
    right = max(0.0, min(0.35, 1.0 - (cx + hx)))
    top = max(0.0, min(0.35, cy - hy))
    bottom = max(0.0, min(0.35, 1.0 - (cy + hy)))
    return left, top, right, bottom


def margins_from_points(
    points: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    """用**多个标定点**圈出的范围反推活动区。

    - 0 个点 → ``None``（什么都不改）
    - 1 个点 → 退回 `margins_from_comfort`（单点只能定"中心在哪"）
    - ≥2 个点 → 中心取包围盒中点，半宽/半高取包围盒的一半再放大 `_SPREAD_GAIN`

    🔴 三点相距都小于 `_SPREAD_MIN` 时按**单点**处理：用户把三次都捏在同一个
    地方，说明他没在给范围，硬按 3 厘米的范围去映射全屏会变成"一碰就飞"。
    """

    if not points:
        return None
    xs = [float(item[0]) for item in points]
    ys = [float(item[1]) for item in points]
    if len(points) == 1:
        return margins_from_comfort(xs[0], ys[0])

    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    if span_x < _SPREAD_MIN and span_y < _SPREAD_MIN:
        return margins_from_comfort(sum(xs) / len(xs), sum(ys) / len(ys))

    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    hx = _axis_half(span_x, _HALF_X)
    hy = _axis_half(span_y, _HALF_Y)
    return _box_to_margins(cx, cy, hx, hy)


def _axis_half(span: float, default: float) -> float:
    """单轴的活动半宽/半高。

    🔴 **某一轴跨度为 0 时不能套下限**（2026-10-25 真实数据发现）：作者第一次
    三点标定给出 y = 0.709 / 0.686 / 0.686 —— 纵向跨度只有 **0.0225**，三个点
    几乎在同一高度。当时按 `_MIN_HALF`(0.12) 兜底，算出的活动区只有 0.47 高，
    比默认 0.68 **敏感 37%**，手一抖光标就飞。

    道理是：跨度小 = **没有拿到这个方向的范围信息**，那就该退回默认尺度
    （0.34），而不是按"极小范围"去映射全屏。横向同理。
    """

    if span < _SPREAD_MIN:
        return default
    return max(_MIN_HALF, min(default, span / 2 * _SPREAD_GAIN))


def speed_profile(samples: list[tuple[float, float, float]]) -> SpeedProfile | None:
    """``[(t_seconds, x, y), ...]`` → 速度分布。

    速度单位是"归一化画面宽度/秒"：0.5 表示每秒移动半个画面宽。
    用 p50/p90 而不是均值——用户中途停顿时速度是 0，均值会被拉低。
    """

    speeds: list[float] = []
    for (t0, x0, y0), (t1, x1, y1) in zip(samples, samples[1:]):
        dt = t1 - t0
        if dt <= 1e-6 or dt > 0.5:  # 丢帧/长时间停顿不算速度
            continue
        speeds.append(((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / dt)
    if len(speeds) < 20:
        return None
    speeds.sort()
    return SpeedProfile(
        p50=speeds[len(speeds) // 2],
        p90=speeds[min(len(speeds) - 1, int(len(speeds) * 0.9))],
        samples=len(speeds),
    )


def _pinch_ratio(observation, aspect: float) -> float | None:
    points = observation.landmarks
    if len(points) != 21:
        return None
    geo = _isotropic(points, _aspect_scale(aspect))
    width = _palm_width(geo)
    if width <= 1e-6:
        return None
    return _distance_2d(geo[4], geo[8]) / width


def _draw_mark(frame, point: tuple[float, float], *, hit: bool, thin: bool = False) -> None:
    """画标点：十字 + 同心圆。捏中后变绿加粗。

    ``thin=True`` 时画细一号、中心点更小 —— 标定十字只表示"屏幕上的方位"，
    画粗了会让人以为必须把手对准它（第一版就是这个误解）。
    """

    height, width = frame.shape[:2]
    cx, cy = int(point[0] * width), int(point[1] * height)
    colour = GREEN if hit else YELLOW
    base = 26 if thin else 34
    length = base + (8 if hit else 0)
    thickness = 2 if thin else 3
    if hit:
        thickness += 1
    cv2.line(frame, (cx - length, cy), (cx + length, cy), colour, thickness)
    cv2.line(frame, (cx, cy - length), (cx, cy + length), colour, thickness)
    cv2.circle(frame, (cx, cy), length // 2, colour, max(1, thickness - 1))
    cv2.circle(frame, (cx, cy), 3 if not thin else 2, colour, -1)


def _phase(
    *,
    frames: LatestSlot[FramePacket],
    detections: LatestSlot[DetectionPacket],
    seconds: float,
    rows_fn,
    on_detection=None,
    mark: tuple[float, float] | None = None,
    mark_hit: bool = False,
    thin: bool = False,
    stop_when=None,
    confirm=None,
) -> tuple[bool, bool]:
    """跑一个阶段。返回 ``(被用户中断, 触发停止条件)``。

    ``confirm()`` 返回 True 时立刻结束本阶段（用于"等你读完题再开始"），
    与 ``stop_when`` 的区别是它不仅结束阶段、还会被视为**用户已确认**。
    """

    deadline = time.monotonic() + seconds
    last_revision = -1
    while time.monotonic() < deadline:
        packet, _ = frames.get_latest()
        detection, revision = detections.get_latest()
        fresh = detection is not None and revision != last_revision
        if fresh:
            last_revision = revision
            if on_detection is not None:
                on_detection(detection)
            if stop_when is not None and stop_when():
                return False, True

        view = packet.frame.copy() if packet is not None else _blank_frame()
        if mark is not None:
            _draw_mark(view, mark, hit=mark_hit, thin=thin)
        if detection is not None and detection.observations:
            observation = detection.observations[0]
            if len(observation.landmarks) == 21:
                draw_skeleton(view, observation.landmarks)
        remaining = max(0.0, deadline - time.monotonic())
        draw_hud(view, rows_fn(remaining))
        cv2.imshow(_WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in _ABORT_KEYS:
            return True, False
        if confirm is not None and confirm(key):
            return False, True
    return False, False


def _calibrate(config: AppConfig) -> tuple[list[ComfortPoint], SpeedProfile | None, bool]:
    """多点标定 + 量速率。返回 ``(标定点列表, 速率分布, 是否被中断)``。"""

    stop_event, frames, detections, errors, workers = _open_workers(config)
    profile: SpeedProfile | None = None
    aborted = False
    try:
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline and not stop_event.is_set():
            _, revision = detections.get_latest()
            if revision > 0:
                break
            time.sleep(0.1)
        if errors:
            raise RuntimeError(errors[0])

        # ---- 第一关：多点标定（三点定范围）----
        collected: list[ComfortPoint] = []
        engage = float(getattr(config, "pinch_engage", 0.42))
        total = len(CALIBRATION_TARGETS)

        for order, target in enumerate(CALIBRATION_TARGETS, start=1):
            hits: list[tuple[float, float, float]] = []
            streak = 0

            def watch_mark(detection, bucket=hits) -> None:
                nonlocal streak
                if not detection.observations:
                    streak = 0
                    return
                observation = detection.observations[0]
                ratio = _pinch_ratio(
                    observation, getattr(observation, "image_aspect", 4 / 3),
                )
                if ratio is None or ratio > engage:
                    streak = 0
                    return
                streak += 1
                if streak >= _PINCH_FRAMES:
                    tip = observation.landmarks[8]
                    bucket.append((tip.x, tip.y, ratio))

            def mark_rows(
                remaining: float,
                target: _Target = target,
                order: int = order,
                bucket=hits,
            ):
                done = len(bucket) >= 8
                return [
                    (f"① 标定 第 {order}/{total} 点：{target.label}", WHITE),
                    ("", WHITE),
                    ("十字只表示【屏幕上的位置】，不是让你把手放这儿", YELLOW),
                    ("手放在你自己最舒服的地方就行 —— " + target.hint, WHITE),
                    ("停住，捏合一下就完成" if not done else "很好，保持住……",
                     GREEN if done else WHITE),
                    (f"（按 ESC 可退出，本点超时会跳过）{remaining:.0f}s", GRAY),
                ]

            aborted, _ = _phase(
                frames=frames,
                detections=detections,
                seconds=_CALIBRATE_TIMEOUT_S / total,
                rows_fn=mark_rows,
                on_detection=watch_mark,
                mark=target.mark,
                mark_hit=len(hits) >= 8,
                stop_when=lambda bucket=hits: len(bucket) >= 8,
                # 十字只是方位示意，画细一点，别让它像"手的目标"
                thin=True,
            )
            if aborted:
                return [], None, True
            if not hits:
                print(f"[标定] {target.label} 这一点没确认到捏合，跳过（不影响其它点）。")
                continue
            xs = sorted(item[0] for item in hits)
            ys = sorted(item[1] for item in hits)
            ratios = sorted(item[2] for item in hits)
            mid = len(hits) // 2
            collected.append(
                ComfortPoint(
                    x=round(xs[mid], 4),
                    y=round(ys[mid], 4),
                    pinch_ratio=round(ratios[mid], 4),
                    samples=len(hits),
                ),
            )
            print(
                f"[标定] {target.label} → 你捏合在画面 ({xs[mid]:.2f}, {ys[mid]:.2f})，"
                f"捏合比 {ratios[mid]:.2f}",
            )

        # ---- 第二关：量速率 ----
        track: list[tuple[float, float, float]] = []
        started_ns: list[int] = []

        def watch_speed(detection) -> None:
            if not detection.observations:
                return
            observation = detection.observations[0]
            if len(observation.landmarks) != 21:
                return
            if not started_ns:
                started_ns.append(observation.timestamp_ns)
            tip = observation.landmarks[8]
            track.append(
                ((observation.timestamp_ns - started_ns[0]) / 1e9, tip.x, tip.y),
            )

        def speed_rows(remaining: float):
            return [
                ("② 量速率：像平时那样动手指", WHITE),
                (f"左右来回移动，**全程别停**（{remaining:.0f}s）", WHITE),
                ("我记录的是你自己习惯的速度，不是标准动作", GRAY),
                ("注意：看屏幕的时候手也别停下来，否则量出来会偏小", GRAY),
            ]

        aborted, _ = _phase(
            frames=frames,
            detections=detections,
            seconds=_SPEED_SECONDS,
            rows_fn=speed_rows,
            on_detection=watch_speed,
        )
        if not aborted:
            profile = speed_profile(track)
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
        try:
            cv2.destroyWindow(_WINDOW)
        except cv2.error:
            pass
    return collected, profile, aborted


def _welcome() -> bool:
    """开屏说明。返回 False 表示用户跳过。"""

    frame = _blank_frame()
    lines = [
        (f"{APP_NAME}：先花两分钟，把它调成你的手", WHITE),
        ("", WHITE),
        ("接下来三步：", WHITE),
        ("1) 用 3 个点标定你的活动范围（屏幕中央 / 左上 / 右下）", WHITE),
        ("    ★ 十字只表示【屏幕上的方位】，不是让你把手放上去", YELLOW),
        ("    手放在你自己觉得最舒服的地方，捏合一下就行 ——", GRAY),
        ("    我会把你捏合的位置，当成\"对应屏幕上那个点\"的位置记住", GRAY),
        ("2) 你随意动一动手指，我记录你自己的习惯速度", WHITE),
        ("3) 每种手势：先给你看题目，你捏合确认之后我才开始录", WHITE),
        ("", WHITE),
        ("这个过程不会动你的鼠标，也不会移动光标。随时按 ESC 退出。", YELLOW),
        ("准备好就按任意键开始", GREEN),
    ]
    draw_hud(frame, lines)
    cv2.imshow(_WINDOW, frame)
    key = cv2.waitKey(0) & 0xFF
    return key not in _ABORT_KEYS


def _apply_margins(config: AppConfig, points: list[ComfortPoint]):
    """把标定点写进 config 的四边内缩。返回 ``(left, top, right, bottom)`` 或 None。"""

    margins = margins_from_points([(item.x, item.y) for item in points])
    if margins is not None:
        left, top, right, bottom = margins
        config.pointer_margin_left = round(left, 3)
        config.pointer_margin_top = round(top, 3)
        config.pointer_margin_right = round(right, 3)
        config.pointer_margin_bottom = round(bottom, 3)
    return margins


def _save(
    config: AppConfig,
    dataset: GestureDataset,
    points: list[ComfortPoint],
    profile: SpeedProfile | None,
) -> Path:
    """落盘：样本库 + 配置 + 一份给人看的记录。"""

    from handmouse.gesture_model import clean_dataset, suggest_reject_distance

    stored = dataset
    if len(stored):
        stored, dropped = clean_dataset(stored)
        if dropped:
            print(f"[清洗] 剔除离群帧 {sum(dropped.values())} 帧")

    path = default_dataset_path()
    written = stored.save(path) if len(stored) else 0
    if written:
        threshold = suggest_reject_distance(stored)
        if threshold is not None:
            config.gesture_model_reject_distance = round(float(threshold), 4)
        config.gesture_model_enabled = True

    margins = _apply_margins(config, points)

    config.onboarding_done = True
    config.validate()
    ConfigStore().save(config)

    record = {
        "done_ns": time.time_ns(),
        "comfort_points": [asdict(item) for item in points],
        "speed": asdict(profile) if profile else None,
        "margins": (
            {"left": margins[0], "top": margins[1], "right": margins[2], "bottom": margins[3]}
            if margins
            else None
        ),
        "frames": stored.counts() if len(stored) else {},
    }
    out = path.with_name("onboarding.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def run_onboarding(config: AppConfig) -> int:
    """跑完整向导。返回进程退出码（0 = 完成）。"""

    from handmouse.collect import _report, run_collect

    print("=" * 62)
    print(f"{APP_NAME} 首次设置：把它调成你的手")
    print("=" * 62)
    if not _welcome():
        print("[设置] 已取消。随时可以重新跑：python -m handmouse.onboarding")
        return 1

    points, profile, aborted = _calibrate(config)
    if aborted:
        print("[设置] 已中断，什么都没改。")
        return 1
    if not points:
        print("[设置] 没读到捏合动作，跳过标定，活动范围保持默认。")
    else:
        listed = "、".join(f"({item.x:.2f}, {item.y:.2f})" for item in points)
        print(f"[标定] 共 {len(points)} 个点：{listed}")
        margins = margins_from_points([(item.x, item.y) for item in points])
        if margins is None:
            print("       三个点都贴近画面中央，活动范围保持默认。")
        else:
            print(
                f"       活动区内缩 left={margins[0]:.3f} top={margins[1]:.3f} "
                f"right={margins[2]:.3f} bottom={margins[3]:.3f}",
            )
            xs_pts = [item.x for item in points]
            ys_pts = [item.y for item in points]
            if len(points) >= 2 and (
                max(xs_pts) - min(xs_pts) < _SPREAD_MIN
                or max(ys_pts) - min(ys_pts) < _SPREAD_MIN
            ):
                print(
                    "       提示：有一边的三个点几乎在一条线上（那个方向没量到范围），"
                    "\n             这个方向已按默认尺度处理。想更贴合你的手，"
                    "重跑一次并把三个点分开些。",
                )
    if profile is not None:
        print(
            f"[速率] 中位 {profile.p50:.2f}、p90 {profile.p90:.2f} 画面宽/秒"
            f"（{profile.label}，{profile.samples} 个采样）→ 已记录",
        )

    print("\n[练习] 接下来每种手势做一遍，屏幕上有提示语。")
    # 🔴 **必须在已有样本库上追加，不能新建**：作者自己机器上有 889 帧标注样本，
    # 早期版本这里传 dataset=None，等于跑一次向导就把整库替换成 6×5 秒的练习帧
    # （`_save` 会覆盖同一个文件）。老用户重跑一次向导就丢光历史数据。
    existing_path = default_dataset_path()
    existing = (
        GestureDataset.load(existing_path)
        if existing_path.exists()
        else GestureDataset()
    )
    if len(existing):
        print(f"[练习] 已载入你原有的 {len(existing)} 帧样本，本次追加。")
    dataset, aborted, coverage = run_collect(
        config,
        steps=PRACTICE_STEPS,
        dataset=existing,
        # 🔴 每种手势先出题目、等捏合确认再录（用户反馈：倒计时 2 秒看不清题目）
        confirm_before_record=True,
    )
    if aborted:
        print("[设置] 练习被中断，样本仍然会保存。")

    if not len(dataset):
        print("[设置] 没采到样本，未写盘。")
        return 1

    out = _save(config, dataset, points, profile)
    print(f"\n[设置] 已保存 {len(dataset)} 帧样本 + 配置 + 记录 → {out}")
    print("       以后想补录，跑 python -m handmouse.collect --append")
    _report(dataset, coverage, collected=[step.label for step in PRACTICE_STEPS])
    return 0


def run_calibration_only(config: AppConfig) -> int:
    """只重做标定（3 点 + 速率），**不录手势、不动样本库**。

    用途：换了摄像头 / 桌面上手的位置变了，只想把活动范围重新对一遍，
    不想再花五分钟把七种手势录一遍。

    与 `run_onboarding` 的区别只有一条：不调 `run_collect`，所以
    `gesture_data/samples.jsonl` 一个字节都不会变；`onboarding.json` 里的
    舒适点与速率会被更新（这是"重新标定"的全部意义）。
    """

    print("=" * 62)
    print(f"{APP_NAME} 重新标定（不动样本库）")
    print("=" * 62)
    print("画面里会出现 3 个十字，依次是屏幕中央 / 左上 / 右下。")
    print("十字只表示屏幕方位 —— 手放在你最舒服的位置，捏合一下即可。")
    print()

    points, profile, aborted = _calibrate(config)
    if aborted:
        print("[标定] 已中断，什么都没改。")
        return 1
    if not points:
        print("[标定] 没读到捏合动作，活动范围保持原样。")
        return 1

    margins = _apply_margins(config, points)
    config.onboarding_done = True
    config.validate()
    ConfigStore().save(config)

    listed = "、".join(f"({item.x:.2f}, {item.y:.2f})" for item in points)
    print(f"[标定] {len(points)} 个点：{listed}")
    if margins is None:
        print("       三点都贴近画面中央，活动范围保持默认。")
    else:
        print(
            f"       活动区内缩 left={margins[0]:.3f} top={margins[1]:.3f} "
            f"right={margins[2]:.3f} bottom={margins[3]:.3f}",
        )
    if profile is not None:
        print(
            f"[速率] 中位 {profile.p50:.3f}、p90 {profile.p90:.3f} 画面宽/秒"
            f"（{profile.label}）→ 已记录",
        )

    # 只更新 onboarding.json 里的标定部分，样本库统计保持原样
    path = default_dataset_path().with_name("onboarding.json")
    record: dict = {}
    if path.exists():
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {}
    record.update(
        {
            "recalibrated_ns": time.time_ns(),
            "comfort_points": [asdict(item) for item in points],
            "speed": asdict(profile) if profile else record.get("speed"),
            "margins": (
                {
                    "left": margins[0],
                    "top": margins[1],
                    "right": margins[2],
                    "bottom": margins[3],
                }
                if margins
                else record.get("margins")
            ),
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[标定] 已写入 {path}")
    print("       想连手势样本一起重录，用 python -m handmouse --onboarding")
    return 0


def main(argv: list[str] | None = None) -> int:
    config = ConfigStore().load()
    return run_onboarding(config)


if __name__ == "__main__":
    # 别删：`python -m handmouse.onboarding` 是向导中断时打印给用户的复跑入口，
    # 少了这个守卫，那条命令会静默什么都不做（跑一遍才发现）。
    raise SystemExit(main())
