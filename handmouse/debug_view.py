"""OpenCV 调试窗口。

刻意不使用 PySide6：MVP 阶段只需要"看得见跟踪质量"和"按键开关接管"，
一个 ``cv2.imshow`` 窗口就够了，省掉一整套 GUI 依赖和打包体积。将来真要做
托盘/设置面板，再按 ``handmouse[ui]`` 单独加。

窗口内容：
    * 手部骨架 + 食指尖高亮
    * 状态行（跟踪中 / 已暂停 / 轨迹手势阶段）
    * 捏合比与阈值、FPS、推理延迟、丢帧、CPU/内存
    * 轨迹手势进行中的原始笔迹与 DTW 候选
    * ScreenMap 摘要（DPI / 显示器数 / 注入坐标系）
"""

from __future__ import annotations

import cv2
import numpy as np

from handmouse.contracts import (
    DetectionPacket,
    FramePacket,
    GestureType,
    PipelineSnapshot,
)
from handmouse.motion_gestures import MotionOutcome
from handmouse.screen_map import ScreenMap

WINDOW_NAME = "HandMouse - 手势追踪"

WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
GREEN = (120, 220, 120)
YELLOW = (80, 200, 240)
RED = (90, 90, 235)
CYAN = (230, 210, 90)
MAGENTA = (200, 120, 240)

# MediaPipe 21 点连线（与 Vimoose 的手绘骨架一致，不依赖 mediapipe 的绘图工具）
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _status_line(snapshot: PipelineSnapshot, motion: MotionOutcome | None) -> tuple[str, tuple[int, int, int]]:
    if motion is not None and motion.active:
        if motion.state.value == "arming" or motion.gesture == GestureType.MOTION_ARMED:
            return (motion.message or "轨迹手势就绪", CYAN)
        if motion.gesture == GestureType.MOTION_DRAWING:
            return (motion.message or "绘制中", CYAN)
        if motion.gesture == GestureType.MOTION_CANDIDATE:
            return (motion.message or "等待确认", MAGENTA)
    if snapshot.camera_error:
        return (f"错误：{snapshot.camera_error}", RED)
    if snapshot.paused:
        return ("控制已暂停（Ctrl+Alt+G 或 M 恢复）", YELLOW)
    if snapshot.gesture == GestureType.LOST:
        return ("未检测到有效手部", YELLOW)
    return (snapshot.message or "跟踪中", GREEN)


def draw_skeleton(frame: np.ndarray, points) -> None:
    """在帧上画骨架与指尖高亮（原地修改）。"""

    pixels = [
        (int(round(point.x * frame.shape[1])), int(round(point.y * frame.shape[0])))
        for point in points
    ]
    for start, end in HAND_CONNECTIONS:
        if start < len(pixels) and end < len(pixels):
            cv2.line(frame, pixels[start], pixels[end], (0, 200, 0), 2, cv2.LINE_AA)
    for index, pixel in enumerate(pixels):
        if index == 8:
            cv2.circle(frame, pixel, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pixel, 11, (255, 0, 255), 2, cv2.LINE_AA)
        else:
            cv2.circle(frame, pixel, 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_trajectory(frame: np.ndarray, outcome: MotionOutcome) -> None:
    """把绘制中的笔迹画回画面，便于确认"手画的和模板差多少"。"""

    if not outcome.drawing:
        return
    height, width = frame.shape[:2]
    pixels = [
        (int(round(point[0] * width)), int(round(point[1] * height)))
        for point in outcome.drawing
    ]
    for start, end in zip(pixels, pixels[1:]):
        cv2.line(frame, start, end, CYAN, 2, cv2.LINE_AA)
    cv2.circle(frame, pixels[-1], 5, MAGENTA, -1, cv2.LINE_AA)


def draw_hud(
    frame: np.ndarray,
    rows: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """带半透明底衬的文字面板，保证在白墙/暗背景上都读得清。"""

    line_height = 26
    padding = 10
    width = max(
        cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0]
        for text, _color in rows
    ) + padding * 2
    height = line_height * len(rows) + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (32, 32, 32), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for index, (text, color) in enumerate(rows):
        origin = (8 + padding, 8 + padding + line_height * (index + 1) - 8)
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )


def build_view(
    frame: np.ndarray,
    snapshot: PipelineSnapshot,
    detection: DetectionPacket | None,
    motion: MotionOutcome | None,
    screen_map: ScreenMap,
    config,
    pinch_ratio: float | None = None,
) -> np.ndarray:
    """在采集帧上叠加所有调试信息，返回可直接 imshow 的画面。"""

    view = frame.copy()
    if detection and detection.observations:
        observation = detection.observations[0]
        if len(observation.landmarks) == 21:
            draw_skeleton(view, observation.landmarks)
    if motion is not None:
        draw_trajectory(view, motion)

    status_text, status_color = _status_line(snapshot, motion)
    metrics = snapshot.metrics
    ratio_text = f"{pinch_ratio:.2f}" if pinch_ratio is not None else "--"
    candidates = " / ".join(snapshot.motion_candidates) or "--"

    rows: list[tuple[str, tuple[int, int, int]]] = [
        (status_text, status_color),
        (
            f"捏合比 {ratio_text}  阈值 on<{config.pinch_engage:.2f} "
            f"off>{config.pinch_release:.2f}  FPS {metrics.capture_fps:.0f}/"
            f"{metrics.inference_fps:.0f}  延迟 p50 {metrics.latency_p50_ms:.0f}ms "
            f"p95 {metrics.latency_p95_ms:.0f}ms",
            WHITE,
        ),
        (
            f"丢帧 {metrics.dropped_frames}  检测率 {metrics.hand_detection_rate:.0f}%  "
            f"CPU {metrics.cpu_percent:.0f}%  内存 {metrics.memory_mb:.0f}MB",
            GRAY,
        ),
        (f"DTW 候选 {candidates}   待确认 {snapshot.motion_pending or '--'}", CYAN),
        (f"最近动作 {snapshot.last_action or '--'}", GRAY),
        (f"磁吸 {snapshot.snap_target_name or '--'}   {screen_map.describe()}", GRAY),
        ("M 暂停/恢复(接管)  S 安全模式数据  Ctrl+Alt+Esc 急停  ESC/Q 退出", GRAY),
    ]
    draw_hud(view, rows)
    return view


# 摄像头打不开时主循环一帧都没显示过，窗口并不存在；此时无条件调用
# cvDestroyWindow 会抛 "NULL window"。用一个标志记住窗口是否真的建过。
_window_opened = False


def present(window_name: str, frame: np.ndarray, scale: float) -> int:
    """按 DPI 缩放后显示，并返回按键码（无按键返回 -1）。"""

    global _window_opened
    shown = frame
    if scale > 1.01:
        shown = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
    cv2.imshow(window_name, shown)
    _window_opened = True
    return cv2.waitKey(1) & 0xFF


def close(window_name: str = WINDOW_NAME) -> None:
    global _window_opened
    if not _window_opened:
        return
    _window_opened = False
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
