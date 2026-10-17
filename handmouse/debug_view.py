"""OpenCV 调试窗口。

刻意不使用 PySide6：MVP 阶段只需要"看得见跟踪质量"和"按键开关接管"，
一个 ``cv2.imshow`` 窗口就够了，省掉一整套 GUI 依赖和打包体积。将来真要做
托盘/设置面板，再按 ``handmouse[ui]`` 单独加。

窗口内容：
    * 手部骨架 + 食指尖高亮
    * 状态行（跟踪中 / 已暂停）
    * 捏合比与阈值、FPS、推理延迟、丢帧、CPU/内存
    * ScreenMap 摘要（DPI / 显示器数 / 注入坐标系）
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from handmouse.contracts import (
    DetectionPacket,
    FramePacket,
    GestureType,
    PipelineSnapshot,
)
from handmouse.screen_map import ScreenMap

# cv2.FONT_HERSHEY_SIMPLEX 不含中文字形，HUD 文字改由 PIL 绘制。
# 按 微软雅黑 → 黑体 → 宋体 顺序加载系统字体，全部缺失时回退 PIL
# 默认字体（回退态中文仍会乱码，但至少导入不崩、HUD 其余信息可读）。
def _load_hud_font(size: int):
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


_HUD_FONT = _load_hud_font(16)

WINDOW_NAME = "HandMouse - 手势追踪"

WHITE = (245, 245, 245)
GRAY = (170, 170, 170)
GREEN = (120, 220, 120)
YELLOW = (80, 200, 240)
RED = (90, 90, 235)

# MediaPipe 21 点连线（与 Vimoose 的手绘骨架一致，不依赖 mediapipe 的绘图工具）
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def _status_line(snapshot: PipelineSnapshot) -> tuple[str, tuple[int, int, int]]:
    if snapshot.camera_error:
        return (f"错误：{snapshot.camera_error}", RED)
    if snapshot.paused:
        return ("控制已暂停（Ctrl+Alt+G 或 M 恢复）", YELLOW)
    if snapshot.gesture == GestureType.LOST:
        return ("未检测到有效手部", YELLOW)
    if snapshot.gesture == GestureType.GATE_IDLE:
        return ("未进入控制态：掌心朝镜 + 五指张开 保持 0.3s", YELLOW)
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


def draw_hud(
    frame: np.ndarray,
    rows: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """带半透明底衬的文字面板，保证在白墙/暗背景上都读得清。"""

    line_height = 24
    padding = 10
    bboxes = [_HUD_FONT.getbbox(text) for text, _ in rows]
    width = max(bbox[2] - bbox[0] for bbox in bboxes) + padding * 2
    height = line_height * len(rows) + padding * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + width, 8 + height), (32, 32, 32), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # PIL 用 RGB，cv2 是 BGR；文字绘制走 PIL，底衬已用 cv2 画好。
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for index, (text, color) in enumerate(rows):
        y = 8 + padding + line_height * index
        # cv2 color 是 BGR，PIL 需要 RGB
        r, g, b = color[2], color[1], color[0]
        draw.text((8 + padding, y), text, font=_HUD_FONT, fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def build_view(
    frame: np.ndarray,
    snapshot: PipelineSnapshot,
    detection: DetectionPacket | None,
    screen_map: ScreenMap,
    config,
    pinch_ratio: float | None = None,
    banner: tuple[str, tuple[int, int, int]] | None = None,
) -> np.ndarray:
    """在采集帧上叠加所有调试信息，返回可直接 imshow 的画面。

    ``banner`` 是可选的首行强提示（gate-diag 预检/中止用），渲染在常规
    HUD 行之前，颜色由调用方给定。
    """

    view = frame.copy()
    if detection and detection.observations:
        observation = detection.observations[0]
        if len(observation.landmarks) == 21:
            draw_skeleton(view, observation.landmarks)

    status_text, status_color = _status_line(snapshot)
    metrics = snapshot.metrics
    ratio_text = f"{pinch_ratio:.2f}" if pinch_ratio is not None else "--"

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
        (f"磁吸 {snapshot.snap_target_name or '--'}   {screen_map.describe()}", GRAY),
        ("M 暂停/恢复(接管)  S 安全模式数据  Ctrl+Alt+Esc 急停  ESC/Q 退出", GRAY),
    ]
    if banner is not None:
        rows.insert(0, banner)
    # VERBOSE_GATE 下追加引擎/采集层 debug 行（灰色小字）。
    for line in snapshot.debug_lines:
        rows.append((line, GRAY))
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
