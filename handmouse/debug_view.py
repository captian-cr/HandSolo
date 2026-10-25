"""OpenCV 调试窗口。

刻意不使用 PySide6：MVP 阶段只需要"看得见跟踪质量"和"按键开关接管"，
一个 ``cv2.imshow`` 窗口就够了，省掉一整套 GUI 依赖和打包体积。将来真要做
托盘/设置面板，再按 ``handmouse[ui]`` 单独加。

窗口内容：
    * 手部骨架 + 食指尖高亮
    * 状态行（跟踪中 / 已暂停）
    * 捏合比与阈值、FPS、推理延迟、丢帧、CPU/内存
    * **指伸四位数**（食指/中指/无名指/小指，1=伸直）—— 2026-10-22 用户反复
      报"只伸食指却不识别"，光看 HUD 无法判断是判据错还是他手型特殊；把
      逐指伸直结果摊开显示，他和排查都能一眼对上。
    * ScreenMap 摘要（DPI / 显示器数 / 注入坐标系）
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from handmouse import APP_NAME
from handmouse.contracts import (
    DetectionPacket,
    FramePacket,
    GestureType,
    PipelineSnapshot,
)
from handmouse.gestures import (
    _FINGER3D_STRAIGHT_MIN,
    _finger3d_feats,
    _finger_extensions,
    _two_finger_diag,
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

#: OpenCV 窗口标题必须用 ASCII。
#:
#: 原因不是没设对，而是 cv2 在 Windows 上把标题当**本地代码页（GBK）**字节串
#: 传给系统：Python 侧的 UTF-8 中文会被按 GBK 解释，显示成"鎵嬪娔杩借釜"这种
#: 乱码。想用 SetWindowTextW 改成宽字符也做不到——OpenCV 靠**标题**查找已有
#: 窗口，标题一变它下次 imshow 就找不到、会重建窗口，每帧闪一次。
#: 所以这里只能老老实实用英文。
WINDOW_NAME = f"{APP_NAME} - Pointer"

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
            # 只留一个小红点标出食指尖。原来还有一圈 11px 的品红环，
            # 叠在手指上看着就像一颗"悬浮球"，用户会误以为是光标。
            cv2.circle(frame, pixel, 4, (0, 0, 255), -1, cv2.LINE_AA)
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
    fingers_text = "-"
    fingers3d_text = "-"
    swipe3d_text = ""
    diag_text = ""
    if detection and detection.observations:
        observation = detection.observations[0]
        if len(observation.landmarks) == 21:
            draw_skeleton(view, observation.landmarks)
            # 现算而不是读状态机：状态机里的 five_open 已经过长度闸+时序确认，
            # 看不出"原始角度判据"认为哪几根手指是伸着的。排查时需要的正是
            # 原始值——"指伸 1111 但状态是控制光标"= 长度闸在起作用；
            # "指伸 1000 却进了待机"= 判据以外的问题。
            #
            # 🔴 但**别把它当真相**：2026-10-22 用他 3378 帧量过，这个角度判据
            # 在"只伸食指"的帧里也报"食指+中指伸出"的比例高达 **52.3%**
            # （侧视下收拢指的关节投影近似共线）。它只能当粗参考，
            # 真正的判定看下面那行 `双指判据` 的数值。
            ext = _finger_extensions(observation.landmarks)
            fingers_text = "".join("1" if value else "0" for value in ext)

            peak, ti, tm, tr, tp, angle = _two_finger_diag(observation.landmarks)
            diag_text = (
                f"双指判据 peak {peak:.2f} i {ti:.2f} m {tm:.2f} "
                f"r {tr:.2f} p {tp:.2f} a {angle:.0f}度"
                f"  阈值 peak>=0.80 i,m>=0.70 r,p<=0.80 a>=110"
            )
        # 3D 版（米制世界坐标）。**必须和 2D 那行并排显示**：两行不一致时，
        # 就是"归一化坐标宽高比 / 透视压缩把判据带偏了"的直接证据。
        if len(observation.world_landmarks) == 21:
            feats3d = _finger3d_feats(observation.world_landmarks)
            fingers3d_text = "".join(
                "1" if feats3d[name] >= _FINGER3D_STRAIGHT_MIN else "0"
                for name in ("index", "middle", "ring", "pinky")
            )
            # 数值也铺开：只给 0/1 看不出"差多少才过线"。四位分别是
            # 食/中/无名/小指的 3D 伸直度，过线阈值 0.97（无名/小指收拢线 0.94）。
            swipe3d_text = "/".join(
                f"{feats3d[n]:.2f}" for n in ("index", "middle", "ring", "pinky")
            )

    status_text, status_color = _status_line(snapshot)
    metrics = snapshot.metrics
    ratio_text = f"{pinch_ratio:.2f}" if pinch_ratio is not None else "--"
    # 停住阈值必须从 config 现读：曾经写死 0.85，改了 config.json 后 HUD
    # 仍显示旧值，用户照着 HUD 调参会越调越偏。
    stop_threshold = getattr(config, "thumb_stop_threshold", 0.85)
    if not getattr(config, "thumb_stop_enabled", True):
        stop_text = "停-关"
    else:
        stop_text = f"停≥{stop_threshold:.2f}"
        if snapshot.thumb_stop:
            # 触发时给个明确回执：否则用户只能靠"光标有没有动"猜手感。
            stop_text += " ●已停住"
    # 锚定回执必须独立于 thumb_stop 显示：2026-10-21 起 thumb_stop 已关闭，
    # 改为「捏合趋向」触发锚定，用户要靠这个标记确认预冻结有没有生效。
    if snapshot.pinch_anchor:
        stop_text += " ⚓锚定中"
    # 瞄准死区回执：让用户能区分"光标不动是因为锁住了"和"卡死了"。
    if snapshot.micro_lock:
        stop_text += " ⌖瞄准锁"
    # 停留回执（2026-10-23 加）：磁吸窗口的唯一入口就是它，而它此前
    # **一次都没亮过**（阈值 0.005 物理不可达）。这里把"已停稳多久 / 要多久"
    # 直接铺开，否则"停住一秒没反应"只能靠猜。
    dwell_need = max(1, int(getattr(config, "pinch_pre_dwell_ms", 1000)))
    if snapshot.dwell_active:
        stop_text += " ◎停留OK"
    elif snapshot.dwell_ms >= 50:
        stop_text += f" 停留{snapshot.dwell_ms / 1000:.1f}/{dwell_need / 1000:.1f}s"

    # 四边内缩每帧只算一次（HUD 要用四个值 + 两个行程）。
    _mg, _ml, _mr, _mb = config.effective_margins()

    rows: list[tuple[str, tuple[int, int, int]]] = [
        (status_text, status_color),
        (
            f"捏合比 {ratio_text}  拇指开 {metrics.thumb_spread_percent/100:.2f}"
            f"({stop_text})  阈值 on<{config.pinch_engage:.2f} "
            f"off>{config.pinch_release:.2f}  FPS {metrics.capture_fps:.0f}/"
            f"{metrics.inference_fps:.0f}  延迟 p50 {metrics.latency_p50_ms:.0f}ms "
            f"p95 {metrics.latency_p95_ms:.0f}ms",
            WHITE,
        ),
        (
            f"丢帧 {metrics.dropped_frames}  检测率 {metrics.hand_detection_rate:.0f}%  "
            f"CPU {metrics.cpu_percent:.0f}%  内存 {metrics.memory_mb:.0f}MB"
            f"  指伸角 {fingers_text}  指伸3D {fingers3d_text} {swipe3d_text}"
            + (f"  手势库 {snapshot.model_note}" if snapshot.model_note else ""),
            GRAY,
        ),
        (diag_text or "双指判据 --", GRAY),
        (snapshot.ball_note or "球 --", GRAY),
        (f"磁吸 {snapshot.snap_target_name or '--'}   {screen_map.describe()}", GRAY),
        (
            f"内缩 上{_ml:.2f} 下{_mb:.2f} 左{_mg:.2f} 右{_mr:.2f}"
            f"（手行程 {1 - _mg - _mr:.0%}×{1 - _ml - _mb:.0%}）"
            f"  静稳 {config.pointer_stability:.2f}"
            f"  动稳 {config.pointer_response:.2f}"
            f"   [ ] 灵敏   , . 静止   < > 大动作",
            GRAY,
        ),
        (
            "M 暂停/接管  S 安全模式  Ctrl+Alt+Esc 急停  ESC/Q 退出",
            GRAY,
        ),
    ]
    # 样本库的逐帧判定单独占一行：光看"五指张开"这个最终状态没法判断到底
    # 是模型说的还是几何判据说的，加这行才能一眼定位。
    if snapshot.model_debug:
        rows.insert(3, (f"模型 {snapshot.model_debug}", GRAY))
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


def present(window_name: str, frame: np.ndarray | None, scale: float) -> int:
    """按 DPI 缩放后显示，并返回按键码（无按键返回 -1）。

    ``frame=None`` 表示"这一轮不重绘、只喂按键"：主循环在摄像头没来新帧时走这条
    路径，既省掉 build_view + imshow 的开销（原本主循环会限流到约 113 次/秒，
    其中约 3/4 是在把同一帧重复画），又不会让按键失灵。
    窗口尚未建立时（摄像头刚打开还没显示过任何一帧）也走这里，
    因为 ``cv2.imshow`` 要求窗口已存在，而 ``waitKey`` 不要求。
    """

    global _window_opened
    if frame is None:
        if not _window_opened:
            return -1
        return cv2.waitKey(1) & 0xFF
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
