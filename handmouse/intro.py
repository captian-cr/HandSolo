"""开屏动效：眼睛扫描 → 三环旋转 → 收束成一个点。

设计意图
--------
只在**第一次启动**、且第一次进入控制态时播放 3.8 秒。结尾那个点落在屏幕
正中，紧接着系统光标出现在同一位置——视觉上是"这一点变成了鼠标"。

为什么不用纯黑背景
------------------
用户明确要求"屏幕渐暗一点就可以，不要黑背景"。所以整窗用 ``-alpha`` 做半透明
压暗（约 55%），图形本身也跟着半透明——发光本来就该是透的，正合适。
这样既能看清中央的眼睛，又不会把桌面整个盖掉。

为什么不用色键透明
------------------
色键是"某种颜色整块变透明"，做不出"发光渐隐到背景"的柔和边缘，会留下硬边、
黑边。这里要的是整体半透明，所以走 ``-alpha`` 而不是 colorkey。

坐标与性能
----------
只渲染中央 ``440×440`` 一块，四周是纯色填充，视觉上无缝。全屏 1920×1080
每帧重画 + 多次高斯模糊扛不住，缩到 440 后每帧约 20ms，跑 20fps 有余量。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


#: 动效总时长（秒）。
TOTAL_SECONDS = 1.0
#: 中央渲染区域边长（像素）。
CANVAS = 440
#: 整窗压暗程度：0=完全透明，1=全黑。
DIM_ALPHA = 0.55
#: 名义帧率。1 秒的动效必须足够密，否则会看出跳帧。
#: 实际由主循环推进（渲染约 22ms + 10ms 让步 ≈ 30fps）。
FPS = 30
#: 背景色，与窗口底色一致才能无缝。
BACKDROP = (5, 7, 8)

_CENTER = CANVAS // 2


def _cubic(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 90,
) -> list[tuple[float, float]]:
    """三次贝塞尔采样成折线。"""

    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = index / steps
        u = 1.0 - t
        x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0]
        y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
        points.append((x, y))
    return points


def _ellipse(
    rx: float,
    ry: float,
    angle: float,
    steps: int = 140,
) -> list[tuple[float, float]]:
    """以画布中心为原点的椭圆，``angle`` 为绕中心的旋转弧度。"""

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    points: list[tuple[float, float]] = []
    for index in range(steps + 1):
        t = 2.0 * math.pi * index / steps
        x = rx * math.cos(t)
        y = ry * math.sin(t)
        points.append((_CENTER + x * cos_a - y * sin_a, _CENTER + x * sin_a + y * cos_a))
    return points


def _segment_subset(
    points: list[tuple[float, float]],
    ratio: float,
    from_middle: bool = True,
) -> list[tuple[float, float]]:
    """按比例截取折线的一段。``from_middle`` 时从中间向两端扩展。"""

    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.0:
        return []
    if not from_middle:
        return points[: max(2, int(len(points) * ratio))]
    half = len(points) // 2
    span = int(half * ratio)
    if span <= 0:
        return []
    return points[half - span : half + span]


def _glow(
    base: Image.Image,
    points: list[tuple[float, float]],
    outer: tuple[int, int, int],
    inner: tuple[int, int, int],
    width: int,
    blur: float,
    opacity: float,
) -> None:
    """叠加一条发光线：**一次模糊同时得到外晕和亮芯**。

    旧写法是"粗线模糊一次 + 细线单独一次"，一条线两次高斯模糊，一帧十几条
    线就是十几次模糊——用户实测"非常卡"。改成：粗线画进图层 → 模糊一次 →
    在**同一图层**上再画细线（细线不改模糊结果，因为模糊已经做完了）。
    视觉效果等价，模糊次数减半。
    """

    if len(points) < 2 or opacity <= 0.0:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = int(255 * max(0.0, min(1.0, opacity)))
    draw.line(points, fill=(*outer, int(alpha * 0.85)), width=width)
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    core = ImageDraw.Draw(layer)
    core.line(points, fill=(*inner, alpha), width=max(2, width // 3))
    base.alpha_composite(layer)


def _glow_ring(
    base: Image.Image,
    points: list[tuple[float, float]],
    outer: tuple[int, int, int],
    inner: tuple[int, int, int],
    width: int,
    blur: float,
    opacity: float,
) -> None:
    """同 ``_glow``，只是闭合起来。"""

    if len(points) < 2 or opacity <= 0.0:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = int(255 * max(0.0, min(1.0, opacity)))
    closed = [*points, points[0]]
    draw.line(closed, fill=(*outer, int(alpha * 0.85)), width=width, joint="curve")
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    core = ImageDraw.Draw(layer)
    core.line(closed, fill=(*inner, alpha), width=max(2, width // 3), joint="curve")
    base.alpha_composite(layer)


def _disc(
    base: Image.Image,
    center: tuple[float, float],
    radius: float,
    color: tuple[int, int, int],
    blur: float,
    opacity: float,
) -> None:
    if radius <= 0.5 or opacity <= 0.0:
        return
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    alpha = int(255 * max(0.0, min(1.0, opacity)))
    x, y = center
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(*color, alpha))
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    base.alpha_composite(layer)


@dataclass(frozen=True, slots=True)
class _Phase:
    """把总时长换算成各阶段进度用的区间。"""

    start: float
    end: float

    def at(self, t: float) -> float:
        if t <= self.start:
            return 0.0
        if t >= self.end:
            return 1.0
        return (t - self.start) / (self.end - self.start)


# 时间轴（秒）。改节奏只动这里。
# 1.0 秒版：太快了，扫描只扫一遍（两遍会糊成一片），其余各段压到极限但仍重叠。
_DRAW = _Phase(0.00, 0.30)     # 眼睑张开
_IRIS = _Phase(0.15, 0.34)     # 虹膜 + 瞳孔
_SCAN = _Phase(0.25, 0.50)     # 扫描一遍
_CROSS = _Phase(0.40, 0.52)    # 十字准线闪过
_EYE_OUT = _Phase(0.46, 0.58)  # 眼睛退场
_ORBIT = _Phase(0.50, 0.76)    # 三环展开
_COLLAPSE = _Phase(0.76, 0.90) # 收束
_POINT = _Phase(0.90, 1.00)    # 成点


def _ease_out(value: float) -> float:
    return 1.0 - (1.0 - value) ** 3


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def render_frame(t: float) -> Image.Image:
    """渲染 ``t`` 秒处的画面。"""

    frame = Image.new("RGBA", (CANVAS, CANVAS), (*BACKDROP, 255))
    eye_scale = 1.0 - 0.14 * _ease_out(_EYE_OUT.at(t))
    eye_alpha = 1.0 - _ease_out(_EYE_OUT.at(t))

    # ---- 眼睛轮廓：从中间向两端张开 ----
    draw_ratio = _ease_out(_DRAW.at(t))
    if draw_ratio > 0.0 and eye_alpha > 0.01:
        half_w = 210.0 * eye_scale
        half_h = 130.0 * eye_scale
        top = _cubic(
            (_CENTER - half_w, _CENTER),
            (_CENTER - half_w * 0.62, _CENTER - half_h),
            (_CENTER + half_w * 0.62, _CENTER - half_h),
            (_CENTER + half_w, _CENTER),
        )
        bottom = _cubic(
            (_CENTER + half_w, _CENTER),
            (_CENTER + half_w * 0.62, _CENTER + half_h),
            (_CENTER - half_w * 0.62, _CENTER + half_h),
            (_CENTER - half_w, _CENTER),
        )
        upper = _segment_subset(top, draw_ratio)
        lower = _segment_subset(bottom, draw_ratio)
        for points in (upper, lower):
            if len(points) < 2:
                continue
            _glow(
                frame, points, (8, 66, 205), (62, 142, 238),
                24, 7.0, 0.95 * eye_alpha * draw_ratio,
            )

    # ---- 虹膜与瞳孔 ----
    iris_progress = _ease_out(_IRIS.at(t))
    if iris_progress > 0.05 and eye_alpha > 0.01:
        radius = 44.0 * eye_scale * iris_progress
        iris = _ellipse(radius, radius * 0.92, 0.0)
        _glow_ring(frame, iris, (12, 90, 222), (78, 155, 242), 10, 3.0, 0.95 * eye_alpha)
        pupil = 13.0 * eye_scale * iris_progress
        _disc(frame, (_CENTER, _CENTER), pupil * 1.8, (70, 170, 255), 8.0, 0.70 * eye_alpha)
        _disc(frame, (_CENTER, _CENTER), pupil, (235, 250, 255), 1.0, 0.98 * eye_alpha)

    # ---- 十字准线 ----
    cross_progress = _ease_out(_CROSS.at(t))
    if cross_progress > 0.01 and eye_alpha > 0.01:
        reach = 150.0 * cross_progress
        gap = 140.0
        for x1, y1, x2, y2 in (
            (_CENTER, _CENTER - gap, _CENTER, _CENTER - gap - reach),
            (_CENTER, _CENTER + gap, _CENTER, _CENTER + gap + reach),
            (_CENTER - gap, _CENTER, _CENTER - gap - reach, _CENTER),
            (_CENTER + gap, _CENTER, _CENTER + gap + reach, _CENTER),
        ):
            _glow(
                frame, [(x1, y1), (x2, y2)], (14, 92, 224), (82, 160, 244),
                7, 2.0, 0.95 * eye_alpha,
            )

    # ---- 扫描线：在眼睛范围内上下扫两遍 ----
    scan_progress = _SCAN.at(t)
    if 0.02 < scan_progress < 0.98 and eye_alpha > 0.05:
        sweep = math.sin(scan_progress * math.pi * 2.0)
        scan_y = _CENTER + sweep * 96.0 * eye_scale
        half_span = 165.0 * eye_scale
        fade = min(1.0, (1.0 - abs(sweep)) * 3.0)
        _glow(
            frame,
            [(_CENTER - half_span, scan_y), (_CENTER + half_span, scan_y)],
            (18, 100, 228),
            (120, 190, 250),
            6,
            3.0,
            1.00 * eye_alpha * fade,
        )

    # ---- 三环：展开 → 旋转 → 收束 ----
    orbit_in = _ease_out(_ORBIT.at(t))
    collapse = _ease_out(_COLLAPSE.at(t))
    if orbit_in > 0.01 and collapse < 0.99:
        ring_scale = _lerp(0.55, 1.0, orbit_in) * (1.0 - collapse * 0.94)
        ring_alpha = min(1.0, orbit_in * 1.4) * (1.0 - collapse * 0.58)
        if ring_scale > 0.02:
            spin = t * 0.55
            for index, (rx, ry, base_angle, color) in enumerate(
                (
                    (196.0, 60.0, 0.00, (10, 92, 226)),
                    (196.0, 60.0, 1.05, (16, 66, 212)),
                    (196.0, 60.0, 2.09, (24, 128, 236)),
                ),
            ):
                angle = base_angle + spin * (1.0 + index * 0.35) * (1.0 + collapse * 1.6)
                ring = _ellipse(rx * ring_scale, ry * ring_scale, angle)
                _glow_ring(frame, ring, color, (96, 176, 248), 9, 3.2, 0.85 * ring_alpha)

    # ---- 收束成一个点 ----
    point_progress = _POINT.at(t)
    if point_progress > 0.0:
        if point_progress < 0.55:
            burst = point_progress / 0.55
            _disc(frame, (_CENTER, _CENTER), 8.0 + 34.0 * burst, (120, 210, 255), 14.0, 0.9 * (1.0 - burst))
        hold = 1.0 - max(0.0, (point_progress - 0.55) / 0.45)
        _disc(frame, (_CENTER, _CENTER), 11.0, (130, 215, 255), 12.0, 1.0 * hold)
        _disc(frame, (_CENTER, _CENTER), 4.2, (250, 254, 255), 0.8, hold)

    return frame.convert("RGB")


def default_marker() -> Path:
    """首播标记文件：存在即表示已经播过。"""

    return Path(__file__).resolve().parent.parent / "gesture_data" / ".intro_played"


def should_play(marker: Path | None = None) -> bool:
    path = marker or default_marker()
    return not path.exists()


def mark_played(marker: Path | None = None) -> None:
    path = marker or default_marker()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(int(time.time())), encoding="utf-8")
    except OSError:
        pass


def _make_click_through(root) -> None:
    """加上 WS_EX_TRANSPARENT，让这层不吞掉点击。

    透明交给 tkinter 的 ``-alpha``（整窗半透明），所以这里只补穿透位，
    不设色键——两个一起设会互相覆盖。
    """

    import ctypes

    from handmouse.grid_ui import _hwnd_of

    user32 = ctypes.windll.user32
    hwnd = _hwnd_of(root)
    style = user32.GetWindowLongW(hwnd, -20)  # GWL_EXSTYLE
    user32.SetWindowLongW(
        hwnd,
        -20,
        style | 0x00080000 | 0x00000020,  # LAYERED | TRANSPARENT
    )


class IntroPlayer:
    """全屏半透明压暗层 + 中央渲染的开屏动效。

    **非阻塞**：由主循环每帧调 :meth:`tick`。开屏只有三秒多，但阻塞主循环会
    让摄像头停摆、按键失联，所以做成逐帧推进。
    """

    def __init__(self, *, alpha: float = DIM_ALPHA) -> None:
        import tkinter as tk

        from PIL import ImageTk

        self._tk = tk
        self._image_tk = ImageTk
        self._alpha = max(0.0, min(1.0, alpha))
        self._started: float | None = None
        self._photo = None
        self._last_alpha: float | None = None
        self.failed = False

        self.root = tk.Tk()
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        background = "#%02x%02x%02x" % BACKDROP
        self.root.configure(bg=background)
        self.label = tk.Label(
            self.root,
            bd=0,
            highlightthickness=0,
            bg=background,
        )
        self.label.place(relx=0.5, rely=0.5, anchor="center")
        self.root.attributes("-alpha", 0.0)

    def start(self) -> bool:
        try:
            self.root.update_idletasks()
            self.root.deiconify()
            self.root.lift()
            _make_click_through(self.root)
            self.root.update()
            return True
        except Exception:
            self.failed = True
            self.close()
            return False

    def tick(self) -> bool:
        """推进一帧。返回 False 表示已经播完（或出错），调用方应关闭它。"""

        now = time.monotonic()
        if self._started is None:
            self._started = now
        elapsed = now - self._started
        if elapsed >= TOTAL_SECONDS:
            return False

        fade = 0.10
        if elapsed < fade:
            alpha = self._alpha * (elapsed / fade)
        elif elapsed > TOTAL_SECONDS - 0.16:
            alpha = self._alpha * max(0.0, (TOTAL_SECONDS - elapsed) / 0.16)
        else:
            alpha = self._alpha
        try:
            # **只在透明度真的变化时才设**。全屏窗口每改一次 -alpha，Windows
            # 就要把整块屏幕重新合成一遍——每帧无脑设置是这个动效卡顿的
            # 主因（渲染只占 14ms，DWM 重合成才是大头）。
            rounded = round(alpha, 3)
            if rounded != self._last_alpha:
                self.root.attributes("-alpha", max(0.0, min(1.0, rounded)))
                self._last_alpha = rounded
            frame = render_frame(elapsed)
            self._photo = self._image_tk.PhotoImage(frame)
            self.label.configure(image=self._photo)
            # 只处理布局，不跑完整事件循环，避免整窗重绘。
            self.root.update_idletasks()
        except Exception:
            return False
        return True

    def close(self) -> None:
        try:
            self.root.destroy()
        except Exception:
            pass
