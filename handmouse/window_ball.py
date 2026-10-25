"""应用窗口底部的白色悬浮球：靠"拉球"来扩展 / 缩放窗口。

用户 2026-10-22 的原话（需求就是规格）：

    在应用的下面加一个白色的圆形悬浮球，当我向左拉，应用就向左扩展，
    向上拉就整体缩小，向下拉就整体变大，向右拉就向右扩展，然后向每个方向
    拉都加一个扯圆球的动态效果，比如向左拉，圆球左边就被扯一小块，松开后
    又会弹回去，然后每个应用都有自己的圆球，但是开很多窗口的时候，就只有
    点击这个应用的时候圆球才会出现，应用静止两秒后就会渐渐淡出。

设计取舍（为什么这样分层）
--------------------------
1. **球是画在我们的全屏透明层上的，不是真窗口。** 现有叠加层是
   ``WS_EX_TRANSPARENT``（点击穿透），做成真窗口就得额外处理命中测试、
   Z 序、焦点，还会把点击漏给桌面。这里的做法是：球只是**画**上去的，
   命中判定由我们自己算——因为鼠标本来就是我们注入的，谁被点到我们说了算。

2. **抓住球的捏合不发给操作系统。** 管道在 ``LEFT_DOWN`` 时先问一句
   "这一下是不是落在球上"，是的话把这条指令**吞掉**，改成进入拉球模式。
   否则这次按下会穿透到球下面的桌面/其他窗口，拉球的同时还顺手点了别处。

3. **窗口几何用"拉之前的矩形"当基准。** 拉左会把窗口左边推出去，窗口
   一变宽，底部居中的球也就跟着移位了。若每帧拿新矩形重算，就会形成
   正反馈、窗口一路跑飞。所以基准矩形在抓球那一刻固定下来。

4. **松手后的回弹是纯动画**，只影响画出来的形变量，不再改窗口。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

from handmouse import grid_winctl

LOGGER = logging.getLogger(__name__)

#: 球的半径（像素）。他截图里的圆球直径约 34px，这里取半径 17。
BALL_RADIUS = 17
#: 球心到窗口底边的距离（像素）：球整体落在窗口外面，不挡标题栏和内容。
BALL_GAP = 12
#: 拉满所需的拖拽量（像素）。超过就按满量算，避免继续拖时窗口飞出屏幕。
PULL_FULL_PX = 220.0
#: 水平拉 1 像素 → 窗口边缘移动多少像素。
EDGE_GAIN = 0.55
#: 垂直拉 1 像素 → 窗口整体缩放多少比例。
SCALE_GAIN = 0.0011
#: 窗口尺寸的硬上限（相对屏幕）：不允许缩放到比这更大，否则会把任务栏顶掉。
MAX_SCREEN_RATIO = 0.98
#: 静止多久开始淡出（毫秒）——他要求 2 秒。
IDLE_FADE_MS = 2000
#: 淡入 / 淡出时长（毫秒）。
FADE_MS = 380
#: 松手回弹的时长（毫秒）与阻尼振荡次数。
SPRING_MS = 260
SPRING_CYCLES = 1.4


@dataclass(frozen=True, slots=True)
class BallAnchor:
    """一个窗口对应的球：位置与它所属窗口的基准矩形。"""

    hwnd: int
    center: tuple[float, float]
    base_rect: tuple[int, int, int, int]


def ball_center(rect: tuple[int, int, int, int]) -> tuple[float, float]:
    """窗口底部居中的球心（在窗口**下方**，不压内容）。"""

    left, _top, right, bottom = rect
    return ((left + right) / 2.0, bottom + BALL_GAP + BALL_RADIUS)


def hits_ball(center: tuple[float, float], px: float, py: float, *, slack: float = 4.0) -> bool:
    """鼠标是否落在球上。``slack`` 给一点容错，隔空点不太准。"""

    return math.hypot(px - center[0], py - center[1]) <= BALL_RADIUS + slack


def clamp_pull(dx: float, dy: float) -> tuple[float, float]:
    """把拉拽向量限制在 ``PULL_FULL_PX`` 之内（等比例缩短）。"""

    length = math.hypot(dx, dy)
    if length <= PULL_FULL_PX or length <= 1e-6:
        return dx, dy
    ratio = PULL_FULL_PX / length
    return dx * ratio, dy * ratio


def resize_for_pull(
    rect: tuple[int, int, int, int],
    dx: float,
    dy: float,
    *,
    screen: tuple[int, int] | None = None,
) -> tuple[int, int, int, int]:
    """把"拉拽向量"翻译成新的窗口矩形。

    * ``dx > 0``（向右拉）→ 右边缘外扩
    * ``dx < 0``（向左拉）→ 左边缘外扩（左边界向左移）
    * ``dy < 0``（向上拉）→ 整体缩小（以中心为基准）
    * ``dy > 0``（向下拉）→ 整体放大（以中心为基准）

    水平与垂直可以同时生效（斜着拉就是"一边扩边一边缩放"），这是他描述的
    四向行为的自然叠加，不需要互斥。
    """

    left, top, right, bottom = rect
    dx, dy = clamp_pull(dx, dy)

    # 水平：只动被拉的那一侧，另一侧钉住 —— "向左扩展"就是这个意思。
    shift = dx * EDGE_GAIN
    if shift > 0:
        right += int(round(shift))
    elif shift < 0:
        left += int(round(shift))

    # 垂直：以中心为基准等比缩放。**向下拉变大、向上拉变小**（他指定的方向）。
    scale = 1.0 + dy * SCALE_GAIN
    scale = max(0.25, min(4.0, scale))
    if abs(scale - 1.0) > 1e-6:
        width = max(1.0, float(right - left))
        height = max(1.0, float(bottom - top))
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        new_w = width * scale
        new_h = height * scale
        if screen is not None:
            new_w = min(new_w, screen[0] * MAX_SCREEN_RATIO)
            new_h = min(new_h, screen[1] * MAX_SCREEN_RATIO)
        left = int(round(cx - new_w / 2.0))
        right = int(round(cx + new_w / 2.0))
        top = int(round(cy - new_h / 2.0))
        bottom = int(round(cy + new_h / 2.0))

    # 夹到最小尺寸（以被固定的那一侧为锚，保证"向左扩展"时右边不动）。
    if right - left < grid_winctl.MIN_WINDOW_W:
        right = left + grid_winctl.MIN_WINDOW_W
    if bottom - top < grid_winctl.MIN_WINDOW_H:
        bottom = top + grid_winctl.MIN_WINDOW_H
    return (int(left), int(top), int(right), int(bottom))


@dataclass(slots=True)
class BallTrack:
    """一个窗口的球的显示状态（每个应用各自一份）。"""

    hwnd: int
    alpha: float = 0.0
    last_touch_ns: int = 0
    center: tuple[float, float] = (0.0, 0.0)
    #: 当前画出来的形变量（像素，指向拉的方向）。松手后按弹簧衰减回 0。
    stretch: tuple[float, float] = (0.0, 0.0)
    #: 弹簧动画起始时刻与起始形变（松手那一刻定格）。
    spring_from_ns: int = 0
    spring_start: tuple[float, float] = (0.0, 0.0)


def spring_offset(elapsed_ms: float, start: tuple[float, float]) -> tuple[float, float]:
    """阻尼振荡的回弹曲线：``A·e^(-t/τ)·cos(2πf t)``。

    ``τ = SPRING_MS/(2π·SPRING_CYCLES)`` 让整个回弹大致在 ``SPRING_MS`` 内
    衰减到看不见，同时有 1~2 次肉眼可辨的过冲——这就是"弹回去"的手感。
    """

    if elapsed_ms <= 0:
        return start
    if elapsed_ms >= SPRING_MS * 3:
        return (0.0, 0.0)
    tau = SPRING_MS / (2.0 * math.pi * SPRING_CYCLES)
    decay = math.exp(-elapsed_ms / tau)
    wave = math.cos(2.0 * math.pi * SPRING_CYCLES * elapsed_ms / SPRING_MS)
    factor = decay * wave
    return (start[0] * factor, start[1] * factor)


class WindowBallController:
    """悬浮球的所有逻辑：谁的球可见、透明度、抓没抓住、窗口该变成多大。

    不碰 tkinter，也不直接调 Win32——窗口几何通过返回值交给调用方执行，
    所以可以完整单测。
    """

    def __init__(
        self,
        *,
        exclude_hwnd: int = 0,
        now_ns=time.monotonic_ns,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.exclude_hwnd = exclude_hwnd
        self._now_ns = now_ns
        self.tracks: dict[int, BallTrack] = {}
        self.active_hwnd = 0
        #: 抓球状态：(hwnd, 基准矩形, 球心)。非 None 表示正在拉。
        self._grab: tuple[int, tuple[int, int, int, int], tuple[float, float]] | None = None
        self._last_rect_sent: tuple[int, int, int, int] | None = None
        #: 上一帧的前台窗口，用来识别"切到另一个应用"这件事。
        self._prev_active_hwnd: int = 0

    # ---- 查询 ----

    @property
    def grabbing(self) -> bool:
        return self._grab is not None

    def track(self, hwnd: int) -> BallTrack:
        found = self.tracks.get(hwnd)
        if found is None:
            found = BallTrack(hwnd=hwnd)
            self.tracks[hwnd] = found
        return found

    def visible_tracks(self) -> list[BallTrack]:
        return [t for t in self.tracks.values() if t.alpha > 0.01]

    # ---- 每帧推进 ----

    def update(self, now_ns: int, cursor: tuple[float, float] | None = None) -> None:
        """推进淡入淡出与回弹。``cursor`` 是屏幕像素坐标。"""

        if not self.enabled:
            for track in self.tracks.values():
                track.alpha = 0.0
            return

        hwnd = grid_winctl.foreground_window(exclude=self.exclude_hwnd)
        self.active_hwnd = hwnd
        if hwnd and hwnd not in self.tracks:
            # 新窗口第一次出现：**立刻叫醒它的球**。
            # 🔴 2026-10-22 实机 bug：只靠"鼠标靠近球"叫醒的话，球一开始
            # alpha=0 是看不见的，用户根本不知道该往哪指 —— 于是永远看不到球。
            # 他要求的是"只有点击这个应用时球才出现"，而切换到某窗口本身
            # 就是点击它，所以这里直接给活跃时间戳。
            track = self.track(hwnd)
            track.last_touch_ns = now_ns
        previous_active = getattr(self, "_prev_active_hwnd", 0)
        if hwnd and hwnd != previous_active:
            # 切回一个已有的窗口也算"点了它"。
            self.track(hwnd).last_touch_ns = now_ns
        self._prev_active_hwnd = hwnd

        for track in self.tracks.values():
            rect = grid_winctl.window_rect(track.hwnd)
            if rect is not None:
                track.center = ball_center(rect)
            is_active = track.hwnd == hwnd
            idle_ms = (now_ns - track.last_touch_ns) / 1_000_000

            # 目标透明度：活跃窗口且未静止超时 → 1；否则 0。
            want = 1.0 if (is_active and idle_ms < IDLE_FADE_MS) else 0.0
            step = 1.0 / max(1.0, FADE_MS / 16.0)
            if track.alpha < want:
                track.alpha = min(want, track.alpha + step)
            elif track.alpha > want:
                track.alpha = max(want, track.alpha - step)

            # 松手后的弹簧回弹（只影响画出来的形变）。
            if track.spring_from_ns:
                elapsed = (now_ns - track.spring_from_ns) / 1_000_000
                track.stretch = spring_offset(elapsed, track.spring_start)
                if elapsed >= SPRING_MS * 3:
                    track.spring_from_ns = 0
                    track.stretch = (0.0, 0.0)

            # 鼠标靠近球 = 活跃（避免用户正对着球却已经淡出）。
            if is_active and cursor is not None and hits_ball(track.center, *cursor):
                track.last_touch_ns = now_ns

    def notice_activity(self, now_ns: int | None = None) -> None:
        """用户点了这个窗口：把它的球叫回来（他要求"只有点击应用时球才出现"）。"""

        if not self.active_hwnd:
            return
        self.track(self.active_hwnd).last_touch_ns = now_ns or self._now_ns()

    # ---- 抓放 ----

    def try_grab(self, cursor: tuple[float, float], now_ns: int | None = None):
        """若鼠标正落在活跃窗口的球上，进入拉球模式。

        返回被抓住的 ``BallAnchor``；没抓住返回 ``None``（调用方照常派发点击）。
        """

        if not self.enabled or not self.active_hwnd:
            return None
        track = self.tracks.get(self.active_hwnd)
        if track is None or track.alpha <= 0.05:
            return None
        if not hits_ball(track.center, *cursor):
            return None
        rect = grid_winctl.window_rect(self.active_hwnd)
        if rect is None or grid_winctl.is_maximized(self.active_hwnd):
            return None
        now = now_ns or self._now_ns()
        track.last_touch_ns = now
        track.spring_from_ns = 0
        anchor = BallAnchor(self.active_hwnd, track.center, rect)
        self._grab = (anchor.hwnd, anchor.base_rect, anchor.center)
        self._last_rect_sent = rect
        return anchor

    def drag_to(self, cursor: tuple[float, float]) -> tuple[int, int, int, int] | None:
        """拉球时的窗口矩形。返回 ``None`` 表示不需要改窗口（没在拉 / 变化太小）。"""

        if self._grab is None:
            return None
        hwnd, base, center = self._grab
        dx = cursor[0] - center[0]
        dy = cursor[1] - center[1]
        rect = resize_for_pull(base, dx, dy, screen=grid_winctl.screen_size())
        track = self.tracks.get(hwnd)
        if track is not None:
            # 球跟着被扯出去（手指拉着它），这就是"左边被扯一小块"的来源。
            track.stretch = clamp_pull(dx, dy)
        if rect == self._last_rect_sent:
            return None
        self._last_rect_sent = rect
        return rect

    def release(self, now_ns: int | None = None) -> int:
        """松手：结束抓取，让球回弹。返回被操作的 hwnd（0 表示没在拉）。"""

        if self._grab is None:
            return 0
        hwnd, _base, _center = self._grab
        self._grab = None
        self._last_rect_sent = None
        track = self.tracks.get(hwnd)
        now = now_ns or self._now_ns()
        if track is not None:
            track.spring_from_ns = now
            track.spring_start = track.stretch
            track.last_touch_ns = now
        return hwnd

    def cancel(self) -> None:
        """中止拉球（暂停、手丢失、窗口没了）——**不**改窗口几何。"""

        self._grab = None
        self._last_rect_sent = None

    def forget(self, hwnd: int) -> None:
        """窗口关闭：丢掉它的球，免得越积越多。"""

        self.tracks.pop(hwnd, None)
        if self._grab is not None and self._grab[0] == hwnd:
            self.cancel()

    def prune(self) -> None:
        """清掉已经不存在的窗口的球。"""

        dead = [h for h in self.tracks if not grid_winctl.window_rect(h)]
        for hwnd in dead:
            self.forget(hwnd)


class WindowBallOverlay:
    """把悬浮球画到全屏透明层上。

    窗口三件套（分层 / 色键透明 / 点击穿透）直接沿用 ``snap_ui`` / ``grid_ui``
    的做法——**必须点击穿透**：球只是画的，命中判定由 `WindowBallController`
    自己算，否则这个全屏窗口会把桌面上所有点击都吃掉。

    不跑自己的事件循环，只提供 ``refresh()`` 由主循环每帧推进。
    """

    #: 色键透明用的背景色，与 snap_ui 保持一致。
    CHROMA = "#010203"

    def __init__(self, controller: WindowBallController, *, title: str = "手势控制 · 悬浮球") -> None:
        import tkinter as tk

        self.closed = False
        self._shown = False
        self.controller = controller

        self.root = tk.Tk()
        self.root.title(title)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.configure(bg=self.CHROMA)
        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg=self.CHROMA,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self._items: dict[int, list[int]] = {}

    def show(self) -> bool:
        from handmouse.grid_ui import _apply_overlay_styles

        if self._shown:
            return True
        self._shown = True
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        click_through, transparent_ok = _apply_overlay_styles(self.root, self.CHROMA)
        if not (click_through and transparent_ok):
            LOGGER.warning("悬浮球层透明或穿透设置失败，已放弃该层")
            self.close()
            return False
        return True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001 - tkinter 关闭阶段的异常无意义
            pass

    def _blend(self, alpha: float) -> str:
        """把白色按 ``alpha`` 混到色键背景上。

        色键透明的层不能设整体透明度（会连穿透一起失效），所以用"把颜色
        往背景色靠"来模拟淡出——背景是色键色，越接近它就越看不见，
        alpha=0 时与背景**逐通道完全相同**，色键直接抠掉。
        """

        alpha = max(0.0, min(1.0, alpha))
        chroma = (0x01, 0x02, 0x03)  # 与 grid_ui._CHROMA 一致
        channels = [
            int(round(base + (255 - base) * alpha)) for base in chroma
        ]
        return "#{:02x}{:02x}{:02x}".format(*channels)

    def refresh(self) -> None:
        """推进一帧绘制。"""

        if self.closed:
            return
        try:
            alive = set()
            for track in self.controller.tracks.values():
                if track.alpha <= 0.01:
                    continue
                alive.add(track.hwnd)
                color = self._blend(track.alpha)
                cx, cy = track.center
                sx, sy = track.stretch
                pull = math.hypot(sx, sy)
                radius = BALL_RADIUS

                ids = self._items.get(track.hwnd)
                if ids is None:
                    ids = [
                        self.canvas.create_polygon(0, 0, 0, 0, 0, 0, outline="", fill=""),
                        self.canvas.create_oval(0, 0, 0, 0, outline="", width=2),
                        self.canvas.create_oval(0, 0, 0, 0, outline="", fill=""),
                    ]
                    self._items[track.hwnd] = ids
                tail_id, ring_id, dot_id = ids

                # 1) 被扯出去的"橡皮筋"：从球心拉出一个水滴，长度按形变量。
                if pull > 2.0:
                    ux, uy = sx / pull, sy / pull
                    length = min(pull * 0.55, BALL_RADIUS * 3.2)
                    tip_x, tip_y = cx + ux * length, cy + uy * length
                    # 侧向法线，用来给水滴一个宽度
                    nx, ny = -uy, ux
                    waist = BALL_RADIUS * 0.45
                    self.canvas.coords(
                        tail_id,
                        cx + nx * waist, cy + ny * waist,
                        tip_x, tip_y,
                        cx - nx * waist, cy - ny * waist,
                        cx, cy,
                    )
                    self.canvas.itemconfigure(tail_id, fill=color, outline=color)
                    self.canvas.itemconfigure(tail_id, state="normal")
                else:
                    self.canvas.itemconfigure(tail_id, state="hidden")

                # 2) 圆环本身。被拉时朝拉的方向偏心一点，看起来像球被拽变形。
                offset = min(pull * 0.12, BALL_RADIUS * 0.5)
                if pull > 1e-6:
                    ux, uy = sx / pull, sy / pull
                else:
                    ux = uy = 0.0
                bx, by = cx + ux * offset, cy + uy * offset
                self.canvas.coords(
                    ring_id,
                    bx - radius, by - radius, bx + radius, by + radius,
                )
                self.canvas.itemconfigure(ring_id, outline=color, width=2, state="normal")

                # 3) 中心的点（他截图里球心有个点）。
                dot = max(2.0, radius * 0.22)
                self.canvas.coords(dot_id, bx - dot, by - dot, bx + dot, by + dot)
                self.canvas.itemconfigure(dot_id, fill=color, outline=color, state="normal")

            # 淡出的球把画布对象藏起来，别留着残影。
            for hwnd, ids in list(self._items.items()):
                if hwnd not in alive:
                    for item in ids:
                        self.canvas.itemconfigure(item, state="hidden")

            self.root.update_idletasks()
            self.root.update()
        except Exception:
            LOGGER.debug("悬浮球层刷新失败", exc_info=True)
