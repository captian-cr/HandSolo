"""网格选择的屏幕层叠加窗口：画图标、画悬浮球、显示命中高亮。

为什么必须画在屏幕层
--------------------
浮动反馈如果只画在摄像头预览小窗里，用户操作时得盯着那个小窗看，眼睛离开
屏幕，完全违背直觉。这里用一个全屏透明置顶窗口，把图标和球画在**屏幕真实
坐标**上，用户看着屏幕就知道自己指着哪。

两个 Windows 细节
-----------------
1. ``-transparentcolor`` 让底色整块透明；底色与 Canvas 背景必须完全一致，
   否则会留下一层灰膜。
2. 必须加 ``WS_EX_TRANSPARENT`` 让鼠标事件穿透，否则这层窗口会挡住所有
   点击，桌面彻底不可用。这是本文件里最容易漏、漏了最致命的一处。
"""

from __future__ import annotations

import colorsys
import ctypes
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field

from handmouse.grid_map import GridLayout
from handmouse.screen_map import enable_dpi_awareness

#: 作为透明键的底色。选一个几乎不会出现在界面里的值，避免误透明。
_CHROMA = "#010203"

_WS_EX_LAYERED = 0x0008_0000
_WS_EX_TRANSPARENT = 0x0000_0020
_GWL_EXSTYLE = -20
_LWA_COLORKEY = 0x1


@dataclass(slots=True)
class PointerState:
    """检测线程与 UI 线程之间传递的指针状态（由外部加锁）。"""

    x: float = 0.5
    y: float = 0.5
    target: int | None = None
    pinching: bool = False
    hand_visible: bool = False
    paused: bool = False
    pending: bool = False
    snapped: bool = False
    pose: str = ""
    #: 各手指伸直度，格式如 "食2.4 中0.6 无0.5 小0.5"，用于诊断为何未识别。
    finger_hint: str = ""
    fps: float = 0.0
    ratio: float = float("inf")
    message: str = ""


def _colorref(hex_color: str) -> int:
    """``#RRGGBB`` -> Windows COLORREF（0x00BBGGRR，字节序是反的）。"""

    value = hex_color.lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return (b << 16) | (g << 8) | r


def _hwnd_of(root: tk.Tk) -> int:
    """取顶层窗口句柄。

    ``wm_frame()`` 直接给出顶层框架的 HWND，比 ``GetParent(winfo_id())``
    可靠：``overrideredirect`` 窗口没有装饰父窗口，``GetParent`` 可能返回 0
    甚至桌面窗口，把样式设到错误的窗口上。
    """

    try:
        value = int(root.wm_frame(), 16)
        if value:
            return value
    except Exception:
        pass
    try:
        user32 = ctypes.windll.user32
        return user32.GetParent(root.winfo_id()) or root.winfo_id()
    except Exception:
        return root.winfo_id()


def _apply_overlay_styles(root: tk.Tk, chroma: str) -> tuple[bool, bool]:
    """一次性设好分层、色键透明、点击穿透。返回 ``(穿透, 透明)`` 是否成功。

    为什么不用 tkinter 的 ``-transparentcolor``：它内部会重写整个 exstyle，
    把先前设好的 ``WS_EX_TRANSPARENT`` 冲掉——实测结果是透明生效、穿透失效，
    于是这个全屏窗口把桌面上所有点击都吃掉了，用户看到的现象是"电脑卡死"。
    这里读-改-写一次设到位，顺序完全可控。

    同时**必须先设 LAYERED 再 SetLayeredWindowAttributes**：一个 layered 但没有
    alpha/colorkey 的窗口在 Windows 上默认完全不可见。
    """

    try:
        user32 = ctypes.windll.user32
        hwnd = _hwnd_of(root)
        style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongW(
            hwnd,
            _GWL_EXSTYLE,
            style | _WS_EX_LAYERED | _WS_EX_TRANSPARENT,
        )
        ok = user32.SetLayeredWindowAttributes(
            hwnd,
            _colorref(chroma),
            0,
            _LWA_COLORKEY,
        )
        return True, bool(ok)
    except Exception:
        return False, False


def window_state(root: tk.Tk) -> str:
    """诊断用：报告窗口句柄、可见性、exstyle 与分层属性。

    "程序没反应"最可能是窗口存在但不可见。把这几项打出来，一眼能分辨
    是"没创建"、"创建了但被隐藏"还是"透明键失效"。
    """

    try:
        user32 = ctypes.windll.user32
        hwnd = _hwnd_of(root)
        visible = bool(user32.IsWindowVisible(hwnd))
        style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        layered = bool(style & _WS_EX_LAYERED)
        transparent = bool(style & _WS_EX_TRANSPARENT)

        key = ctypes.c_ulong(0)
        alpha = ctypes.c_ubyte(0)
        flags = ctypes.c_ulong(0)
        got = user32.GetLayeredWindowAttributes(
            hwnd,
            ctypes.byref(key),
            ctypes.byref(alpha),
            ctypes.byref(flags),
        )
        if got:
            value = key.value & 0xFFFFFF
            r = value & 0xFF
            g = (value >> 8) & 0xFF
            b = (value >> 16) & 0xFF
            key_text = f"#{r:02x}{g:02x}{b:02x}"
        else:
            key_text = "无"
        return (
            f"hwnd={hwnd} visible={visible} layered={layered} "
            f"transparent={transparent} colorkey={key_text} "
            f"alpha={alpha.value} lwa_flags={flags.value}"
        )
    except Exception as exc:
        return f"诊断失败: {exc}"


def _hue_color(hue: float, saturation: float = 0.72, value: float = 1.0) -> str:
    """HSV -> ``#rrggbb``。光环用它做色相环。"""

    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


class HaloRenderer:
    """围绕选中目标旋转的彩色光环。

    用若干段弧拼成一个圆环，每段取自色相环的不同位置，随时间整体旋转。
    这样"选中的东西"在一屏图标里一眼可辨，比单色边框醒目得多。

    段数取 18：再多在低帧率下看不出渐变，再少会显得是一节一节跳色。
    """

    def __init__(self, canvas: tk.Canvas, segments: int = 18, width: int = 4) -> None:
        self.canvas = canvas
        self.segments = segments
        self.width = width
        self._items: list[int] = []
        step = 360.0 / segments
        for index in range(segments):
            self._items.append(
                canvas.create_arc(
                    0, 0, 0, 0,
                    start=index * step,
                    extent=step * 0.86,
                    style="arc",
                    outline=_hue_color(index / segments),
                    width=width,
                    state="hidden",
                ),
            )

    def hide(self) -> None:
        for item in self._items:
            self.canvas.itemconfigure(item, state="hidden")

    def update(self, cx: float, cy: float, radius: float, phase: float) -> None:
        """把光环放到 ``(cx, cy)`` 并旋转到 ``phase``（0~1 一圈）。"""

        self.update_box(
            cx - radius, cy - radius, cx + radius, cy + radius, phase,
        )

    def update_box(
        self, x0: float, y0: float, x1: float, y1: float, phase: float,
    ) -> None:
        """把光环套在一个任意矩形外围（网格格子是宽扁的，不能只支持正圆）。"""

        box = (x0, y0, x1, y1)
        step = 360.0 / self.segments
        offset = phase * 360.0
        for index, item in enumerate(self._items):
            self.canvas.coords(item, *box)
            self.canvas.itemconfigure(
                item,
                start=(offset + index * step) % 360.0,
                outline=_hue_color(index / self.segments + phase),
                state="normal",
            )


def _rounded_rect(
    canvas: tk.Canvas,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    radius: float,
    **kwargs: object,
) -> int:
    """用平滑多边形近似圆角矩形（Canvas 没有原生圆角矩形）。"""

    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    points = [
        x0 + r, y0, x1 - r, y0, x1, y0,
        x1, y0 + r, x1, y1 - r, x1, y1,
        x1 - r, y1, x0 + r, y1, x0, y1,
        x0, y1 - r, x0, y0 + r, x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class GridOverlay:
    """全屏透明叠加层。

    由外部在主线程调用 ``run()``；检测线程只更新 ``PointerState``。
    """

    def __init__(
        self,
        layout: GridLayout,
        labels: list[str],
        state: PointerState,
        *,
        quit_keys: tuple[int, ...] = (0x1B,),  # VK_ESCAPE
        title: str = "手势选择",
        font_size: int = 30,
    ) -> None:
        if len(labels) != layout.cols * layout.rows:
            raise ValueError("labels 数量必须等于 cols*rows")

        self.layout = layout
        self.labels = labels
        self.state = state
        self.quit_keys = quit_keys
        self.title = title
        self.font_size = font_size
        self.closed = False

        enable_dpi_awareness()
        self.root = tk.Tk()
        self.root.title(title)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.configure(bg=_CHROMA)

        # 样式在窗口真正 map 之后再设（见 show()）：overrideredirect 窗口在
        # 实体化之前 SetWindowLongW 会被系统丢弃，表现为穿透与透明双双失效。
        self._click_through = False
        self._transparent_ok = False
        self._shown = False
        self.hwnd: int = 0

        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg=_CHROMA,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self._item_ids: list[tuple[int, int, int]] = []
        self._ball_id: int = 0
        self._ball_glow_id: int = 0
        self._ball_gloss_id: int = 0
        self._hud_id: int = 0
        self._hint_id: int = 0
        self._flash_until_s: float = 0.0
        self._flash_index: int | None = None
        # 球的显示位置（做一阶低通，让它"滑"向磁吸后的位置）
        self._ball_x = self.screen_w / 2
        self._ball_y = self.screen_h / 2

        self._draw_grid()
        self._draw_ball()
        self._draw_hud()
        self._halo = HaloRenderer(self.canvas, segments=18, width=4)

    # ---- 绘制 ----

    def _px(self, x: float, y: float) -> tuple[float, float]:
        return x * self.screen_w, y * self.screen_h

    def _draw_grid(self) -> None:
        for rect in self.layout.rects():
            label = self.labels[rect.index]
            if not label:
                # 空位不绘制：留着可点的外观会让人以为那里有东西。
                self._item_ids.append((0, 0, rect.index))
                continue
            x0, y0 = self._px(rect.x0, rect.y0)
            x1, y1 = self._px(rect.x1, rect.y1)
            pad = 6
            fill_id = _rounded_rect(
                self.canvas,
                x0 + pad, y0 + pad, x1 - pad, y1 - pad,
                radius=26,
                fill="#171b24",
                outline="#39415a",
                width=2,
            )
            text_id = self.canvas.create_text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                text=label,
                fill="#e8ecf5",
                font=("Microsoft YaHei UI", self.font_size),
                width=(x1 - x0) - 40,
            )
            self._item_ids.append((fill_id, text_id, rect.index))

    def _draw_ball(self) -> None:
        # 捏合时的外圈光晕
        self._ball_glow_id = self.canvas.create_oval(
            0, 0, 0, 0, outline="#a9bacf", width=2, state="hidden",
        )
        # 球体：小、浅、细白边——大了会挡住底下的内容
        self._ball_id = self.canvas.create_oval(
            0, 0, 0, 0, fill="#eef2f8", outline="#ffffff", width=1,
        )
        # 左上高光，给一点球面感（Canvas 无渐变，用色块近似）
        self._ball_gloss_id = self.canvas.create_oval(
            0, 0, 0, 0, fill="#ffffff", outline="",
        )

    def _draw_hud(self) -> None:
        self._hud_id = self.canvas.create_text(
            24, 24,
            anchor="nw",
            text="",
            fill="#9fb0c9",
            font=("Consolas", 14),
        )
        self._hint_id = self.canvas.create_text(
            self.screen_w - 24, 24,
            anchor="ne",
            text="ESC 退出",
            fill="#5f6b80",
            font=("Microsoft YaHei UI", 15),
        )

    # ---- 更新 ----

    def refresh(self) -> None:
        """由主线程定时调用，把 ``PointerState`` 反映到画面上。"""

        target = self.state.target
        pinching = self.state.pinching
        flashing = (
            self._flash_index
            if time.monotonic() < self._flash_until_s
            else None
        )

        # 选中目标外圈套一道旋转的彩色光环，用于在多个图标之间快速区分。
        if target is not None and 0 <= target < len(self._item_ids):
            rect = self.layout.rect(target)
            x0, y0 = self._px(rect.x0, rect.y0)
            x1, y1 = self._px(rect.x1, rect.y1)
            pad = 12
            self._halo.update_box(
                x0 - pad, y0 - pad, x1 + pad, y1 + pad,
                phase=(time.monotonic() * 0.30) % 1.0,
            )
        else:
            self._halo.hide()

        for fill_id, text_id, index in self._item_ids:
            if not fill_id:
                continue
            if index == flashing:
                self.canvas.itemconfigure(
                    fill_id, fill="#4a3a12", outline="#ffd166", width=4,
                )
                self.canvas.itemconfigure(text_id, fill="#ffd166")
            elif index == target:
                fill = "#1d3f7a" if not pinching else "#2b6cd4"
                outline = "#7db2ff" if not pinching else "#ffd166"
                self.canvas.itemconfigure(fill_id, fill=fill, outline=outline, width=3)
                self.canvas.itemconfigure(text_id, fill="#ffffff")
            else:
                self.canvas.itemconfigure(
                    fill_id, fill="#171b24", outline="#39415a", width=2,
                )
                self.canvas.itemconfigure(text_id, fill="#8e9ab3")

        # 磁吸：命中时球被拉到图标中心，未命中时停在手指原始位置。
        if target is not None:
            tx, ty = self._px(*self.layout.center(target))
        else:
            tx, ty = self._px(self.state.x, self.state.y)

        if not self.state.hand_visible:
            self.canvas.itemconfigure(self._ball_id, state="hidden")
            self.canvas.itemconfigure(self._ball_glow_id, state="hidden")
            self.canvas.itemconfigure(self._ball_gloss_id, state="hidden")
        else:
            # 一阶低通：0.35 的比例在 30fps 下约 100ms 收敛，视觉上像吸附。
            self._ball_x += (tx - self._ball_x) * 0.35
            self._ball_y += (ty - self._ball_y) * 0.35
            r = 10.0 if pinching else 14.0
            self.canvas.coords(
                self._ball_id,
                self._ball_x - r, self._ball_y - r,
                self._ball_x + r, self._ball_y + r,
            )
            self.canvas.itemconfigure(self._ball_id, state="normal")

            gx = self._ball_x - r * 0.34
            gy = self._ball_y - r * 0.34
            gr = r * 0.30
            self.canvas.coords(
                self._ball_gloss_id, gx - gr, gy - gr, gx + gr, gy + gr,
            )
            self.canvas.itemconfigure(self._ball_gloss_id, state="normal")

            self.canvas.coords(
                self._ball_glow_id,
                self._ball_x - r - 8, self._ball_y - r - 8,
                self._ball_x + r + 8, self._ball_y + r + 8,
            )
            self.canvas.itemconfigure(
                self._ball_glow_id,
                state="normal" if pinching else "hidden",
            )

        ratio = self.state.ratio
        ratio_text = "--" if ratio == float("inf") else f"{ratio:.2f}"
        self.canvas.itemconfigure(
            self._hud_id,
            text=(
                f"捏合比 {ratio_text}   FPS {self.state.fps:.0f}   "
                f"目标 {'--' if target is None else self.labels[target]}\n"
                f"{self.state.message}"
            ),
        )

    def set_labels(self, labels: list[str]) -> None:
        """替换图标文字（数量必须与格子数一致）。

        用于在"启动器"与"窗口控制"两个面板之间切换：格子数量不变，只换文案，
        因此不需要重建 Canvas 元素，切换是瞬时的。
        """

        if len(labels) != len(self._item_ids):
            raise ValueError(
                f"labels 数量 {len(labels)} 与格子数 {len(self._item_ids)} 不一致",
            )
        self.labels = list(labels)
        for _fill_id, text_id, index in self._item_ids:
            if not text_id:
                continue
            self.canvas.itemconfigure(text_id, text=self.labels[index])

    def flash(self, index: int) -> None:
        """触发后的短促反馈：该图标描边闪金 0.35 秒。"""

        self._flash_index = index
        self._flash_until_s = time.monotonic() + 0.35

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def show(self) -> bool:
        """把窗口实体化、设好透明与穿透。返回是否一切正常。

        单独抽出来是为了让调用方**先把界面显示出来，再去加载模型**——
        模型初始化要几秒，期间主线程如果没进消息循环，窗口不会刷新，
        用户看到的就是"程序卡住了"。
        """

        if self._shown:
            return self._click_through and self._transparent_ok
        self._shown = True

        # 先映射窗口：IsWindowVisible 只有在实体化之后才有意义，而
        # overrideredirect 窗口在实体化之前收到的样式设置会被系统丢弃。
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self.root.update()

        self.hwnd = _hwnd_of(self.root)
        self._click_through, self._transparent_ok = _apply_overlay_styles(
            self.root, _CHROMA,
        )
        print(f"[窗口] {window_state(self.root)}", file=sys.stderr)
        print(
            f"[窗口] click_through={self._click_through} "
            f"transparent_ok={self._transparent_ok} "
            f"尺寸={self.screen_w}x{self.screen_h}",
            file=sys.stderr,
        )
        if not (self._click_through and self._transparent_ok):
            # 失败要安全：一个不可穿透的全屏窗口会把桌面所有点击吃掉，
            # 用户会以为电脑死机。
            print(
                "[窗口] 透明或穿透设置失败。全屏窗口会挡住桌面点击，已停止。\n"
                "        请改用 `--dry-run` 查看数值，或在正常桌面上运行。",
                file=sys.stderr,
            )
        return self._click_through and self._transparent_ok

    def run(self, on_tick, interval_ms: int = 33) -> None:
        """进入主循环。``on_tick`` 每帧被调用一次（用于推进检测/触发逻辑）。"""

        if not self.show():
            self.close()
            return

        def _loop() -> None:
            if self.closed:
                return
            if self._quit_pressed():
                self.close()
                return
            try:
                on_tick()
                self.refresh()
            except tk.TclError:
                # 窗口正在销毁（用户按 ESC / 进程退出）时，残留的 after 回调
                # 会打到一个已不存在的 canvas 上。静默收尾，不要刷屏。
                self.closed = True
                return
            except Exception as exc:  # 不让单帧异常打死整个界面
                self.state.message = f"错误: {exc}"
            if not self.closed:
                self.root.after(interval_ms, _loop)

        self.root.after(interval_ms, _loop)
        self.root.mainloop()

    def _quit_pressed(self) -> bool:
        try:
            user32 = ctypes.windll.user32
            return any(user32.GetAsyncKeyState(key) & 0x8000 for key in self.quit_keys)
        except Exception:
            return False
