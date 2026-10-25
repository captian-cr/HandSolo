"""光标状态指示器：没有预览窗时，用**颜色**告诉你当前在什么状态。

为什么需要它（2026-10-21 用户原话）：
> 「真正使用这个系统的时候我们是不会要预览窗的，所以当我们再控制鼠标、
>   未进入控制状态、捏合、拖拽，这些功能都要用颜色区分出来」

设计取舍
--------
* **不做全屏层**：全屏透明窗口每帧重绘成本高（用户已经在意风扇），
  改成一个小圆环窗口跟着光标走，只在小范围重绘。
* 窗口**偏移**到光标右下方，不盖住你正在瞄的目标。
* 必须色键透明 + 点击穿透（沿用 `grid_ui._apply_overlay_styles`）：
  漏掉穿透的话这个小窗会吃掉点击，比全屏版还难受。
* 位置变化小于 2px 就不挪窗口，减少合成开销。

颜色语义
--------
绿 = 正在控制光标 · 橙 = 捏合 · 蓝 = 拖拽 · 红 = 未进入控制态 · 灰 = 暂停/丢手
"""

from __future__ import annotations

import logging
import tkinter as tk

from handmouse.grid_ui import _CHROMA, _apply_overlay_styles

LOGGER = logging.getLogger(__name__)

#: 状态 → (主色, 说明)。键与 `GestureType.value` 对齐。
STATE_COLORS: dict[str, tuple[str, str]] = {
    "pointer": ("#22c55e", "控制中"),
    "partial_index": ("#22c55e", "控制中"),
    "pinch_armed": ("#f59e0b", "捏合"),
    "left_pinch": ("#f59e0b", "捏合"),
    "right_pinch": ("#f59e0b", "捏合"),
    "drag": ("#3b82f6", "拖拽"),
    "swipe": ("#a855f7", "滑动"),
    "page_next": ("#a855f7", "滑动"),
    "page_previous": ("#a855f7", "滑动"),
    "open_palm": ("#9ca3af", "待机"),
    "fist": ("#9ca3af", "握拳"),
    "gate_idle": ("#ef4444", "未进入控制态"),
    "paused": ("#6b7280", "已暂停"),
    "lost": ("#6b7280", "丢手"),
}


class StateIndicator:
    """跟着光标走的彩色小圆环，用于无预览窗时的状态反馈。"""

    def __init__(self, *, size: int = 46, offset: tuple[int, int] = (26, 26)) -> None:
        self.closed = False
        self._shown = False
        self._ok = False
        self._size = max(24, int(size))
        self._offset = offset
        self._last_pos: tuple[int, int] | None = None
        self._last_state: str | None = None
        self._hidden = True

        self.root = tk.Tk()
        self.root.title("手势状态指示")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.configure(bg=_CHROMA)
        self.canvas = tk.Canvas(
            self.root,
            width=self._size,
            height=self._size,
            bg=_CHROMA,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()
        pad = 4
        self._ring = self.canvas.create_oval(
            pad, pad, self._size - pad, self._size - pad,
            outline="#22c55e", width=3,
        )
        self._dot = self.canvas.create_oval(
            self._size / 2 - 3, self._size / 2 - 3,
            self._size / 2 + 3, self._size / 2 + 3,
            fill="#22c55e", outline="",
        )
        # 「手出画」的专属符号：一道斜杠。没有预览窗时，圆环突然没了很难和
        # "程序崩了"区分开，所以先给它一个明确形状，再按超时收掉。
        self._slash = self.canvas.create_line(
            pad + 6, pad + 6, self._size - pad - 6, self._size - pad - 6,
            fill="#6b7280", width=3, state="hidden",
        )
        # 初始放到屏幕角落外，避免一启动就在左上角闪一下
        self.root.geometry(f"{self._size}x{self._size}+{-self._size}+{-self._size}")

    # ---- 生命周期 ----

    def show(self) -> bool:
        if self._shown:
            return self._ok
        self._shown = True
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        click_through, transparent = _apply_overlay_styles(self.root, _CHROMA)
        if not (click_through and transparent):
            LOGGER.warning("状态指示器透明/穿透设置失败，已放弃")
            self.close()
            return False
        self._ok = True
        return True

    def close(self) -> None:
        self.closed = True
        self._ok = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ---- 内容 ----

    def update(
        self,
        *,
        screen_x: float | None,
        screen_y: float | None,
        gesture: str,
        visible: bool = True,
    ) -> None:
        """设置一帧的显示内容。``screen_x/y`` 是像素坐标。"""

        if not self._ok or self.closed:
            return
        colour, _label = STATE_COLORS.get(gesture, ("#9ca3af", ""))
        # 「手出画」用斜杠标记，其余状态一律把斜杠收起来。
        try:
            self.canvas.itemconfigure(
                self._slash, state="normal" if gesture == "lost" else "hidden",
            )
        except tk.TclError:
            self.close()
            return
        if gesture != self._last_state:
            self._last_state = gesture
            try:
                self.canvas.itemconfigure(self._ring, outline=colour)
                self.canvas.itemconfigure(self._dot, fill=colour)
            except tk.TclError:
                self.close()
                return
        if not visible or screen_x is None or screen_y is None:
            self._hide()
            return
        pos = (
            int(screen_x) + self._offset[0],
            int(screen_y) + self._offset[1],
        )
        self._hidden = False
        if self._last_pos is not None and abs(pos[0] - self._last_pos[0]) < 2 and abs(
            pos[1] - self._last_pos[1],
        ) < 2:
            return
        self._last_pos = pos
        try:
            self.root.geometry(
                f"{self._size}x{self._size}+{pos[0]}+{pos[1]}",
            )
        except tk.TclError:
            self.close()

    def refresh(self) -> None:
        """推进一帧（由主循环调用，不自起事件循环）。"""

        if not self._ok or self.closed:
            return
        try:
            self.root.update_idletasks()
        except tk.TclError:
            self.close()

    def _hide(self) -> None:
        if self._hidden:
            return
        self._hidden = True
        self._last_pos = None
        try:
            self.root.geometry(
                f"{self._size}x{self._size}+{-self._size}+{-self._size}",
            )
        except tk.TclError:
            self.close()

    def hide(self) -> None:
        """立刻把圆环收走（用户要求：光标/手没了就别留在屏幕上）。"""

        if not self._ok or self.closed:
            return
        self._hide()
