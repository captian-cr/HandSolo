"""桌面图标模式：在真实图标位置做命中高亮，不重画图标。

与网格模式的区别
----------------
网格模式自己铺格子；这里**一个像素都不画图标**，只在用户指中的那个真实
桌面图标外面浮一圈高亮框。桌面原样可见，用户看到什么就能点什么。

命中为什么要用"最近邻 + 距离上限"而不是"落在矩形内"
----------------------------------------------------
桌面图标实测只有约 94×87 像素，且两两紧挨。要手指精确落进 94px 的方框，
按当前的映射精度几乎不可能，而且偏差一点就会点错隔壁。用最近邻 + 上限
则把"精确命中"换算成"大方向对"：指到附近就吸过去，指到远处（屏幕下半
部分）则判为无命中，不会误触第一行的图标。
"""

from __future__ import annotations

import ctypes
import logging
import sys
import time
from math import hypot

from handmouse.grid_desktop import DesktopIcon


LOGGER = logging.getLogger(__name__)
from handmouse.grid_ui import (
    _CHROMA,
    HaloRenderer,
    PointerState,
    _apply_overlay_styles,
    _hwnd_of,
    window_state,
)

import tkinter as tk


class DesktopIconMap:
    """目标命中判定（屏幕像素坐标）。

    虽然名字带 Icon，但接受任何具备 ``name`` / ``center`` / ``left`` 等属性的
    对象——桌面图标（``DesktopIcon``）与窗口控件（``UITarget``）都满足，
    因此两种模式共用同一套命中与绘制逻辑。
    """

    def __init__(self, icons: list, max_pick_px: float = 120.0) -> None:
        self.icons = icons
        #: 指到多远还算命中。默认 120px ≈ 1.3 倍图标宽度。
        self.max_pick_px = max_pick_px

    def hit(self, px: float, py: float) -> int | None:
        """返回命中图标在 ``icons`` 里的下标；超出上限返回 ``None``。

        用下标而不是 icon 对象，是为了和 ``PointerState.target`` 的整数语义
        保持一致，上层可以直接当索引使用。
        """

        best_index: int | None = None
        best_distance = self.max_pick_px
        for position, icon in enumerate(self.icons):
            cx, cy = icon.center
            distance = hypot(px - cx, py - cy)
            if distance <= best_distance:
                if distance < best_distance or best_index is None:
                    best_distance = distance
                    best_index = position
        return best_index


class DesktopOverlay:
    """全屏透明层：只在命中图标时浮一个高亮框 + 悬浮球。"""

    def __init__(
        self,
        icon_map: DesktopIconMap,
        state: PointerState,
        *,
        title: str = "手势选择 · 桌面图标",
    ) -> None:
        self.icon_map = icon_map
        self.state = state
        self.closed = False
        self._click_through = False
        self._transparent_ok = False
        self._shown = False
        self.hwnd = 0

        self.root = tk.Tk()
        self.root.title(title)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.configure(bg=_CHROMA)

        self.canvas = tk.Canvas(
            self.root,
            width=self.screen_w,
            height=self.screen_h,
            bg=_CHROMA,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self._highlight_id = self.canvas.create_rectangle(
            0, 0, 0, 0, outline="#5ec8ff", width=4, state="hidden",
        )
        self._caption_id = self.canvas.create_text(
            0, 0, text="", fill="#5ec8ff",
            font=("Microsoft YaHei UI", 20),
            anchor="n", state="hidden",
        )
        self._ball_glow_id = self.canvas.create_oval(
            0, 0, 0, 0, outline="#a9bacf", width=2, state="hidden",
        )
        self._ball_id = self.canvas.create_oval(
            0, 0, 0, 0, fill="#eef2f8", outline="#ffffff", width=1,
        )
        self._ball_gloss_id = self.canvas.create_oval(
            0, 0, 0, 0, fill="#ffffff", outline="",
        )
        self._hud_id = self.canvas.create_text(
            24, 24, anchor="nw", text="", fill="#9fb0c9",
            font=("Consolas", 14),
        )
        self._hint_id = self.canvas.create_text(
            self.screen_w - 24, 24, anchor="ne",
            text="ESC 退出", fill="#5f6b80",
            font=("Microsoft YaHei UI", 15),
        )

        self._ball_x = self.screen_w / 2
        self._ball_y = self.screen_h / 2
        self._halo = HaloRenderer(self.canvas, segments=20, width=4)

    # ---- 生命周期 ----

    def show(self) -> bool:
        if self._shown:
            return self._click_through and self._transparent_ok
        self._shown = True
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self.root.update()
        self.hwnd = _hwnd_of(self.root)
        self._click_through, self._transparent_ok = _apply_overlay_styles(
            self.root, _CHROMA,
        )
        print(f"[窗口] {window_state(self.root)}", file=sys.stderr)
        if not (self._click_through and self._transparent_ok):
            print(
                "[窗口] 透明或穿透设置失败，为避免挡住桌面点击，已停止。",
                file=sys.stderr,
            )
        return self._click_through and self._transparent_ok

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ---- 绘制 ----

    def refresh(self) -> None:
        icon = None
        if self.state.target is not None and self.state.target < len(self.icon_map.icons):
            icon = self.icon_map.icons[self.state.target]

        if icon is not None:
            pad = 10
            self._halo.update_box(
                icon.left - pad, icon.top - pad,
                icon.right + pad, icon.bottom + pad,
                phase=(time.monotonic() * 0.30) % 1.0,
            )
            self.canvas.itemconfigure(self._highlight_id, state="hidden")
            self.canvas.coords(
                self._caption_id,
                (icon.left + icon.right) / 2,
                icon.bottom + pad + 6,
            )
            self.canvas.itemconfigure(
                self._caption_id, text=icon.name, state="normal",
            )
            tx, ty = icon.center
        else:
            self._halo.hide()
            self.canvas.itemconfigure(self._highlight_id, state="hidden")
            self.canvas.itemconfigure(self._caption_id, state="hidden")
            tx = self.state.x * self.screen_w
            ty = self.state.y * self.screen_h

        # 自绘悬浮球已被**系统鼠标光标**取代（2026-10-19 用户要求"把悬浮球
        # 换成鼠标"）：屏幕上应该只有手里那个真实的箭头在图标之间跳，
        # 而不是多出一个圈。光晕框与名称标签保留，用来提示当前命中的是哪个。
        if not self.state.hand_visible:
            for item in (self._ball_id, self._ball_gloss_id, self._ball_glow_id):
                self.canvas.itemconfigure(item, state="hidden")
        else:
            self._ball_x += (tx - self._ball_x) * 0.35
            self._ball_y += (ty - self._ball_y) * 0.35
            try:
                ctypes.windll.user32.SetCursorPos(
                    int(round(self._ball_x)),
                    int(round(self._ball_y)),
                )
            except Exception:
                LOGGER.debug("移动系统光标失败", exc_info=True)
            for item in (self._ball_id, self._ball_gloss_id, self._ball_glow_id):
                self.canvas.itemconfigure(item, state="hidden")

        ratio = self.state.ratio
        ratio_text = "--" if ratio == float("inf") else f"{ratio:.2f}"
        locked = "（已锁定）" if self.state.pinching and icon is not None else ""
        self.canvas.itemconfigure(
            self._hud_id,
            text=(
                f"捏合比 {ratio_text}   FPS {self.state.fps:.0f}   "
                f"图标 {len(self.icon_map.icons)} 个\n"
                f"{self.state.message}{locked}"
            ),
        )

    def run(self, on_tick, interval_ms: int = 33) -> None:
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
                self.closed = True
                return
            except Exception as exc:
                self.state.message = f"错误: {exc}"
            if not self.closed:
                self.root.after(interval_ms, _loop)

        self.root.after(interval_ms, _loop)
        self.root.mainloop()

    def _quit_pressed(self) -> bool:
        try:
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
        except Exception:
            return False
