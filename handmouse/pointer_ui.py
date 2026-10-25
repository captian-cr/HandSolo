"""指针模式的叠加层：一个环套住真实光标，标记手指指着哪。

为什么用环而不是实心球
----------------------
这里球是**套在系统光标外围**的，光标本身必须一直可见（用户要靠它看准点击
落点）。所以画成一个中空的环 + 中心一个小点：环标位置，中心不遮挡光标。
实心球会把光标盖住，反而看不出精确落点。
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk

from handmouse.grid_ui import (
    _CHROMA,
    PointerState,
    _apply_overlay_styles,
    _hwnd_of,
    window_state,
)


class PointerOverlay:
    """全屏透明层：环 + 中心点 + HUD。"""

    def __init__(self, state: PointerState, *, title: str = "手势指针") -> None:
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

        # 外环：浅灰细线，压住位置但不抢视线。
        self._ring_id = self.canvas.create_oval(
            0, 0, 0, 0, outline="#d5dce8", width=2, state="hidden",
        )
        # 内环：半格大小，给"越近越准"的视觉梯度。
        self._inner_id = self.canvas.create_oval(
            0, 0, 0, 0, outline="#eef2f8", width=1, state="hidden",
        )
        # 中心点：标记光标的精确落点。
        self._dot_id = self.canvas.create_oval(
            0, 0, 0, 0, fill="#ffffff", outline="#8fa3bf", width=1, state="hidden",
        )
        self._hud_id = self.canvas.create_text(
            24, 24, anchor="nw", text="", fill="#9fb0c9",
            font=("Consolas", 14),
        )
        self._hint_id = self.canvas.create_text(
            self.screen_w - 24, 24, anchor="ne",
            text="M 暂停　ESC 退出", fill="#5f6b80",
            font=("Microsoft YaHei UI", 15),
        )

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
        cx = self.state.x * self.screen_w
        cy = self.state.y * self.screen_h
        visible = self.state.hand_visible

        if not visible:
            for item in (self._ring_id, self._inner_id, self._dot_id):
                self.canvas.itemconfigure(item, state="hidden")
        else:
            r = 16.0 if self.state.pinching else 24.0
            # 颜色只跟着"手"的状态走：捏合=白，吸住=蓝，其余=灰。
            # 待定双击（pending）不靠颜色表达——否则松手后颜色还挂着，
            # 会让人以为手还按着。它改用 HUD 文字 + 环加粗来提示。
            if self.state.pinching:
                ring_color = "#ffffff"
            elif self.state.snapped:
                ring_color = "#7dd3fc"
            else:
                ring_color = "#d5dce8"
            self.canvas.itemconfigure(
                self._ring_id,
                outline=ring_color,
                width=4 if self.state.pending else 2,
            )
            self.canvas.coords(
                self._ring_id, cx - r, cy - r, cx + r, cy + r,
            )
            self.canvas.itemconfigure(self._ring_id, state="normal")
            inner = r * 0.45
            self.canvas.coords(
                self._inner_id, cx - inner, cy - inner, cx + inner, cy + inner,
            )
            self.canvas.itemconfigure(self._inner_id, state="normal")
            self.canvas.coords(self._dot_id, cx - 2, cy - 2, cx + 2, cy + 2)
            self.canvas.itemconfigure(self._dot_id, state="normal")

        ratio = self.state.ratio
        ratio_text = "--" if ratio == float("inf") else f"{ratio:.2f}"
        paused = "【已暂停】" if self.state.paused else ""
        self.canvas.itemconfigure(
            self._hud_id,
            text=(
                f"捏合比 {ratio_text}   FPS {self.state.fps:.0f}   "
                f"{'捏合中' if self.state.pinching else '张开'}\n"
                f"{paused}{self.state.message}"
            ),
        )

    def run(self, on_tick, interval_ms: int = 16) -> None:
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
