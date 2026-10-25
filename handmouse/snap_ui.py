"""主程序里的吸附高亮层：命中目标时浮出一圈动态彩色光环。

为什么单独一个文件：`desktop_select` 的 ``DesktopOverlay`` 绑定了桌面图标
列表（``icon_map``），而主程序的吸附目标可以是任意 UIA 控件、也可以是桌面
图标，只需要一个矩形 + 名字。所以把窗口与光环这部分抽出来复用，
窗口三件套（分层 / 色键透明 / 点击穿透）直接沿用 ``grid_ui`` 的实现——
漏掉 ``WS_EX_TRANSPARENT`` 会让这个全屏窗口吃掉桌面上所有点击，是最致命的一处。

与 ``DesktopOverlay`` 的另一个区别：这里不自己跑 ``mainloop``，只提供
``refresh()`` 由主循环每帧推进。OpenCV 窗口和 tkinter 共存时，两个事件循环
同时跑会互相抢，手动推进最稳。
"""

from __future__ import annotations

import logging
import time
import tkinter as tk

from handmouse.grid_ui import _CHROMA, HaloRenderer, _apply_overlay_styles


LOGGER = logging.getLogger(__name__)


class SnapHaloOverlay:
    """全屏透明、点击穿透的吸附提示层。"""

    def __init__(self, *, title: str = "手势控制 · 吸附提示") -> None:
        self.closed = False
        self._shown = False
        self._click_through = False
        self._transparent_ok = False
        self._rect: tuple[int, int, int, int] | None = None
        self._name = ""

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
        self._caption_id = self.canvas.create_text(
            0,
            0,
            text="",
            fill="#5ec8ff",
            font=("Microsoft YaHei UI", 18),
            anchor="n",
            state="hidden",
        )
        self._halo = HaloRenderer(self.canvas, segments=20, width=5)

    # ---- 生命周期 ----

    def show(self) -> bool:
        """建立窗口并设置透明/穿透。失败返回 False，调用方应直接放弃这层。"""

        if self._shown:
            return self._click_through and self._transparent_ok
        self._shown = True
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self._click_through, self._transparent_ok = _apply_overlay_styles(
            self.root,
            _CHROMA,
        )
        if not (self._click_through and self._transparent_ok):
            # 宁可没有光环，也不能让全屏窗口挡住桌面点击。
            LOGGER.warning("吸附提示层透明或穿透设置失败，已放弃该层")
            self.close()
            return False
        return True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    # ---- 内容 ----

    def update(
        self,
        rect: tuple[int, int, int, int] | None,
        name: str = "",
    ) -> None:
        """设置当前吸附目标。``rect=None`` 表示没有目标（光环隐藏）。"""

        self._rect = rect
        self._name = name

    def refresh(self) -> None:
        """推进一帧。由主循环每帧调用，不要自己起事件循环。"""

        if self.closed:
            return
        rect = self._rect
        try:
            if rect is None:
                self._halo.hide()
                self.canvas.itemconfigure(self._caption_id, state="hidden")
            else:
                left, top, right, bottom = rect
                pad = 8
                self._halo.update_box(
                    left - pad,
                    top - pad,
                    right + pad,
                    bottom + pad,
                    phase=(time.monotonic() * 0.30) % 1.0,
                )
                self.canvas.coords(
                    self._caption_id,
                    (left + right) / 2,
                    bottom + pad + 4,
                )
                self.canvas.itemconfigure(
                    self._caption_id,
                    text=self._name,
                    state="normal" if self._name else "hidden",
                )
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            self.closed = True
        except Exception:
            LOGGER.debug("吸附提示层刷新失败", exc_info=True)
