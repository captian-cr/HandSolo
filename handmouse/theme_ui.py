"""主题桌面：全屏应用面板（4×3 分页 + 底部三键胶囊 + 抽屉）。

为什么不再操作 Windows 桌面
--------------------------
系统桌面图标实测约 94×87 像素、两两紧挨；手指抖动折算到屏幕有 15–38 像素，
精确落进那个方框几乎不可能——这就是"很难移动到想要的位置"和"误触"的根源。
这里换成自己画的一层：图标 168×168，栅格位置固定，可以靠肌肉记忆盲指。

结构（1920×1080 基准，尺寸全部来自 ``theme_config``）
----------------------------------------------------
* 壁纸层：``mesh`` 渐变 / 纯色 / 图片，底部渐变压暗保证白线可见
* 中部 4×3 图标网格，每页 12 个，左右挥手切页
* 底部圆角胶囊：**文件 / HOME / 设置** 三个大按钮（原来的 6 图标 dock 已废弃：
  60px 的小图标低于手势精度门槛）
* HOME 弹出"全部应用"抽屉，文件弹出"桌面文件"面板，都是 4×2 = 8 个/页 的
  磨砂圆角面板

几何与分页算术在 ``theme_model``（可单测），本模块只负责画和命中。
"""

from __future__ import annotations

import ctypes
import math
import sys
import tkinter as tk
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import ImageTk

from handmouse.grid_ui import PointerState, _hue_color, _hwnd_of, window_state
from handmouse.theme_config import (
    DEFAULT_THEME,
    ThemeConfig,
    load_theme_config,
    theme_wallpaper,
)
from handmouse.theme_icons import IconCache
from handmouse.theme_model import (
    PILL_LABELS,
    PillLayout,
    clamp_page,
    filter_items,
    grid_layout as model_grid_layout,
    panel_tabs,
    pill_hit,
    pill_layout,
    scale_factor,
    slice_page,
)

#: 兼容旧调用（测试与外部模块直接从 theme_ui 取这几个常量）。
_DEFAULT = load_theme_config(None)
ICON = _DEFAULT.icon
COLS = _DEFAULT.cols
ROWS = _DEFAULT.rows
PER_PAGE = _DEFAULT.per_page

BG = "#0b0e14"
BG_CARD = "#151a23"
TEXT = "#e6ebf4"
TEXT_DIM = "#8a94a6"
PILL_BG = "#1a1f29"
PANEL_BG = "#171a21"

#: 「设置」键循环切换的内置壁纸。
BUILTIN_WALLPAPERS: tuple[str, ...] = ("mesh", "#0e1420", "#1b2a5e", "#2a1030")

_WS_EX_TRANSPARENT = 0x0000_0020
_GWL_EXSTYLE = -20


@dataclass(slots=True)
class ThemeApp:
    name: str
    #: 真实文件路径，或 ``shell:`` 别名（回收站、此电脑这类虚拟项没有路径）。
    path: Path | str


def _scale(screen_w: int, screen_h: int) -> float:
    return scale_factor(screen_w, screen_h)


def grid_layout(count: int, screen_w: int, screen_h: int, cfg: ThemeConfig | None = None):
    """兼容旧签名：返回带 ``icon_boxes`` / ``label_boxes`` 的布局对象。"""

    return model_grid_layout(
        count, screen_w, screen_h, cfg or _DEFAULT,
        bottom_reserve=_DEFAULT.pill_bottom + _DEFAULT.pill_height + 50,
    )


def _apply_click_through(root: tk.Tk) -> bool:
    """只加点击穿透位。主题面板不透明，不能再设分层（会整个不可见）。"""

    try:
        user32 = ctypes.windll.user32
        hwnd = _hwnd_of(root)
        style = user32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, _GWL_EXSTYLE, style | _WS_EX_TRANSPARENT)
        return True
    except Exception:
        return False


def _round_rect(canvas: tk.Canvas, x0, y0, x1, y1, radius, **kwargs) -> int:
    """Canvas 没有圆角矩形，用平滑多边形近似。"""

    r = max(0.0, min(radius, (x1 - x0) / 2, (y1 - y0) / 2))
    points = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r,
        x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


class ThemePanel:
    """全屏主题面板。"""

    def __init__(
        self,
        apps: list[ThemeApp],
        state: PointerState,
        *,
        files: list[ThemeApp] | None = None,
        cfg: ThemeConfig | None = None,
        title: str = "手势桌面",
    ) -> None:
        self.cfg = cfg or load_theme_config(None)
        self.apps = apps
        #: 文件面板的内容；没给就退回全部应用（桌面上本来就只有这些）。
        self.files = files if files is not None else list(apps)
        self.state = state
        self.closed = False
        self._shown = False
        self._click_through = False
        self.hwnd = 0

        self.page = 0
        self.page_count = max(1, math.ceil(len(apps) / self.cfg.per_page))

        #: ``None`` / ``"apps"``（HOME 抽屉）/ ``"files"``（文件面板）
        self.panel_kind: str | None = None
        self.panel_page = 0
        self.panel_tab = "全部"

        self.root = tk.Tk()
        self.root.title(title)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        self.screen_w = self.root.winfo_screenwidth()
        self.screen_h = self.root.winfo_screenheight()
        self.factor = _scale(self.screen_w, self.screen_h)
        self.root.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.root.configure(bg=BG)

        self.canvas = tk.Canvas(
            self.root, width=self.screen_w, height=self.screen_h,
            bg=BG, highlightthickness=0, bd=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self._icons = IconCache(int(self.cfg.icon * self.factor))
        self._panel_icons = IconCache(int(self.cfg.icon * 0.86 * self.factor))
        self._page_items: list[int] = []
        self._label_ids: dict[int, int] = {}
        self._panel_items: list[int] = []
        self._targets: list[tuple[float, float, float, float]] = []
        self._panel_targets: list[tuple[float, float, float, float]] = []
        self._tab_boxes: list[tuple[str, tuple[float, float, float, float]]] = []
        self._wallpaper_photo: ImageTk.PhotoImage | None = None
        self._wallpaper_id: int | None = None

        self.pill: PillLayout = pill_layout(self.screen_w, self.screen_h, self.cfg)

        self._draw_wallpaper()
        self._draw_page()
        self._draw_pager()
        self._draw_pill()
        self._draw_cursor()
        self._hud_id = self.canvas.create_text(
            40, 26, anchor="nw", text="", fill="#7c8aa0", font=("Consolas", 14),
        )

    # ---- 壁纸 ----

    def _draw_wallpaper(self) -> None:
        try:
            image = theme_wallpaper(
                self.cfg.wallpaper, (self.screen_w, self.screen_h),
                self.cfg.wallpaper_dim,
            )
            self._wallpaper_photo = ImageTk.PhotoImage(image)
        except Exception:
            self._wallpaper_photo = None
            return
        if self._wallpaper_id is None:
            self._wallpaper_id = self.canvas.create_image(
                0, 0, anchor="nw", image=self._wallpaper_photo,
            )
            self.canvas.tag_lower(self._wallpaper_id)
        else:
            self.canvas.itemconfigure(self._wallpaper_id, image=self._wallpaper_photo)

    def set_wallpaper(self, spec: str) -> None:
        self.cfg = replace(self.cfg, wallpaper=spec)
        self._draw_wallpaper()

    def cycle_wallpaper(self) -> str:
        try:
            index = BUILTIN_WALLPAPERS.index(self.cfg.wallpaper)
        except ValueError:
            index = -1
        spec = BUILTIN_WALLPAPERS[(index + 1) % len(BUILTIN_WALLPAPERS)]
        self.set_wallpaper(spec)
        return spec

    # ---- 页面 ----

    def _clear_page(self) -> None:
        for item in self._page_items:
            self.canvas.delete(item)
        self._page_items.clear()
        self._label_ids.clear()
        self._targets.clear()

    def _grid_bottom_reserve(self) -> float:
        """底部留给胶囊与页码点的基准像素（模型内部会再乘缩放）。"""

        return self.cfg.pill_bottom + self.cfg.pill_height + 50

    def _draw_page(self) -> None:
        self._clear_page()
        chunk = slice_page(self.apps, self.page, self.cfg.per_page)
        layout = model_grid_layout(
            len(chunk), self.screen_w, self.screen_h, self.cfg,
            bottom_reserve=self._grid_bottom_reserve(),
        )
        factor = self.factor
        size = self.cfg.icon * factor

        for offset, app in enumerate(chunk):
            x, y = layout.icon_boxes[offset]
            lx, ly = layout.label_boxes[offset]
            accent = _hue_color((self.page * self.cfg.per_page + offset) * 0.13, 0.55, 0.85)

            photo = self._icons.get(app.path, accent, app.name)
            image_id = self.canvas.create_image(x, y, anchor="nw", image=photo)
            label_id = self.canvas.create_text(
                lx, ly, text=app.name[:11], fill=TEXT,
                font=("Microsoft YaHei UI", max(10, int(14 * factor))),
                width=self.cfg.col_pitch * factor * 0.92,
            )
            self._page_items.extend([image_id, label_id])
            self._label_ids[offset] = label_id
            self._targets.append((x, y, x + size, y + size))

        self._set_page_visible(self.panel_kind is None)

    def _set_page_visible(self, visible: bool) -> None:
        state = "normal" if visible else "hidden"
        for item in self._page_items:
            self.canvas.itemconfigure(item, state=state)

    def set_page(self, page: int) -> None:
        page = clamp_page(page, len(self.apps), self.cfg.per_page)
        if page == self.page:
            return
        self.page = page
        self._draw_page()
        self._draw_pager()

    def turn_page(self, delta: int) -> None:
        if self.panel_kind is not None:
            self.turn_panel_page(delta)
            return
        self.set_page(self.page + delta)

    def home(self) -> None:
        if self.panel_kind is not None:
            self.close_panel()
            return
        self.set_page(0)

    # ---- 抽屉面板 ----

    @property
    def panel_open(self) -> bool:
        return self.panel_kind is not None

    def _panel_apps(self) -> list[ThemeApp]:
        base = self.files if self.panel_kind == "files" else self.apps
        return filter_items(list(base), self.panel_kind or "apps", self.panel_tab)

    def panel_page_count(self) -> int:
        return max(1, math.ceil(len(self._panel_apps()) / self.cfg.panel_per_page))

    def open_panel(self, kind: str) -> None:
        if kind not in ("apps", "files"):
            kind = "apps"
        self.panel_kind = kind
        self.panel_page = 0
        labels, active = panel_tabs(kind)
        self.panel_tab = labels[active]
        self._set_page_visible(False)
        self._draw_panel()

    def close_panel(self) -> None:
        if self.panel_kind is None:
            return
        self.panel_kind = None
        self.panel_page = 0
        self._clear_panel()
        self._set_page_visible(True)

    def set_panel_tab(self, label: str) -> None:
        if self.panel_kind is None:
            return
        labels, _ = panel_tabs(self.panel_kind)
        if label not in labels:
            return
        self.panel_tab = label
        self.panel_page = 0
        self._draw_panel()

    def turn_panel_page(self, delta: int) -> None:
        if self.panel_kind is None:
            return
        pages = self.panel_page_count()
        page = max(0, min(self.panel_page + delta, pages - 1))
        if page == self.panel_page:
            return
        self.panel_page = page
        self._draw_panel()

    def _clear_panel(self) -> None:
        for item in self._panel_items:
            self.canvas.delete(item)
        self._panel_items.clear()
        self._panel_targets.clear()
        self._tab_boxes.clear()

    def _draw_panel(self) -> None:
        self._clear_panel()
        if self.panel_kind is None:
            return

        factor = self.factor
        pw = 1480 * factor
        ph = 636 * factor
        px0 = (self.screen_w - pw) / 2
        py0 = (self.screen_h - ph) / 2 - 24 * factor
        px1, py1 = px0 + pw, py0 + ph

        # 遮罩：压暗底下的网格
        self._panel_items.append(self.canvas.create_rectangle(
            0, 0, self.screen_w, self.screen_h, fill="#05070c", outline="",
        ))
        self._panel_items.append(_round_rect(
            self.canvas, px0, py0, px1, py1, 38 * factor,
            fill=PANEL_BG, outline="#3a4252",
        ))

        title = "全部应用" if self.panel_kind == "apps" else "桌面文件"
        self._panel_items.append(self.canvas.create_text(
            px0 + 52 * factor, py0 + 44 * factor, anchor="w", text=title,
            fill=TEXT, font=("Microsoft YaHei UI", max(11, int(17 * factor)), "bold"),
        ))

        labels, default_active = panel_tabs(self.panel_kind)
        active = labels.index(self.panel_tab) if self.panel_tab in labels else default_active
        tx = px0 + 52 * factor
        tab_font = ("Microsoft YaHei UI", max(9, int(14 * factor)))
        for index, label in enumerate(labels):
            width = 32 * factor + len(label) * 15 * factor
            box = (tx, py0 + 74 * factor, tx + width, py0 + 108 * factor)
            if index == active:
                self._panel_items.append(_round_rect(
                    self.canvas, *box, 17 * factor, fill="#2b3240", outline="",
                ))
            self._panel_items.append(self.canvas.create_text(
                tx + width / 2, py0 + 91 * factor, text=label,
                fill=TEXT if index == active else TEXT_DIM, font=tab_font,
            ))
            self._tab_boxes.append((label, box))
            tx += width + 12 * factor

        items = slice_page(self._panel_apps(), self.panel_page, self.cfg.panel_per_page)
        if not items:
            self._panel_items.append(self.canvas.create_text(
                self.screen_w / 2, py0 + ph / 2, text="这个分类下没有内容",
                fill=TEXT_DIM, font=("Microsoft YaHei UI", max(10, int(15 * factor))),
            ))
        cols = self.cfg.panel_cols
        icon = self.cfg.icon * 0.86 * factor
        pitch = self.cfg.col_pitch * factor
        row_pitch = self.cfg.row_pitch * 0.94 * factor
        total_w = (cols - 1) * pitch + icon
        gx = px0 + (pw - total_w) / 2
        gy = py0 + 150 * factor

        for offset, app in enumerate(items):
            row, col = divmod(offset, cols)
            x = gx + col * pitch
            y = gy + row * row_pitch
            photo = self._panel_icons.get(app.path, "#3a4354", app.name)
            self._panel_items.append(
                self.canvas.create_image(x, y, anchor="nw", image=photo),
            )
            self._panel_items.append(self.canvas.create_text(
                x + icon / 2, y + icon + 12 * factor, text=app.name[:11],
                fill=TEXT, font=("Microsoft YaHei UI", max(9, int(13 * factor))),
                width=pitch * 0.9,
            ))
            self._panel_targets.append((x, y, x + icon, y + icon))

        pages = self.panel_page_count()
        if pages > 1:
            cx = self.screen_w / 2 - (pages - 1) * 12 * factor
            for index in range(pages):
                r = (4.5 if index == self.panel_page else 3.5) * factor
                self._panel_items.append(self.canvas.create_oval(
                    cx - r, py1 - 38 * factor - r, cx + r, py1 - 38 * factor + r,
                    fill="#e8ecf5" if index == self.panel_page else "#3a4354",
                    outline="",
                ))
                cx += 24 * factor

    # ---- 底部胶囊 / 页码 ----

    def _draw_pill(self) -> None:
        self.pill = pill_layout(self.screen_w, self.screen_h, self.cfg)
        factor = self.factor
        x0, y0, x1, y1 = self.pill.pill
        self._pill_items: list[int] = [
            _round_rect(
                self.canvas, x0, y0, x1, y1, (y1 - y0) / 2,
                fill=PILL_BG, outline="#39414f",
            )
        ]

        cy = (y0 + y1) / 2
        for name in ("folder", "home", "settings"):
            bx0, _, bx1, _ = self.pill.boxes[name]
            cx = (bx0 + bx1) / 2
            if name == "home":
                r = 42 * factor
                self._pill_items.append(self.canvas.create_oval(
                    cx - r, cy - r, cx + r, cy + r, fill="#f5f8ff", outline="",
                ))
                self._draw_home_glyph(cx, cy, 21 * factor, "#171b23")
                self._pill_items.append(self.canvas.create_text(
                    cx, self.pill.caption_y, text=PILL_LABELS[name],
                    fill="#c9d3e3", font=("Consolas", max(9, int(11 * factor))),
                ))
                continue

            r = 34 * factor
            self._pill_items.append(self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r, fill="#252b36", outline="#3d4657",
            ))
            if name == "folder":
                self._draw_folder_glyph(cx, cy, 20 * factor, "#eef2f8")
            else:
                self._draw_settings_glyph(cx, cy, 20 * factor, "#eef2f8")

    def _draw_folder_glyph(self, cx: float, cy: float, s: float, color: str) -> None:
        w = max(2, int(2 * self.factor))
        self._pill_items.append(self.canvas.create_line(
            cx - s, cy - s * 0.10, cx - s, cy + s * 0.80, cx + s, cy + s * 0.80,
            cx + s, cy - s * 0.30, cx + s * 0.1, cy - s * 0.30,
            fill=color, width=w, capstyle="round",
        ))
        self._pill_items.append(self.canvas.create_line(
            cx - s, cy - s * 0.52, cx - s * 0.15, cy - s * 0.52,
            fill=color, width=w, capstyle="round",
        ))

    def _draw_home_glyph(self, cx: float, cy: float, s: float, color: str) -> None:
        w = max(2, int(2 * self.factor))
        self._pill_items.append(self.canvas.create_line(
            cx - s, cy - s * 0.05, cx, cy - s * 0.95, cx + s, cy - s * 0.05,
            fill=color, width=w, capstyle="round",
        ))
        self._pill_items.append(self.canvas.create_line(
            cx - s * 0.62, cy - s * 0.20, cx - s * 0.62, cy + s * 0.85,
            cx + s * 0.62, cy + s * 0.85, cx + s * 0.62, cy - s * 0.20,
            fill=color, width=w, capstyle="round",
        ))

    def _draw_settings_glyph(self, cx: float, cy: float, s: float, color: str) -> None:
        w = max(2, int(2 * self.factor))
        for k, knob in enumerate((-0.42, 0.38, -0.10)):
            y = cy + (k - 1) * s * 0.62
            self._pill_items.append(self.canvas.create_line(
                cx - s * 0.92, y, cx + s * 0.92, y, fill=color, width=w,
            ))
            r = s * 0.26
            kx = cx + knob * s
            self._pill_items.append(self.canvas.create_oval(
                kx - r, y - r, kx + r, y + r, fill=color, outline="",
            ))

    def _draw_pager(self) -> None:
        for item in getattr(self, "_pager_items", []):
            self.canvas.delete(item)
        self._pager_items: list[int] = []

        factor = self.factor
        radius = 4 * factor
        gap = 18 * factor
        total = self.page_count * (2 * radius) + (self.page_count - 1) * gap
        cx = (self.screen_w - total) / 2 + radius
        cy = self.pill.pill[1] - 26 * factor

        for index in range(self.page_count):
            active = index == self.page
            self._pager_items.append(self.canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#e8ecf5" if active else "#3a4354", outline="",
            ))
            cx += 2 * radius + gap

    # ---- 光标 ----

    def _draw_cursor(self) -> None:
        self._cursor_ring = self.canvas.create_oval(
            0, 0, 0, 0, outline="#c9d3e3", width=2, state="hidden",
        )
        self._cursor_dot = self.canvas.create_oval(
            0, 0, 0, 0, fill="#ffffff", outline="", state="hidden",
        )

    # ---- 命中 ----

    def hit(self, x: float, y: float) -> int | None:
        """返回命中的图标在**当前上下文**里的下标（网格页或抽屉页）。"""

        targets = self._panel_targets if self.panel_open else self._targets
        margin = 22 * self.factor
        for index, (x0, y0, x1, y1) in enumerate(targets):
            if (
                x0 - margin <= x <= x1 + margin
                and y0 - margin <= y <= y1 + margin
            ):
                return index
        return None

    def pill_button(self, x: float, y: float) -> str | None:
        return pill_hit(self.pill, x, y)

    def tab_hit(self, x: float, y: float) -> str | None:
        for label, (x0, y0, x1, y1) in self._tab_boxes:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return label
        return None

    def current_apps(self) -> list[ThemeApp]:
        if self.panel_open:
            return slice_page(self._panel_apps(), self.panel_page, self.cfg.panel_per_page)
        return slice_page(self.apps, self.page, self.cfg.per_page)

    # ---- 刷新 ----

    def refresh(self) -> None:
        target = self.state.target
        factor = self.factor

        for index, item in self._label_ids.items():
            try:
                self.canvas.itemconfigure(
                    item, fill="#ffffff" if index == target else TEXT,
                )
            except tk.TclError:
                pass

        cx = self.state.x * self.screen_w
        cy = self.state.y * self.screen_h
        if not self.state.hand_visible or self.state.pose != "point":
            self.canvas.itemconfigure(self._cursor_ring, state="hidden")
            self.canvas.itemconfigure(self._cursor_dot, state="hidden")
        else:
            r = (13.0 if self.state.pinching else 19.0) * factor
            ring_color = "#ffffff" if self.state.pinching else "#c9d3e3"
            self.canvas.itemconfigure(self._cursor_ring, outline=ring_color)
            self.canvas.coords(self._cursor_ring, cx - r, cy - r, cx + r, cy + r)
            self.canvas.itemconfigure(self._cursor_ring, state="normal")
            self.canvas.coords(self._cursor_dot, cx - 2, cy - 2, cx + 2, cy + 2)
            self.canvas.itemconfigure(self._cursor_dot, state="normal")

        ratio = self.state.ratio
        ratio_text = "--" if ratio == float("inf") else f"{ratio:.2f}"
        pose_label = {
            "point": "食指操控", "open": "五指张开·挥手翻页",
            "fist": "握拳（空闲）", "other": "姿态未识别",
        }.get(self.state.pose, self.state.pose)
        where = (
            f"面板：{'全部应用' if self.panel_kind == 'apps' else '桌面文件'}"
            f" {self.panel_page + 1}/{self.panel_page_count()}"
            if self.panel_open
            else f"第 {self.page + 1}/{self.page_count} 页"
        )
        self.canvas.itemconfigure(
            self._hud_id,
            text=(
                f"{pose_label}　FPS {self.state.fps:.0f}　捏合比 {ratio_text}　{where}\n"
                f"{self.state.message}"
            ),
        )

    # ---- 生命周期 ----

    def show(self) -> bool:
        if self._shown:
            return True
        self._shown = True
        self.root.deiconify()
        self.root.lift()
        self.root.update_idletasks()
        self.root.update()
        self.hwnd = _hwnd_of(self.root)
        self._click_through = _apply_click_through(self.root)
        print(
            f"[面板] {window_state(self.root)} 穿透={self._click_through}",
            file=sys.stderr,
        )
        return True

    def close(self) -> None:
        self.closed = True
        try:
            self.root.destroy()
        except tk.TclError:
            pass

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
