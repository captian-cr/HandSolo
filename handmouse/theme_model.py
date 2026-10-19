"""主题桌面的纯几何模型：底部三键胶囊、抽屉面板分页、栅格定位。

为什么单独拆一个模块
--------------------
这些逻辑全是"位置算在哪、点中了谁"，但它们原本长在 tkinter 面板里，
而无显示环境（CI / 测试机）跑不起 Tk。抽到这里以后，几何与分页可以用
pytest 锁死，``theme_ui`` 只负责把这些坐标画出来。

坐标一律是 1920×1080 基准像素，运行时乘 ``factor`` 等比放大，
所以模型函数保留"基准值进、基准值出"，不感知真实分辨率。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from handmouse.theme_config import ThemeConfig

#: 底部胶囊的三个键，顺序即绘制顺序。
PILL_BUTTONS: tuple[str, ...] = ("folder", "home", "settings")

PILL_LABELS: dict[str, str] = {
    "folder": "文件",
    "home": "HOME",
    "settings": "设置",
}

#: 抽屉面板的分类标签：(标签列表, 默认选中下标)
PANEL_TABS: dict[str, tuple[tuple[str, ...], int]] = {
    "apps": (("全部", "应用"), 0),
    "files": (("全部", "文件夹", "文档", "图片"), 0),
}

#: 「文档」「图片」按后缀认。刻意只列常见项：认不出的一律归到「全部」，
#: 不猜内容，免得文件夹里混进"看起来像文档"的 exe。
DOC_SUFFIXES = frozenset({
    ".txt", ".md", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".csv", ".json",
})
IMAGE_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico",
})


def filter_items(items: list, kind: str, tab: str) -> list:
    """按分类标签过滤；``全部`` 原样返回。

    只认 ``Path`` 型路径（``shell:`` 别名是字符串，一律算作「应用」）。
    """

    if tab == "全部":
        return list(items)
    if tab == "文件夹":
        return [it for it in items if isinstance(it.path, Path) and it.path.is_dir()]
    if tab == "文档":
        return [
            it for it in items
            if isinstance(it.path, Path) and it.path.suffix.lower() in DOC_SUFFIXES
        ]
    if tab == "图片":
        return [
            it for it in items
            if isinstance(it.path, Path) and it.path.suffix.lower() in IMAGE_SUFFIXES
        ]
    if tab == "应用":
        return [it for it in items if not (isinstance(it.path, Path) and it.path.is_dir())]
    return list(items)


@dataclass(frozen=True, slots=True)
class PillLayout:
    """底部胶囊的几何：整体框 + 三个键各自的框 + 标题基线。"""

    pill: tuple[float, float, float, float]
    boxes: dict[str, tuple[float, float, float, float]]
    caption_y: float


def scale_factor(screen_w: int, screen_h: int) -> float:
    """按 1920×1080 基准等比缩放（取两边较小者，避免拉变形）。"""

    return min(screen_w / 1920.0, screen_h / 1080.0)


def pill_layout(screen_w: int, screen_h: int, cfg: ThemeConfig) -> PillLayout:
    """胶囊水平居中贴底，三键等宽平分。"""

    factor = scale_factor(screen_w, screen_h)
    width = cfg.pill_width * factor
    height = cfg.pill_height * factor
    x0 = (screen_w - width) / 2
    y1 = screen_h - cfg.pill_bottom * factor
    y0 = y1 - height

    slot = width / len(PILL_BUTTONS)
    boxes: dict[str, tuple[float, float, float, float]] = {}
    for index, name in enumerate(PILL_BUTTONS):
        bx0 = x0 + index * slot
        boxes[name] = (bx0, y0, bx0 + slot, y0 + height)

    return PillLayout(
        pill=(x0, y0, x0 + width, y1),
        boxes=boxes,
        caption_y=y1 + 14 * factor,
    )


def pill_hit(layout: PillLayout, x: float, y: float) -> str | None:
    """返回点中的键名；落在胶囊外返回 None。"""

    for name, (x0, y0, x1, y1) in layout.boxes.items():
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return None


@dataclass(frozen=True, slots=True)
class GridLayout:
    """一页图标的几何（左上角 + 名称中心点），全部是屏幕像素。"""

    icon_boxes: tuple[tuple[float, float], ...]
    label_boxes: tuple[tuple[float, float], ...]


def grid_layout(
    count: int,
    screen_w: int,
    screen_h: int,
    cfg: ThemeConfig,
    *,
    bottom_reserve: float = 200.0,
    top_reserve: float = 24.0,
) -> GridLayout:
    """按 ``cfg`` 排栅格并整体居中。

    水平居中在整屏，垂直居中在"顶部留白 → 底部胶囊上沿"之间，
    这样换行数（3 行还是 2 行）都不会贴着胶囊。
    """

    if count <= 0:
        return GridLayout(icon_boxes=(), label_boxes=())

    factor = scale_factor(screen_w, screen_h)
    icon = cfg.icon * factor
    col_pitch = cfg.col_pitch * factor
    row_pitch = cfg.row_pitch * factor
    label_gap = 15 * factor
    label_h = 22 * factor

    cols = max(1, cfg.cols)
    rows = max(1, math.ceil(count / cols))

    total_w = (cols - 1) * col_pitch + icon
    total_h = (rows - 1) * row_pitch + icon + label_gap + label_h

    origin_x = (screen_w - total_w) / 2
    band_top = top_reserve * factor
    band_bottom = screen_h - bottom_reserve * factor
    origin_y = band_top + max(0.0, (band_bottom - band_top - total_h) / 2)

    boxes: list[tuple[float, float]] = []
    labels: list[tuple[float, float]] = []
    for index in range(count):
        col = index % cols
        row = index // cols
        x = origin_x + col * col_pitch
        y = origin_y + row * row_pitch
        boxes.append((x, y))
        labels.append((x + icon / 2, y + icon + label_gap))
    return GridLayout(icon_boxes=tuple(boxes), label_boxes=tuple(labels))


def page_count(total: int, per_page: int) -> int:
    """总页数，至少 1 页（空列表也要有第 1 页可显示）。"""

    return max(1, math.ceil(total / max(1, per_page)))


def clamp_page(page: int, total: int, per_page: int) -> int:
    return max(0, min(int(page), page_count(total, per_page) - 1))


def slice_page(items: list, page: int, per_page: int) -> list:
    """取出第 ``page`` 页（0 起）的内容；越界页会被夹回最后一页。"""

    per_page = max(1, per_page)
    page = clamp_page(page, len(items), per_page)
    start = page * per_page
    return items[start:start + per_page]


def panel_tabs(kind: str) -> tuple[tuple[str, ...], int]:
    """抽屉面板的分类标签；未知类型退回 ``apps``。"""

    return PANEL_TABS.get(kind, PANEL_TABS["apps"])
