"""固定图标网格的命中判定：手指坐标 -> 图标索引。

设计要点
--------
1. **等尺寸等间距图标天然构成 Voronoi 网格**，因此"判断指的地方有没有图标"
   与"找离手指最近的图标"是同一个运算。本模块只做后者，再叠一个磁吸半径
   把"太远"的情况判成无命中。
2. **磁吸半径必须小于图标间距的一半**，否则相邻图标的吸引区重叠，重叠区
   只能靠猜。``GridLayout.max_snap_radius`` 给出该上限，``validate`` 在
   构造时把越界的半径夹回去。
3. **留白是安全区**。手指落在页边留白或图标间隙之外时为无命中，系统什么都
   不做——这比误触好。

坐标约定：全部为归一化到屏幕的 ``[0, 1]``，原点左上，x 向右、y 向下。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class GridRect:
    """一个图标占据的矩形（归一化坐标）。"""

    index: int
    col: int
    row: int
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class GridLayout:
    """等尺寸图标的网格布局。

    ``margin_x`` / ``margin_y`` 是四周留白占屏幕的比例，``gap`` 是图标之间的
    相对间距（也按屏幕比例）。留白与间距一起构成安全区。
    """

    cols: int = 2
    rows: int = 2
    margin_x: float = 0.20
    margin_y: float = 0.24
    gap: float = 0.06

    def __post_init__(self) -> None:
        if self.cols < 1 or self.rows < 1:
            raise ValueError("cols/rows 必须 >= 1")
        if not 0.0 <= self.margin_x < 0.5:
            raise ValueError("margin_x 必须在 [0, 0.5) 内")
        if not 0.0 <= self.margin_y < 0.5:
            raise ValueError("margin_y 必须在 [0, 0.5) 内")
        if self.gap < 0.0:
            raise ValueError("gap 不能为负")

    # ---- 几何 ----

    @property
    def cell_width(self) -> float:
        usable = 1.0 - 2 * self.margin_x
        return (usable - (self.cols - 1) * self.gap) / self.cols

    @property
    def cell_height(self) -> float:
        usable = 1.0 - 2 * self.margin_y
        return (usable - (self.rows - 1) * self.gap) / self.rows

    def rect(self, index: int) -> GridRect:
        if not 0 <= index < self.cols * self.rows:
            raise IndexError(index)
        col = index % self.cols
        row = index // self.cols
        x0 = self.margin_x + col * (self.cell_width + self.gap)
        y0 = self.margin_y + row * (self.cell_height + self.gap)
        return GridRect(
            index=index,
            col=col,
            row=row,
            x0=x0,
            y0=y0,
            x1=x0 + self.cell_width,
            y1=y0 + self.cell_height,
        )

    def rects(self) -> tuple[GridRect, ...]:
        return tuple(self.rect(i) for i in range(self.cols * self.rows))

    @property
    def max_snap_radius(self) -> float:
        """磁吸半径上限：相邻图标中心距的一半。

        超过此值两个图标的吸引区就会重叠，重叠区无法唯一归属。
        取水平与垂直间距中较小者，保证两个方向都安全。
        """

        pitch_x = self.cell_width + self.gap
        pitch_y = self.cell_height + self.gap
        return min(pitch_x, pitch_y) / 2.0

    # ---- 命中 ----

    def hit(self, x: float, y: float, snap_radius: float) -> int | None:
        """返回命中的图标索引；落在安全区或超出磁吸半径时返回 ``None``。"""

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        # 页边留白内一律不命中：这是显式的安全区，避免手刚进画面就误选。
        if (
            x < self.margin_x
            or x > 1.0 - self.margin_x
            or y < self.margin_y
            or y > 1.0 - self.margin_y
        ):
            return None

        radius = max(0.0, min(float(snap_radius), self.max_snap_radius))
        best_index: int | None = None
        best_distance = radius
        for rect in self.rects():
            distance = hypot(x - rect.cx, y - rect.cy)
            if distance <= best_distance:
                # 严格更近才替换，保证等距时结果稳定（先出现的赢）。
                if distance < best_distance or best_index is None:
                    best_distance = distance
                    best_index = rect.index
        return best_index

    def center(self, index: int) -> tuple[float, float]:
        rect = self.rect(index)
        return rect.cx, rect.cy


def map_roi(x: float, y: float, roi: tuple[float, float, float, float]) -> tuple[float, float]:
    """把手部活动区内的坐标拉伸到整屏 ``[0, 1]``。

    ``roi`` 为 ``(x_min, x_max, y_min, y_max)``，是"手实际会活动的画面范围"。
    手通常只占画面中间一块，直接拿整帧坐标会让用户永远够不到屏幕边缘。
    """

    x_min, x_max, y_min, y_max = roi
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("ROI 的 max 必须大于 min")
    return (
        (x - x_min) / (x_max - x_min),
        (y - y_min) / (y_max - y_min),
    )


def scale_roi(
    roi: tuple[float, float, float, float],
    sensitivity: float,
) -> tuple[float, float, float, float]:
    """以中心为锚点收缩 ROI，等价于提高灵敏度。

    ``sensitivity=2.0`` 表示"手只移动原来一半的距离，就能覆盖整块屏幕"。
    收缩的是**映射范围**而不是阈值，所以磁吸半径仍是屏幕归一化坐标下的
    绝对值，不会随灵敏度一起被放大——这是"提灵敏度"和"放宽判据"的关键
    区别：前者只让手少跑路，后者会引入误触。
    """

    if sensitivity <= 0:
        raise ValueError("sensitivity 必须为正")
    x_min, x_max, y_min, y_max = roi
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half_w = (x_max - x_min) / 2 / sensitivity
    half_h = (y_max - y_min) / 2 / sensitivity
    return (cx - half_w, cx + half_w, cy - half_h, cy + half_h)
