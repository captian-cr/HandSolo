"""四边独立内缩（非对称活动区）的行为约定。

背景：2026-10-21 点击标定实测，该用户的舒适指尖范围是
x∈[0.208, 0.676]、y∈[0.472, 0.818] —— **上面够不到、下面有余量**。
对称内缩 `[m, 1-m]` 无法表达这种范围，会让屏幕上方一大块永远点不到
（实测：margin_y=0.22 时上方 45% 不可达）。
"""

from __future__ import annotations

import pytest

from handmouse.config import AppConfig
from handmouse.filters import map_activity_region, map_activity_region_edges


def test_symmetric_mapping_still_equivalent_to_edges() -> None:
    """对称入口必须和四边等值调用完全一致（不能改变旧行为）。"""

    for x, y in ((0.0, 0.0), (0.1, 0.2), (0.5, 0.5), (0.95, 0.85)):
        assert map_activity_region(x, y, 0.1, 0.2) == map_activity_region_edges(
            x, y, 0.1, 0.2, 0.1, 0.2,
        )


def test_asymmetric_edges_map_range_to_full_screen() -> None:
    """把实测范围的四角喂进去，应该正好打到屏幕四角。"""

    left, top, right, bottom = 0.208, 0.472, 0.324, 0.182

    low = map_activity_region_edges(0.208, 0.472, left, top, right, bottom)
    high = map_activity_region_edges(0.676, 0.818, left, top, right, bottom)

    assert low[0] == pytest.approx(0.0, abs=1e-9)
    assert low[1] == pytest.approx(0.0, abs=1e-9)
    assert high[0] == pytest.approx(1.0, abs=1e-9)
    assert high[1] == pytest.approx(1.0, abs=1e-9)


def test_effective_margins_falls_back_to_symmetric() -> None:
    config = AppConfig()
    config.pointer_margin_x = 0.12
    config.pointer_margin_y = 0.18
    config.pointer_margin_top = None
    config.pointer_margin_bottom = None
    config.pointer_margin_left = None
    config.pointer_margin_right = None

    assert config.effective_margins() == (0.12, 0.18, 0.12, 0.18)


def test_effective_margins_prefers_per_edge_values() -> None:
    config = AppConfig()
    config.pointer_margin_x = 0.12
    config.pointer_margin_y = 0.18
    config.pointer_margin_left = 0.20
    config.pointer_margin_top = 0.45
    config.pointer_margin_right = 0.30
    config.pointer_margin_bottom = 0.16

    assert config.effective_margins() == (0.20, 0.45, 0.30, 0.16)


def test_validate_clamps_per_edge_and_keeps_active_area() -> None:
    config = AppConfig()
    config.pointer_margin_top = 0.99
    config.pointer_margin_bottom = 0.99
    config.pointer_margin_left = -0.5

    config.validate()

    assert config.pointer_margin_top == 0.5          # 上限 0.5（比对称的 0.4 宽）
    assert config.pointer_margin_left == 0.0         # 下限 0
    left, top, right, bottom = config.effective_margins()
    assert top + bottom <= 0.9 + 1e-9, "上下不能吃掉整个画面"
    assert left <= 1.0 and right <= 1.0
