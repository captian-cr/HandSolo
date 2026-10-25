"""``handmouse.grid_map`` 的命中判定测试。"""

from __future__ import annotations

import pytest

from handmouse.grid_map import GridLayout, map_roi, scale_roi


def simple_layout() -> GridLayout:
    """2×2、无间隙、四边留白 0.2。

    图标中心分别是 (0.35,0.35) (0.65,0.35) (0.35,0.65) (0.65,0.65)，
    相邻中心距 0.3，故磁吸半径上限为 0.15。
    """

    return GridLayout(cols=2, rows=2, margin_x=0.2, margin_y=0.2, gap=0.0)


def test_rect_geometry() -> None:
    layout = simple_layout()
    rects = layout.rects()

    assert len(rects) == 4
    assert rects[0].index == 0
    assert rects[0].col == 0 and rects[0].row == 0
    assert rects[0].cx == pytest.approx(0.35)
    assert rects[0].cy == pytest.approx(0.35)
    assert rects[1].cx == pytest.approx(0.65)
    assert rects[2].cy == pytest.approx(0.65)


def test_max_snap_radius_is_half_pitch() -> None:
    layout = simple_layout()

    assert layout.max_snap_radius == pytest.approx(0.15)


def test_hit_at_center() -> None:
    layout = simple_layout()

    for index in range(4):
        cx, cy = layout.center(index)
        assert layout.hit(cx, cy, 0.1) == index


def test_margin_is_safety_zone() -> None:
    layout = simple_layout()

    assert layout.hit(0.05, 0.35, 0.5) is None
    assert layout.hit(0.95, 0.35, 0.5) is None
    assert layout.hit(0.35, 0.05, 0.5) is None
    assert layout.hit(0.35, 0.95, 0.5) is None


def test_out_of_screen_is_none() -> None:
    layout = simple_layout()

    assert layout.hit(-0.1, 0.35, 0.5) is None
    assert layout.hit(1.2, 0.35, 0.5) is None


def test_beyond_snap_radius_is_none() -> None:
    layout = simple_layout()

    assert layout.hit(0.5, 0.35, 0.02) is None


def test_snap_radius_clamped_to_limit() -> None:
    layout = simple_layout()

    # 距图标中心 0.13：小半径够不到，上限 0.15 之内则命中。
    assert layout.hit(0.48, 0.35, 0.01) is None
    assert layout.hit(0.48, 0.35, 0.14) == 0
    # 传入远超上限的半径时会被夹到上限，结果与 0.14 一致。
    assert layout.hit(0.48, 0.35, 0.9) == 0


def test_equidistant_result_is_stable() -> None:
    layout = simple_layout()

    results = {layout.hit(0.5, 0.35, 0.15) for _ in range(20)}

    assert len(results) == 1


def test_gap_widens_pitch_and_radius() -> None:
    spaced = GridLayout(cols=2, rows=2, margin_x=0.2, margin_y=0.2, gap=0.1)
    tight = simple_layout()

    assert spaced.max_snap_radius > tight.max_snap_radius


def test_map_roi_stretches_to_unit_range() -> None:
    roi = (0.2, 0.8, 0.2, 0.8)

    assert map_roi(0.2, 0.2, roi) == pytest.approx((0.0, 0.0))
    assert map_roi(0.8, 0.8, roi) == pytest.approx((1.0, 1.0))
    assert map_roi(0.5, 0.5, roi) == pytest.approx((0.5, 0.5))


def test_map_roi_rejects_degenerate_roi() -> None:
    with pytest.raises(ValueError):
        map_roi(0.5, 0.5, (0.5, 0.5, 0.2, 0.8))


def test_invalid_layout_rejected() -> None:
    with pytest.raises(ValueError):
        GridLayout(cols=0)
    with pytest.raises(ValueError):
        GridLayout(margin_x=0.6)
    with pytest.raises(ValueError):
        GridLayout(gap=-0.1)


def test_scale_roi_shrinks_around_center() -> None:
    roi = (0.2, 0.8, 0.2, 0.8)

    scaled = scale_roi(roi, 2.0)

    # 中心不变，半宽半高各减半：手移动原来一半的距离即可覆盖全屏。
    assert scaled == pytest.approx((0.35, 0.65, 0.35, 0.65))


def test_scale_roi_one_is_identity() -> None:
    roi = (0.1, 0.9, 0.15, 0.85)

    assert scale_roi(roi, 1.0) == pytest.approx(roi)


def test_scale_roi_does_not_touch_snap_radius() -> None:
    """提灵敏度只收缩映射范围，不应改变磁吸半径的绝对值。

    若磁吸半径跟着放大，两个图标的吸引区就会重叠，退化成"指哪都可能中"。
    """

    layout = simple_layout()
    center_x, center_y = layout.center(0)
    # 在图标 0 的中心偏 0.1 处：调整灵敏度前后都该命中同一个图标。
    assert layout.hit(center_x + 0.1, center_y, 0.14) == 0
    assert layout.hit(center_x + 0.1, center_y, layout.max_snap_radius) == 0


def test_scale_roi_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        scale_roi((0.2, 0.8, 0.2, 0.8), 0.0)
