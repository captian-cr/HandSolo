"""主题桌面的纯几何模型：胶囊三键、栅格、分页、分类过滤。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from handmouse.theme_config import load_theme_config
from handmouse.theme_model import (
    PILL_BUTTONS,
    clamp_page,
    filter_items,
    grid_layout,
    page_count,
    panel_tabs,
    pill_hit,
    pill_layout,
    scale_factor,
    slice_page,
)

CFG = load_theme_config(None)


@dataclass
class _Item:
    name: str
    path: Path | str


# ---------------------------------------------------------------- 胶囊


def test_pill_is_horizontally_centered() -> None:
    layout = pill_layout(1920, 1080, CFG)

    x0, _, x1, _ = layout.pill
    assert (x0 + x1) / 2 == pytest.approx(960)


def test_pill_sits_above_the_bottom_edge() -> None:
    layout = pill_layout(1920, 1080, CFG)

    assert layout.pill[3] == pytest.approx(1080 - CFG.pill_bottom)
    assert layout.pill[3] - layout.pill[1] == pytest.approx(CFG.pill_height)


def test_three_buttons_tile_the_pill_without_overlap() -> None:
    layout = pill_layout(1920, 1080, CFG)

    boxes = [layout.boxes[name] for name in PILL_BUTTONS]
    assert [name for name in PILL_BUTTONS] == ["folder", "home", "settings"]
    for left, right in zip(boxes, boxes[1:]):
        assert left[2] == pytest.approx(right[0])
    widths = {round(box[2] - box[0], 6) for box in boxes}
    assert len(widths) == 1
    assert boxes[0][0] == pytest.approx(layout.pill[0])
    assert boxes[-1][2] == pytest.approx(layout.pill[2])


def test_pill_hit_returns_the_button() -> None:
    layout = pill_layout(1920, 1080, CFG)

    for name, (x0, y0, x1, y1) in layout.boxes.items():
        assert pill_hit(layout, (x0 + x1) / 2, (y0 + y1) / 2) == name
    assert pill_hit(layout, 10, 10) is None


def test_pill_scales_with_resolution() -> None:
    big = pill_layout(1920, 1080, CFG)
    small = pill_layout(1280, 720, CFG)

    big_w = big.pill[2] - big.pill[0]
    small_w = small.pill[2] - small.pill[0]
    assert small_w < big_w
    assert scale_factor(1920, 1080) == pytest.approx(1.0)


# ---------------------------------------------------------------- 栅格


def test_grid_is_centered() -> None:
    layout = grid_layout(12, 1920, 1080, CFG)

    left = min(x for x, _ in layout.icon_boxes)
    right = max(x for x, _ in layout.icon_boxes) + CFG.icon
    assert left == pytest.approx(1920 - right, abs=1)


def test_grid_wraps_at_configured_column_count() -> None:
    """换行位置跟随配置，不写死列数——否则调布局就会连带挂测试。"""

    cols = CFG.cols
    layout = grid_layout(cols * 2, 1920, 1080, CFG)

    first_row = {layout.icon_boxes[i][1] for i in range(cols)}
    assert len(first_row) == 1
    assert layout.icon_boxes[cols][1] > layout.icon_boxes[0][1]
    assert layout.icon_boxes[cols][0] == pytest.approx(layout.icon_boxes[0][0])


def test_labels_sit_below_icons() -> None:
    layout = grid_layout(12, 1920, 1080, CFG)

    icon_x, icon_y = layout.icon_boxes[0]
    label_x, label_y = layout.label_boxes[0]
    assert label_y > icon_y + CFG.icon
    assert label_x == pytest.approx(icon_x + CFG.icon / 2)


def test_empty_grid() -> None:
    layout = grid_layout(0, 1920, 1080, CFG)

    assert layout.icon_boxes == ()
    assert layout.label_boxes == ()


def test_grid_stays_above_the_pill() -> None:
    layout = grid_layout(12, 1920, 1080, CFG)
    pill = pill_layout(1920, 1080, CFG)

    last_bottom = max(y for _, y in layout.icon_boxes) + CFG.icon
    assert last_bottom < pill.pill[1]


# ---------------------------------------------------------------- 分页


@pytest.mark.parametrize(
    "total, per_page, expected",
    [(0, 12, 1), (1, 12, 1), (12, 12, 1), (13, 12, 2), (24, 12, 2), (25, 12, 3)],
)
def test_page_count(total: int, per_page: int, expected: int) -> None:
    assert page_count(total, per_page) == expected


def test_clamp_page() -> None:
    assert clamp_page(-3, 30, 12) == 0
    assert clamp_page(1, 30, 12) == 1
    assert clamp_page(99, 30, 12) == 2


def test_slice_page() -> None:
    items = list(range(30))

    assert slice_page(items, 0, 12) == list(range(12))
    assert slice_page(items, 2, 12) == list(range(24, 30))
    assert slice_page(items, 99, 12) == list(range(24, 30))


# ---------------------------------------------------------------- 分类过滤


def test_filter_items(tmp_path: Path) -> None:
    folder = tmp_path / "某文件夹"
    folder.mkdir()
    doc = tmp_path / "报告.docx"
    doc.write_text("x", encoding="utf-8")
    pic = tmp_path / "壁纸.png"
    pic.write_bytes(b"")
    app = tmp_path / "game.exe"
    app.write_bytes(b"")

    items = [
        _Item("某文件夹", folder),
        _Item("报告", doc),
        _Item("壁纸", pic),
        _Item("游戏", app),
        _Item("回收站", "shell:RecycleBinFolder"),
    ]

    assert len(filter_items(items, "files", "全部")) == 5
    assert [i.name for i in filter_items(items, "files", "文件夹")] == ["某文件夹"]
    assert [i.name for i in filter_items(items, "files", "文档")] == ["报告"]
    assert [i.name for i in filter_items(items, "files", "图片")] == ["壁纸"]
    apps = filter_items(items, "apps", "应用")
    assert [i.name for i in apps] == ["报告", "壁纸", "游戏", "回收站"]
    assert len(filter_items(items, "files", "不存在的分类")) == 5


def test_panel_tabs() -> None:
    labels, active = panel_tabs("files")

    assert labels[active] == "全部"
    assert "文件夹" in labels
    assert panel_tabs("未知")[0] == panel_tabs("apps")[0]
