"""主题面板布局与应用装载测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from handmouse import grid_desktop
from handmouse.theme_select import load_theme_apps
from handmouse.theme_ui import ICON, PER_PAGE, grid_layout


def test_cells_are_centered() -> None:
    layout = grid_layout(13, 1920, 1080)

    assert len(layout.icon_boxes) == 13
    left = min(x for x, _ in layout.icon_boxes)
    right = max(x for x, _ in layout.icon_boxes) + ICON
    assert left == pytest.approx(1920 - right, abs=1)


def test_four_columns_and_row_break() -> None:
    layout = grid_layout(13, 1920, 1080)

    # 前四个同一行，y 相同、x 递增。
    first_row_y = {layout.icon_boxes[i][1] for i in range(4)}
    assert len(first_row_y) == 1
    assert (
        layout.icon_boxes[0][0]
        < layout.icon_boxes[1][0]
        < layout.icon_boxes[3][0]
    )
    # 第 5 个换行，并回到第一列的 x。
    assert layout.icon_boxes[4][1] > layout.icon_boxes[0][1]
    assert layout.icon_boxes[4][0] == pytest.approx(layout.icon_boxes[0][0])


def test_per_page_is_twelve() -> None:
    assert PER_PAGE == 12
    assert len(grid_layout(PER_PAGE, 1920, 1080).icon_boxes) == PER_PAGE


def test_labels_sit_below_icons() -> None:
    layout = grid_layout(13, 1920, 1080)

    icon_x, icon_y = layout.icon_boxes[0]
    label_x, label_y = layout.label_boxes[0]
    assert label_y > icon_y + ICON
    assert label_x == pytest.approx(icon_x + ICON / 2)


def test_empty_layout() -> None:
    layout = grid_layout(0, 1920, 1080)

    assert layout.icon_boxes == ()
    assert layout.label_boxes == ()


def test_scales_with_resolution() -> None:
    """1080p 是基准；720p 下排布跨度应当等比缩小。"""

    base = grid_layout(9, 1920, 1080)
    small = grid_layout(9, 1280, 720)

    base_span = max(x for x, _ in base.icon_boxes) - min(x for x, _ in base.icon_boxes)
    small_span = max(x for x, _ in small.icon_boxes) - min(x for x, _ in small.icon_boxes)
    assert small_span < base_span


def _fake_icons(names: list[str]) -> list:
    from handmouse.grid_desktop import DesktopIcon

    return [
        DesktopIcon(index=i, name=n, left=0, top=0, right=80, bottom=80)
        for i, n in enumerate(names)
    ]


def test_load_theme_apps_maps_virtual_items(monkeypatch: pytest.MonkeyPatch) -> None:
    """回收站这类虚拟项没有文件路径，要映射成 shell 别名而不是丢掉。"""

    monkeypatch.setattr(
        grid_desktop, "read_desktop_icons",
        lambda: _fake_icons(["回收站", "Chrome", "未知项"]),
    )
    monkeypatch.setattr(
        grid_desktop, "resolve_icon_path",
        lambda name: Path("C:/Apps/Chrome.lnk") if name == "Chrome" else None,
    )

    apps = load_theme_apps()

    # 解析不到文件、也不在别名表里的条目会被跳过。
    assert [app.name for app in apps] == ["回收站", "Chrome"]
    assert apps[0].path == "shell:RecycleBinFolder"


def test_load_theme_apps_respects_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    names = [f"app{index:02d}" for index in range(12)]
    monkeypatch.setattr(
        grid_desktop, "read_desktop_icons", lambda: _fake_icons(names),
    )
    monkeypatch.setattr(
        grid_desktop, "resolve_icon_path",
        lambda name: Path(f"C:/Apps/{name}.lnk"),
    )

    assert len(load_theme_apps(limit=9)) == 9
