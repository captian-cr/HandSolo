"""桌面图标命中判定与路径解析测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from handmouse import grid_desktop
from handmouse.desktop_ui import DesktopIconMap
from handmouse.grid_desktop import DesktopIcon, resolve_icon_path


def make_icons() -> list[DesktopIcon]:
    return [
        DesktopIcon(index=0, name="A", left=0, top=0, right=100, bottom=100),
        DesktopIcon(index=1, name="B", left=100, top=0, right=200, bottom=100),
        DesktopIcon(index=2, name="C", left=0, top=200, right=100, bottom=300),
    ]


def test_hit_picks_nearest_icon() -> None:
    icon_map = DesktopIconMap(make_icons(), max_pick_px=120.0)

    assert icon_map.hit(50, 50) == 0
    assert icon_map.hit(150, 50) == 1
    assert icon_map.hit(50, 250) == 2


def test_hit_boundary_goes_to_closer_side() -> None:
    icon_map = DesktopIconMap(make_icons(), max_pick_px=120.0)

    # 两个图标中心在 x=50 与 x=150，分界在 100。
    assert icon_map.hit(99, 50) == 0
    assert icon_map.hit(101, 50) == 1


def test_hit_returns_none_beyond_limit() -> None:
    """空白区域必须判为无命中，否则手放在屏幕下半部分会误选第一行。"""

    icon_map = DesktopIconMap(make_icons(), max_pick_px=120.0)

    assert icon_map.hit(50, 700) is None
    assert icon_map.hit(1200, 50) is None


def test_hit_returns_none_for_empty_map() -> None:
    assert DesktopIconMap([], max_pick_px=120.0).hit(50, 50) is None


def test_pick_limit_is_respected() -> None:
    icons = make_icons()
    tight = DesktopIconMap(icons, max_pick_px=60.0)
    loose = DesktopIconMap(icons, max_pick_px=200.0)

    # (150,150) 距 B 中心正好 100px：紧的够不到，松的能吸到 B。
    assert tight.hit(150, 150) is None
    assert loose.hit(150, 150) == 1


def test_resolve_icon_path_tries_suffixes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "微信.lnk").write_text("", encoding="utf-8")
    folder = tmp_path / "项目"
    folder.mkdir()
    monkeypatch.setattr(grid_desktop, "desktop_dirs", lambda: [tmp_path])

    assert resolve_icon_path("微信") == tmp_path / "微信.lnk"
    assert resolve_icon_path("项目") == folder
    assert resolve_icon_path("不存在") is None
    assert resolve_icon_path("") is None
