"""主题外观配置：壁纸三种写法、尺寸夹取、压暗。"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from handmouse.theme_config import (
    DEFAULT_THEME,
    ThemeConfig,
    load_theme_config,
    parse_color,
    theme_wallpaper,
    wallpaper_kind,
)


def test_defaults_match_spec() -> None:
    cfg = load_theme_config(None)

    assert isinstance(cfg, ThemeConfig)
    assert (cfg.cols, cfg.rows) == (4, 3)
    assert cfg.per_page == 12
    assert (cfg.panel_cols, cfg.panel_rows) == (4, 2)
    assert cfg.panel_per_page == 8
    assert cfg.icon == 168
    assert cfg.wallpaper == "mesh"


def test_theme_cols_do_not_collide_with_grid_cols() -> None:
    """grid_config.json 里的 cols/rows 是 grid_select 的 2×2，不能污染主题。"""

    cfg = load_theme_config({"cols": 2, "rows": 2, "theme_cols": 5, "theme_rows": 2})

    assert (cfg.cols, cfg.rows) == (5, 2)
    assert cfg.per_page == 10


def test_out_of_range_is_clamped() -> None:
    cfg = load_theme_config({
        "icon": 10, "theme_cols": 0, "theme_rows": 99, "wallpaper_dim": 5.0,
        "pill_bottom": -40,
    })

    assert cfg.icon == 96
    assert cfg.cols == 1
    assert cfg.rows == 5
    assert cfg.wallpaper_dim == 0.9
    assert cfg.pill_bottom == 0


def test_bad_types_fall_back_to_defaults() -> None:
    cfg = load_theme_config({
        "icon": "很大", "theme_cols": None, "wallpaper": "", "pill_width": "宽",
    })

    assert cfg.icon == DEFAULT_THEME["icon"]
    assert cfg.cols == DEFAULT_THEME["theme_cols"]
    assert cfg.wallpaper == DEFAULT_THEME["wallpaper"]
    assert cfg.pill_width == DEFAULT_THEME["pill_width"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("#fff", (255, 255, 255)),
        ("#1b2a5e", (27, 42, 94)),
        ("1b2a5e", (27, 42, 94)),
        ("#12345", None),
        ("#zzzzzz", None),
        ("", None),
    ],
)
def test_parse_color(text: str, expected) -> None:
    assert parse_color(text) == expected


def test_wallpaper_kind(tmp_path: Path) -> None:
    image = tmp_path / "wall.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(image)

    assert wallpaper_kind("") == "mesh"
    assert wallpaper_kind("mesh") == "mesh"
    assert wallpaper_kind("#112233") == "solid"
    assert wallpaper_kind("#zzzzzz") == "mesh"
    assert wallpaper_kind(str(image)) == "image"
    assert wallpaper_kind(str(tmp_path / "gone.png")) == "mesh"


def test_mesh_wallpaper_size() -> None:
    img = theme_wallpaper("mesh", (48, 27), dim=0.0)

    assert img.size == (48, 27)
    assert img.mode == "RGB"


def test_solid_wallpaper_uses_that_color() -> None:
    img = theme_wallpaper("#204060", (16, 9), dim=0.0)

    assert img.getpixel((8, 4)) == (32, 64, 96)


def test_bottom_gets_darker_with_dim() -> None:
    img = theme_wallpaper("#ffffff", (8, 100), dim=0.6)

    top = img.getpixel((4, 5))[0]
    bottom = img.getpixel((4, 94))[0]
    assert bottom < top


def test_image_wallpaper_is_cover_cropped(tmp_path: Path) -> None:
    source = tmp_path / "wide.png"
    Image.new("RGB", (200, 50), (255, 0, 0)).save(source)

    img = theme_wallpaper(str(source), (40, 30), dim=0.0)

    assert img.size == (40, 30)
    assert img.getpixel((20, 15)) == (255, 0, 0)


def test_missing_image_falls_back_to_mesh(tmp_path: Path) -> None:
    img = theme_wallpaper(str(tmp_path / "nope.png"), (32, 18), dim=0.0)

    assert img.size == (32, 18)
    # 退回的 mesh 不是纯黑：确认真的画出了东西
    assert max(max(pixel) for pixel in img.getdata()) > 0
