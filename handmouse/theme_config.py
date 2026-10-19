"""主题桌面的外观配置：壁纸与栅格尺寸。

壁纸支持三种写法（都在 ``grid_config.json`` 的 ``wallpaper`` 里）
------------------------------------------------------------------
* ``"mesh"``         内置 iOS 风格的多色团渐变（默认）
* ``"#1b2a5e"``      纯色
* ``"D:/wall/a.jpg"`` 图片路径（必须存在，否则退回 mesh）

为什么壁纸要压暗
----------------
白线射线、白色标签、深色玻璃胶囊都靠"高对比"读出来。浅色壁纸会让白线
彻底消失——所以无论哪种壁纸，底部都叠一层渐变压暗（``wallpaper_dim``）。

只依赖 Pillow：引擎不该为了画个渐变再拖进 numpy。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


#: iOS 风格色团：(横向比例, 纵向比例, 半径比例, RGB)
MESH_BLOBS: tuple[tuple[float, float, float, tuple[int, int, int]], ...] = (
    (0.10, 0.02, 0.62, (54, 68, 164)),     # 左上 靛蓝
    (0.92, 0.10, 0.56, (124, 62, 174)),    # 右上 紫
    (0.86, 0.98, 0.60, (16, 134, 148)),    # 右下 青
    (0.06, 0.94, 0.56, (172, 54, 96)),     # 左下 玫红
)

#: 叠完色团后整体压暗的比例：太亮会吃掉白色标签和白线。
MESH_BRIGHTNESS = 0.72

MESH_BASE = (16, 22, 40)
#: 小画布尺寸：太大就慢，太小色团会被糊成一团（实测 64×36 会糊成纯渐变）。
MESH_SAMPLE = (240, 135)

DEFAULT_THEME: dict = {
    "wallpaper": "mesh",
    "wallpaper_dim": 0.42,
    "icon": 168,
    # 用 theme_cols / theme_rows 而不是 cols / rows：grid_config.json 里
    # 的 cols / rows 是 grid_select 自己的 2×2 网格，同名会互相污染。
    "theme_cols": 4,
    "theme_rows": 3,
    "col_pitch": 280,
    "row_pitch": 232,
    "panel_cols": 4,
    "panel_rows": 2,
    "pill_width": 460,
    "pill_height": 112,
    "pill_bottom": 70,
}

_ICON_RANGE = (96, 320)
_PITCH_RANGE = (140, 520)
_COLS_RANGE = (1, 8)
_ROWS_RANGE = (1, 5)


@dataclass(frozen=True, slots=True)
class ThemeConfig:
    """主题桌面的全部可调尺寸，单位是 1920×1080 基准像素。"""

    wallpaper: str = "mesh"
    wallpaper_dim: float = 0.42
    icon: int = 168
    cols: int = 4
    rows: int = 3
    col_pitch: int = 280
    row_pitch: int = 232
    panel_cols: int = 4
    panel_rows: int = 2
    pill_width: int = 460
    pill_height: int = 112
    pill_bottom: int = 70

    @property
    def per_page(self) -> int:
        return max(1, self.cols * self.rows)

    @property
    def panel_per_page(self) -> int:
        return max(1, self.panel_cols * self.panel_rows)


def _clamp_int(value: object, low: int, high: int, default: int) -> int:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _clamp_float(value: object, low: float, high: float, default: float) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def load_theme_config(raw: dict | None = None) -> ThemeConfig:
    """把 ``grid_config.json`` 里的主题字段揉成不可变配置，越界一律夹回。

    配置是给人手改的，写错（``"icon": "大"``、负数列距）不能把界面画崩，
    所以每一项都夹取，坏值退回默认。
    """

    source = raw or {}
    data = dict(DEFAULT_THEME)
    for key in DEFAULT_THEME:
        if key in source and source[key] is not None:
            data[key] = source[key]

    wallpaper = data.get("wallpaper")
    if not isinstance(wallpaper, str) or not wallpaper.strip():
        wallpaper = DEFAULT_THEME["wallpaper"]

    return ThemeConfig(
        wallpaper=wallpaper.strip(),
        wallpaper_dim=_clamp_float(
            data.get("wallpaper_dim"), 0.0, 0.9, DEFAULT_THEME["wallpaper_dim"],
        ),
        icon=_clamp_int(data.get("icon"), *_ICON_RANGE, DEFAULT_THEME["icon"]),
        cols=_clamp_int(data.get("theme_cols"), *_COLS_RANGE, DEFAULT_THEME["theme_cols"]),
        rows=_clamp_int(data.get("theme_rows"), *_ROWS_RANGE, DEFAULT_THEME["theme_rows"]),
        col_pitch=_clamp_int(
            data.get("col_pitch"), *_PITCH_RANGE, DEFAULT_THEME["col_pitch"],
        ),
        row_pitch=_clamp_int(
            data.get("row_pitch"), *_PITCH_RANGE, DEFAULT_THEME["row_pitch"],
        ),
        panel_cols=_clamp_int(
            data.get("panel_cols"), *_COLS_RANGE, DEFAULT_THEME["panel_cols"],
        ),
        panel_rows=_clamp_int(
            data.get("panel_rows"), *_ROWS_RANGE, DEFAULT_THEME["panel_rows"],
        ),
        pill_width=_clamp_int(data.get("pill_width"), 240, 1200, DEFAULT_THEME["pill_width"]),
        pill_height=_clamp_int(
            data.get("pill_height"), 64, 240, DEFAULT_THEME["pill_height"],
        ),
        pill_bottom=_clamp_int(data.get("pill_bottom"), 0, 300, DEFAULT_THEME["pill_bottom"]),
    )


def parse_color(text: str) -> tuple[int, int, int] | None:
    """``#rgb`` / ``#rrggbb`` → RGB；不是颜色就返回 None。"""

    raw = text.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        return None
    try:
        return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)
    except ValueError:
        return None


def wallpaper_kind(spec: str) -> str:
    """返回 ``mesh`` / ``solid`` / ``image``；路径不存在时退回 ``mesh``。"""

    text = (spec or "").strip()
    if not text or text.lower() == "mesh":
        return "mesh"
    if text.startswith("#"):
        return "solid" if parse_color(text) is not None else "mesh"
    return "image" if Path(text).is_file() else "mesh"


def _mesh(size: tuple[int, int]) -> Image.Image:
    """先在小画布上叠色团再放大：等价于大半径模糊，而且快得多。

    半径按**短边**算。按长边算会让纵向铺满、横向溢出，五个色团糊成一块
    纯渐变（这版第一眼就是"没有色"）。
    """

    small = Image.new("RGB", MESH_SAMPLE, MESH_BASE)
    short = min(MESH_SAMPLE)
    for fx, fy, fr, color in MESH_BLOBS:
        layer = Image.new("RGBA", MESH_SAMPLE, (0, 0, 0, 0))
        cx, cy = fx * MESH_SAMPLE[0], fy * MESH_SAMPLE[1]
        radius = fr * short
        ImageDraw.Draw(layer).ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(*color, 205),
        )
        small = Image.alpha_composite(
            small.convert("RGBA"),
            layer.filter(ImageFilter.GaussianBlur(8)),
        ).convert("RGB")

    small = small.filter(ImageFilter.GaussianBlur(5))
    img = small.resize(size, Image.BICUBIC)
    return ImageEnhance.Brightness(img).enhance(MESH_BRIGHTNESS)


def _cover(path: Path, size: tuple[int, int]) -> Image.Image:
    """等比裁切填满（宁可裁掉边角，也不留白也不拉伸）。"""

    img = Image.open(path).convert("RGB")
    scale = max(size[0] / img.width, size[1] / img.height)
    img = img.resize(
        (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
        Image.LANCZOS,
    )
    left = (img.width - size[0]) // 2
    top = (img.height - size[1]) // 2
    return img.crop((left, top, left + size[0], top + size[1]))


def _dim_bottom(img: Image.Image, strength: float) -> Image.Image:
    """底部渐变压暗，让白线与胶囊有对比。"""

    if strength <= 0:
        return img
    w, h = img.size
    grad = Image.new("L", (1, h))
    for y in range(h):
        t = max(0.0, (y / h - 0.35) / 0.65)
        grad.putpixel((0, y), int(255 * strength * (t ** 1.6)))
    grad = grad.resize((w, h))
    return Image.composite(Image.new("RGB", (w, h), (0, 0, 0)), img, grad)


def theme_wallpaper(spec: str, size: tuple[int, int], dim: float = 0.42) -> Image.Image:
    """按规格生成壁纸位图；任何失败都退回内置 mesh，界面不会空白。"""

    kind = wallpaper_kind(spec)
    if kind == "solid":
        color = parse_color(spec) or MESH_BASE
        img = Image.new("RGB", size, color)
    elif kind == "image":
        try:
            img = _cover(Path(spec.strip()), size)
        except OSError:
            img = _mesh(size)
    else:
        img = _mesh(size)
    return _dim_bottom(img, dim)
