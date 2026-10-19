"""主题桌面外观参考图生成器（一次性脚本，不进主引擎）。

产出 4 张：
  docs/theme-home.png    主界面：可换壁纸 + 4x3 网格 + 底部三键胶囊 + 白线射线
  docs/theme-drawer.png  点 HOME 弹出应用抽屉（磨砂面板 + 分类标签 + 网格）
  docs/theme-files.png   点「文件夹」弹出文件面板（同一套面板壳，换标签与内容）
  docs/theme-spec.png    主界面 + 尺寸标注

关键规格
--------
  4 列 x 3 行 = 12/页，图标 168，列距 280，行距 232
  底部胶囊：文件夹 / HOME / 设置 三键（取代原来的 6 图标 dock）
  白线射线：手 -> 落点，承担手眼分离的视差补偿
  壁纸：mesh 渐变，可替换（真机上是 config 里的图片路径）

图标提取为什么用「黑底 + 白底两次绘制」
--------------------------------------
SHGetFileInfo 回来的是 32x32 的 HICON，DrawIconEx 画到 DDB 上时透明区不是
保留底色就是被写成纯黑——两种行为都取决于目标 DC，不可控。同一图标在黑、白
两种底色上各画一次，按 a = 1 - (I_white - I_black)/255 反解真实 alpha，
两种行为都能得到正确结果。

运行：
  python docs/_mockup_theme.py
"""

from __future__ import annotations

import ctypes
import math
import re
import sys
from ctypes import wintypes
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1920, 1080
SS = 2

TILE = (27, 31, 39)
TEXT = (238, 243, 250)
CORNER = 0.23

COLS = 4
ICON = 168
COL_PITCH = 280
ROW_PITCH = 232
ROW_TOPS = [136, 368, 600]
COL_CENTERS = [540, 820, 1100, 1380]

PILL_Y, PILL_H, PILL_W = 898, 112, 460
EXTRACT = 256

HOVER = (0, 3)  # 主界面被指到的那一格

# ---------------------------------------------------------------- WIN32

SHGFI_ICON = 0x100
SHGFI_LARGEICON = 0x0

_shell32 = ctypes.windll.shell32
_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32


class _SHFILEINFOW(ctypes.Structure):
    _fields_ = [
        ("hIcon", wintypes.HICON),
        ("iIcon", ctypes.c_int),
        ("dwAttributes", wintypes.DWORD),
        ("szDisplayName", ctypes.c_wchar * 260),
        ("szTypeName", ctypes.c_wchar * 80),
    ]


class _BITHDR(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


_shell32.SHGetFileInfoW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(_SHFILEINFOW),
    wintypes.UINT, wintypes.UINT,
]
_shell32.SHGetFileInfoW.restype = ctypes.c_void_p

_user32.GetDC.argtypes = [wintypes.HWND]
_user32.GetDC.restype = wintypes.HDC
_user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
_user32.FillRect.argtypes = [wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.HBRUSH]
_user32.DrawIconEx.argtypes = [
    wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.HICON,
    ctypes.c_int, ctypes.c_int, wintypes.UINT, wintypes.HBRUSH, wintypes.UINT,
]
_user32.DestroyIcon.argtypes = [wintypes.HICON]

_gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
_gdi32.CreateCompatibleDC.restype = wintypes.HDC
_gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
_gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP
_gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
_gdi32.SelectObject.restype = wintypes.HGDIOBJ
_gdi32.CreateSolidBrush.argtypes = [wintypes.DWORD]
_gdi32.CreateSolidBrush.restype = wintypes.HBRUSH
_gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
_gdi32.DeleteDC.argtypes = [wintypes.HDC]
_gdi32.GetDIBits.argtypes = [
    wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
    ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT,
]


def _draw_on(hicon: int, size: int, bg: tuple[int, int, int]) -> Image.Image:
    hdc_screen = _user32.GetDC(None)
    hdc = _gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = _gdi32.CreateCompatibleBitmap(hdc_screen, size, size)
    _gdi32.SelectObject(hdc, hbmp)

    brush = _gdi32.CreateSolidBrush(bg[0] | bg[1] << 8 | bg[2] << 16)
    _user32.FillRect(hdc, ctypes.byref(wintypes.RECT(0, 0, size, size)), brush)
    _gdi32.DeleteObject(brush)
    _user32.DrawIconEx(hdc, 0, 0, hicon, size, size, 0, None, 0x0003)

    buf = ctypes.create_string_buffer(size * size * 4)
    hdr = _BITHDR()
    hdr.biSize = ctypes.sizeof(_BITHDR)
    hdr.biWidth = size
    hdr.biHeight = -size
    hdr.biPlanes = 1
    hdr.biBitCount = 32

    _gdi32.GetDIBits(hdc, hbmp, 0, size, buf, ctypes.byref(hdr), 0)
    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(hdc)
    _user32.ReleaseDC(None, hdc_screen)
    return Image.frombuffer("RGBA", (size, size), buf.raw, "raw", "BGRA", 0, 1).convert("RGB")


def _private_icon(path: Path, size: int) -> int:
    try:
        proto = ctypes.WINFUNCTYPE(
            wintypes.UINT, wintypes.LPCWSTR, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.POINTER(wintypes.HICON),
            ctypes.POINTER(wintypes.UINT), wintypes.UINT, wintypes.UINT,
        )
        fn = proto(("PrivateExtractIconsW", _user32))
        h, cid = wintypes.HICON(), wintypes.UINT()
        if fn(str(path), 0, size, size, ctypes.byref(h), ctypes.byref(cid), 1, 0):
            return int(h.value or 0)
    except Exception:
        pass
    return 0


def _shell_icon(path: Path) -> int:
    info = _SHFILEINFOW()
    _shell32.SHGetFileInfoW(str(path), 0, ctypes.byref(info), ctypes.sizeof(info),
                            SHGFI_ICON | SHGFI_LARGEICON)
    return int(info.hIcon or 0)


def _lnk_target(lnk: Path) -> Path | None:
    try:
        data = lnk.read_bytes()
    except OSError:
        return None
    pat = re.compile(r"[A-Za-z]:\\[^\x00-\x1f\"<>|*?]{2,180}?\.exe", re.I)
    for enc in ("utf-16-le", "mbcs"):
        try:
            text = data.decode(enc, errors="ignore")
        except LookupError:
            continue
        for m in pat.finditer(text):
            cand = Path(m.group(0))
            if cand.exists():
                return cand
    return None


def _hicon_for(path: Path) -> int:
    target = _lnk_target(path) if path.suffix.lower() == ".lnk" else path
    hicon = 0
    if target is not None:
        hicon = _private_icon(target, EXTRACT)
    if not hicon:
        hicon = _private_icon(path, EXTRACT)
    if not hicon:
        hicon = _shell_icon(path)
    return hicon


def icon_rgba(path: Path | None, size: int) -> Image.Image | None:
    if path is None:
        return None
    hicon = _hicon_for(path)
    if not hicon:
        return None
    try:
        black = np.asarray(_draw_on(hicon, EXTRACT, (0, 0, 0)), dtype=np.float32)
        white = np.asarray(_draw_on(hicon, EXTRACT, (255, 255, 255)), dtype=np.float32)
    finally:
        _user32.DestroyIcon(hicon)

    alpha = np.clip(255.0 - (white - black).mean(axis=2), 0, 255) / 255.0
    color = np.zeros_like(black)
    solid = alpha > 0.06
    for c in range(3):
        color[:, :, c] = np.where(solid, black[:, :, c] / np.maximum(alpha, 0.06), 0)

    rgba = np.dstack([np.clip(color, 0, 255), alpha * 255.0]).astype(np.uint8)
    src = Image.fromarray(rgba, "RGBA")

    box = src.getchannel("A").point(lambda v: 255 if v > 26 else 0).getbbox()
    if not box:
        return None
    bw, bh = box[2] - box[0], box[3] - box[1]
    if (bw * bh) / float(EXTRACT * EXTRACT) < 0.55:
        src = src.crop(box)
        scale = 0.72
    else:
        scale = 1.0

    size_px = int(size * SS)
    inner = max(int(size_px * scale), 1)
    src = src.resize((inner, inner), Image.LANCZOS)
    tile = Image.new("RGBA", (size_px, size_px), TILE + (255,))
    tile.alpha_composite(src, ((size_px - inner) // 2, (size_px - inner) // 2))

    mask = Image.new("L", (size_px, size_px), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size_px - 1, size_px - 1), radius=int(size_px * CORNER), fill=255)
    tile.putalpha(mask)
    return tile


def font(size: int):
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_badge(img: Image.Image, text: str) -> Image.Image:
    """文件夹都是同一张黄夹子，加首字徽标才能在网格里区分。"""

    size_px = img.size[0]
    d = ImageDraw.Draw(img)
    r = size_px * 0.135
    cx = cy = size_px * 0.765
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(9, 11, 15, 232))
    f = font(max(int(r * 1.45), 8))
    t = text[:1]
    b = d.textbbox((0, 0), t, font=f)
    d.text((cx - (b[2] - b[0]) / 2 - b[0], cy - (b[3] - b[1]) / 2 - b[1]),
           t, font=f, fill=(255, 255, 255, 242))
    return img


def load_icon(path: Path | None, size: int, label: str = "") -> Image.Image:
    img = icon_rgba(path, size)
    if img is not None:
        if path is not None and path.is_dir() and label:
            img = add_badge(img, label)
        return img
    size_px = int(size * SS)
    img = Image.new("RGBA", (size_px, size_px), (46, 52, 64, 255))
    mask = Image.new("L", (size_px, size_px), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size_px - 1, size_px - 1), radius=int(size_px * CORNER), fill=255)
    img.putalpha(mask)
    if label:
        d = ImageDraw.Draw(img)
        f = font(int(size_px * 0.42))
        t = label[:1]
        b = d.textbbox((0, 0), t, font=f)
        d.text(((size_px - (b[2] - b[0])) / 2 - b[0],
                (size_px - (b[3] - b[1])) / 2 - b[1] - size_px * 0.02),
               t, font=f, fill=(255, 255, 255, 235))
    return img


# ---------------------------------------------------------------- 壁纸

def wallpaper() -> Image.Image:
    """直接调引擎里的 ``theme_wallpaper``，保证参考图与真机是同一张壁纸。"""

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from handmouse.theme_config import theme_wallpaper

    return theme_wallpaper("mesh", (W * SS, H * SS), 0.42).convert("RGBA")


# ---------------------------------------------------------------- 素材

DESKTOP = Path.home() / "Desktop"
ITEMS = [
    ("微信", Path("C:/Users/Public/Desktop/微信.lnk")),
    ("豆包", DESKTOP / "豆包.lnk"),
    ("TRAE CN", DESKTOP / "TRAE CN.lnk"),
    ("酷狗音乐", DESKTOP / "酷狗音乐.lnk"),
    ("DeepSeek", DESKTOP / "DeepSeek Harness.lnk"),
    ("OCS Desktop", DESKTOP / "OCS Desktop.lnk"),
    ("ReactBits", DESKTOP / "启动ReactBits.bat"),
    ("WorkBuddy", DESKTOP / "WorkBuddy.lnk"),
    ("Blue", DESKTOP / "Blue"),
    ("学习", DESKTOP / "学习"),
    ("湘商学院", DESKTOP / "湘商学院-办公室"),
    ("电脑", DESKTOP / "电脑"),
]

FILES = [ITEMS[9], ITEMS[10], ITEMS[11], ITEMS[3],
         ITEMS[0], ITEMS[1], ITEMS[2], ITEMS[4],
         ITEMS[7], ITEMS[5], ITEMS[6], ITEMS[8]]


# ---------------------------------------------------------------- 绘图基元

def center_text(draw, cx, y, text, fnt, fill, spacing=0):
    box = draw.textbbox((0, 0), text, font=fnt)
    tw = box[2] - box[0]
    if spacing:
        total = tw + spacing * (len(text) - 1)
        x = cx - total / 2
        for ch in text:
            b = draw.textbbox((0, 0), ch, font=fnt)
            draw.text((x - b[0], y), ch, font=fnt, fill=fill)
            x += (b[2] - b[0]) + spacing
        return
    draw.text((cx - tw / 2 - box[0], y), text, font=fnt, fill=fill)


def shadowed(layer: Image.Image, icon: Image.Image, x: float, y: float, blur: int = 9) -> None:
    sh = Image.new("RGBA", icon.size, (0, 0, 0, 0))
    sh.paste((0, 0, 0, 120), (0, 0), icon.getchannel("A"))
    layer.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur * SS)),
                          (int(x * SS), int((y + 10) * SS)))


def glyph_folder(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, col):
    wd = max(int(s * 0.16), 2)
    r = int(s * 0.24)
    d.rounded_rectangle((cx - s, cy - s * 0.40, cx + s, cy + s * 0.80),
                        radius=r, outline=col, width=wd)
    d.rounded_rectangle((cx - s, cy - s * 0.76, cx - s * 0.10, cy - s * 0.30),
                        radius=int(s * 0.16), outline=col, width=wd)


def glyph_home(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, col):
    wd = max(int(s * 0.19), 2)
    d.line([(cx - s * 0.95, cy - s * 0.05), (cx, cy - s * 0.95),
            (cx + s * 0.95, cy - s * 0.05)], fill=col, width=wd, joint="curve")
    d.line([(cx - s * 0.62, cy - s * 0.22), (cx - s * 0.62, cy + s * 0.85),
            (cx + s * 0.62, cy + s * 0.85), (cx + s * 0.62, cy - s * 0.22)],
           fill=col, width=wd, joint="curve")


def glyph_settings(d: ImageDraw.ImageDraw, cx: float, cy: float, s: float, col):
    """三行滑杆：比齿轮更容易在小尺寸下认出来。"""

    wd = max(int(s * 0.17), 2)
    for k, knob in enumerate((-0.42, 0.38, -0.10)):
        y = cy + (k - 1) * s * 0.62
        d.line([(cx - s * 0.92, y), (cx + s * 0.92, y)], fill=col, width=wd)
        kx = cx + knob * s
        r = s * 0.28
        d.ellipse((kx - r, y - r, kx + r, y + r), fill=col)


# ---------------------------------------------------------------- 主界面

def draw_pill(layer: Image.Image) -> None:
    """底部三键胶囊：文件夹 / HOME / 设置。"""

    x0 = (W - PILL_W) // 2
    pill = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    pd = ImageDraw.Draw(pill)
    pd.rounded_rectangle((x0 * SS, PILL_Y * SS, (x0 + PILL_W) * SS, (PILL_Y + PILL_H) * SS),
                         radius=PILL_H // 2 * SS, fill=(255, 255, 255, 30))
    layer.alpha_composite(pill.filter(ImageFilter.GaussianBlur(1.2 * SS)))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x0 * SS, PILL_Y * SS, (x0 + PILL_W) * SS, (PILL_Y + PILL_H) * SS),
                        radius=PILL_H // 2 * SS, outline=(255, 255, 255, 72), width=1 * SS)

    cy = PILL_Y + PILL_H / 2
    cx = W / 2
    # 两侧按钮
    for dx, fn in ((-140, glyph_folder), (140, glyph_settings)):
        bx = cx + dx
        d.ellipse(((bx - 36) * SS, (cy - 36) * SS, (bx + 36) * SS, (cy + 36) * SS),
                  fill=(255, 255, 255, 22))
        fn(d, bx * SS, cy * SS, 22 * SS, (240, 245, 252, 238))
    # 中间 HOME：实心白 + 深色字形
    d.ellipse(((cx - 42) * SS, (cy - 42) * SS, (cx + 42) * SS, (cy + 42) * SS),
              fill=(252, 253, 255, 245))
    glyph_home(d, cx * SS, cy * SS, 21 * SS, (24, 27, 34, 245))
    center_text(d, cx * SS, (PILL_Y + PILL_H + 14) * SS, "HOME",
                font(12 * SS), (255, 255, 255, 190), spacing=3 * SS)


def draw_ray(layer: Image.Image, tx: float, ty: float, ox: float = 1806, oy: float = 986) -> None:
    ray = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(ray).line((ox * SS, oy * SS, tx * SS, ty * SS),
                             fill=(255, 255, 255, 105), width=6 * SS)
    layer.alpha_composite(ray.filter(ImageFilter.GaussianBlur(6 * SS)))
    d = ImageDraw.Draw(layer)
    d.line((ox * SS, oy * SS, tx * SS, ty * SS), fill=(255, 255, 255, 212), width=2 * SS)
    d.ellipse(((tx - 6) * SS, (ty - 6) * SS, (tx + 6) * SS, (ty + 6) * SS),
              fill=(255, 255, 255, 248))

    spot = Image.new("L", layer.size, 0)
    ImageDraw.Draw(spot).ellipse(((ox - 28) * SS, (oy - 28) * SS,
                                  (ox + 28) * SS, (oy + 28) * SS), fill=55)
    hl = Image.new("RGBA", layer.size, (255, 255, 255, 0))
    hl.putalpha(spot.filter(ImageFilter.GaussianBlur(22 * SS)))
    layer.alpha_composite(hl)


def render_home(with_ray: bool = True) -> Image.Image:
    base = wallpaper()
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    lf = font(15 * SS)

    for i, (name, path) in enumerate(ITEMS):
        if i == HOVER[0] * COLS + HOVER[1]:
            continue
        row, col = divmod(i, COLS)
        x0 = COL_CENTERS[col] - ICON // 2
        y0 = ROW_TOPS[row]
        icon = load_icon(path if path.exists() else None, ICON, name)
        shadowed(layer, icon, x0, y0)
        layer.alpha_composite(icon, (x0 * SS, y0 * SS))
        center_text(draw, COL_CENTERS[col] * SS, (y0 + ICON + 15) * SS,
                    name, lf, TEXT + (232,))

    hi = HOVER[0] * COLS + HOVER[1]
    name, path = ITEMS[hi]
    hx = COL_CENTERS[HOVER[1]]
    hy = ROW_TOPS[HOVER[0]] + ICON // 2
    big = int(ICON * 1.12)

    gm = Image.new("L", layer.size, 0)
    ImageDraw.Draw(gm).ellipse(((hx - big * 0.78) * SS, (hy - big * 0.78) * SS,
                                (hx + big * 0.78) * SS, (hy + big * 0.78) * SS), fill=54)
    halo = Image.new("RGBA", layer.size, (255, 255, 255, 0))
    halo.putalpha(gm.filter(ImageFilter.GaussianBlur(22 * SS)))
    layer.alpha_composite(halo)

    big_icon = load_icon(path if path.exists() else None, big, name)
    shadowed(layer, big_icon, hx - big // 2, hy - big // 2, blur=12)
    layer.alpha_composite(big_icon, ((hx - big // 2) * SS, (hy - big // 2) * SS))
    ImageDraw.Draw(layer).rounded_rectangle(
        ((hx - big // 2) * SS, (hy - big // 2) * SS,
         (hx + big // 2) * SS, (hy + big // 2) * SS),
        radius=int(big * CORNER) * SS, outline=(255, 255, 255, 242), width=3 * SS)
    center_text(draw, hx * SS, (hy + big // 2 + 15) * SS, name, lf, (255, 255, 255, 255))

    for k in range(2):
        cxp = 960 + (k - 0.5) * 26
        r = 5 if k == 0 else 4
        ImageDraw.Draw(layer).ellipse(
            ((cxp - r) * SS, (852 - r) * SS, (cxp + r) * SS, (852 + r) * SS),
            fill=(255, 255, 255, 240 if k == 0 else 100))

    draw_pill(layer)
    if with_ray:
        draw_ray(layer, hx + big // 2 + 30, hy + 20)
    return Image.alpha_composite(base, layer)


# ---------------------------------------------------------------- 弹出面板

TABS = {
    "apps": (["全部", "应用", "媒体", "游戏"], 0),
    "files": (["全部", "文件夹", "文档", "图片"], 0),
}


def render_panel(kind: str) -> Image.Image:
    """HOME 抽屉 / 文件夹面板共用同一套壳：模糊压暗的底 + 磨砂圆角面板。"""

    bg = render_home(with_ray=False)
    bg = bg.filter(ImageFilter.GaussianBlur(7 * SS))
    scrim = Image.new("RGBA", bg.size, (6, 8, 12, 168))
    bg = Image.alpha_composite(bg, scrim)

    layer = Image.new("RGBA", bg.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)

    px0, py0, pw, ph = 220, 212, 1480, 636
    px1, py1 = px0 + pw, py0 + ph

    shadow = Image.new("RGBA", layer.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        ((px0 - 6) * SS, (py0 + 18) * SS, (px1 + 6) * SS, (py1 + 34) * SS),
        radius=44 * SS, fill=(0, 0, 0, 165))
    layer.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(26 * SS)))

    d.rounded_rectangle((px0 * SS, py0 * SS, px1 * SS, py1 * SS),
                        radius=38 * SS, fill=(23, 26, 33, 250))
    d.rounded_rectangle((px0 * SS, py0 * SS, px1 * SS, py1 * SS),
                        radius=38 * SS, outline=(255, 255, 255, 44), width=1 * SS)
    # 顶部高光
    d.rounded_rectangle((px0 * SS, py0 * SS, px1 * SS, (py0 + 2) * SS),
                        radius=38 * SS, fill=(255, 255, 255, 26))

    # 标题
    title = "全部应用" if kind == "apps" else "桌面文件"
    d.text(((px0 + 52) * SS, (py0 + 34) * SS), title, font=font(17 * SS),
           fill=(255, 255, 255, 236))

    # 分类标签
    labels, active = TABS[kind]
    tx = px0 + 52
    tf = font(14 * SS)
    for i, lb in enumerate(labels):
        b = d.textbbox((0, 0), lb, font=tf)
        tw = b[2] - b[0]
        pad = 16
        if i == active:
            d.rounded_rectangle((tx * SS, (py0 + 74) * SS, (tx + tw + pad * 2) * SS,
                                 (py0 + 108) * SS), radius=17 * SS,
                                fill=(255, 255, 255, 36))
        d.text((tx * SS + pad * SS, (py0 + 82) * SS), lb, font=tf,
               fill=(255, 255, 255, 240 if i == active else 150))
        tx += tw + pad * 2 + 12

    # 路径（文件面板才有）
    if kind == "files":
        d.text(((px1 - 52 - 260) * SS, (py0 + 34) * SS),
               "桌面 \\ 全部", font=font(12 * SS), fill=(255, 255, 255, 135))

    items = ITEMS if kind == "apps" else FILES
    n_col, pitch = 4, 340
    icon = ICON
    grid_w = (n_col - 1) * pitch + icon
    gx = px0 + (pw - grid_w) / 2 + icon / 2
    gy0 = py0 + 150
    row_pitch = 218

    lf = font(14 * SS)
    hover_idx = 3
    for i, (name, path) in enumerate(items[:8]):
        row, col = divmod(i, n_col)
        cx = gx + col * pitch
        cy = gy0 + row * row_pitch + icon / 2
        is_hover = i == hover_idx
        s = int(icon * 1.08) if is_hover else icon
        img = load_icon(path if path.exists() else None, s, name)
        shadowed(layer, img, cx - s // 2, cy - s // 2, blur=10)
        layer.alpha_composite(img, (int((cx - s // 2) * SS), int((cy - s // 2) * SS)))
        if is_hover:
            d.rounded_rectangle(((cx - s // 2) * SS, (cy - s // 2) * SS,
                                 (cx + s // 2) * SS, (cy + s // 2) * SS),
                                radius=int(s * CORNER) * SS,
                                outline=(255, 255, 255, 240), width=3 * SS)
        center_text(d, cx * SS, (cy + s // 2 + 12) * SS, name, lf,
                    (255, 255, 255, 255 if is_hover else 218))

    # 页码点（8 个一页 x 2 页）
    for k in range(2):
        cxp = 960 + (k - 0.5) * 24
        r = 4.5 if k == 0 else 3.5
        d.ellipse(((cxp - r) * SS, (py1 - 38 - r) * SS, (cxp + r) * SS, (py1 - 38 + r) * SS),
                  fill=(255, 255, 255, 235 if k == 0 else 90))

    # 白线指到面板里的一格
    row, col = divmod(hover_idx, n_col)
    hx = gx + col * pitch
    hy = gy0 + row * row_pitch + icon / 2
    draw_ray(layer, hx + icon * 0.54 + 26, hy + 16)

    return Image.alpha_composite(bg, layer)


# ---------------------------------------------------------------- 规格图

def render_spec() -> Image.Image:
    out = render_home(with_ray=True).copy()
    mark = Image.new("RGBA", out.size, (0, 0, 0, 0))
    md = ImageDraw.Draw(mark)
    f = font(13 * SS)
    ft = font(15 * SS)

    def dash(x1, y1, x2, y2, color, w=2, step=9):
        ln = math.hypot(x2 - x1, y2 - y1)
        n = max(int(ln / (step * SS)), 1)
        for i in range(n):
            if i % 2:
                continue
            a, b = i / n, min((i + 1) / n, 1)
            md.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a,
                     x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=color, width=w)

    Y, B, P, G = ((255, 214, 102, 235), (120, 220, 255, 235),
                  (255, 160, 200, 235), (180, 255, 180, 235))

    gx0 = COL_CENTERS[1] - COL_PITCH // 2
    gy0 = ROW_TOPS[2] - 30
    for a, b in (((gx0, gy0), (gx0 + COL_PITCH, gy0)),
                 ((gx0 + COL_PITCH, gy0), (gx0 + COL_PITCH, gy0 + ROW_PITCH)),
                 ((gx0 + COL_PITCH, gy0 + ROW_PITCH), (gx0, gy0 + ROW_PITCH)),
                 ((gx0, gy0 + ROW_PITCH), (gx0, gy0))):
        dash(a[0] * SS, a[1] * SS, b[0] * SS, b[1] * SS, Y)

    ix0 = COL_CENTERS[1] - ICON // 2
    iy0 = ROW_TOPS[2]
    dash((ix0 - 18) * SS, iy0 * SS, (ix0 - 18) * SS, (iy0 + ICON) * SS, B)
    for y in (iy0, iy0 + ICON):
        dash((ix0 - 26) * SS, y * SS, (ix0 - 6) * SS, y * SS, B)

    pill_x0 = (W - PILL_W) // 2
    dash(pill_x0 * SS, (PILL_Y - 16) * SS, (pill_x0 + PILL_W) * SS, (PILL_Y - 16) * SS, P)
    dash(893 * SS, 852 * SS, 1027 * SS, 852 * SS, G)

    lx, ly = 96, 470
    md.text((lx * SS, (ly - 40) * SS), "规格 · 1920 x 1080", font=ft,
            fill=(255, 255, 255, 238))
    rows = [
        (Y, "4 列 × 3 行 = 12 个 / 页"),
        (Y, "列距 280 / 行距 232（命中走最近邻 + 磁吸）"),
        (B, "图标 168 ≈ 3.5°，≥ 4× 抖动"),
        (P, "底部胶囊 460 × 112，三键：文件夹 / HOME / 设置"),
        (G, "页码点 2 页"),
        ((255, 255, 255, 235), "白线：手 → 落点，补偿视差"),
        ((255, 255, 255, 180), "壁纸：mesh 渐变，可替换"),
    ]
    for k, (col, text) in enumerate(rows):
        y = (ly + k * 40) * SS
        dash(lx * SS, y + 9 * SS, (lx + 22) * SS, y + 9 * SS, col)
        md.text(((lx + 32) * SS, y), text, font=f, fill=col)

    return Image.alpha_composite(out, mark).resize((W, H), Image.LANCZOS).convert("RGB")


def main() -> None:
    out = Path(__file__).resolve().parent
    jobs = {
        "theme-home.png": render_home(with_ray=True),
        "theme-drawer.png": render_panel("apps"),
        "theme-files.png": render_panel("files"),
        "theme-spec.png": render_spec(),
    }
    for name, img in jobs.items():
        img.resize((W, H), Image.LANCZOS).convert("RGB").save(out / name, "PNG")
        print(f"{name} ok")


if __name__ == "__main__":
    sys.exit(main())
