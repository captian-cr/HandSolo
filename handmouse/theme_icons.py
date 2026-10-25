"""应用图标提取与 iOS 风格圆角处理。

为什么需要一层处理
------------------
1. `.lnk` 快捷方式本身**不含图标**，必须让 Shell 解析（`SHGetFileInfoW`）。
2. 系统给的是 32×32 的小图标，直接贴到大卡片上会糊。
3. iOS 风格要求"圆角方形 + 裁切"，Windows 图标多是透明底异形，需要套遮罩。

失败策略
--------
任何一步失败都返回 ``None``，由调用方退回"色块 + 首字"。取图标依赖 Shell
图标缓存，在某些受限会话里拿不到——界面不能因此空白或崩溃。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageTk

SHGFI_ICON = 0x0000_0100
SHGFI_LARGEICON = 0x0000_0000
SHGFI_USEFILEATTRIBUTES = 0x0000_0010

#: 圆角半径占比。iOS 的 squircle 比等半径圆角更"方"，取 0.23 接近。
CORNER_RATIO = 0.23


class _SHFILEINFOW(ctypes.Structure):
    _fields_ = [
        ("hIcon", wintypes.HICON),
        ("iIcon", ctypes.c_int),
        ("dwAttributes", wintypes.DWORD),
        ("szDisplayName", ctypes.c_wchar * 260),
        ("szTypeName", ctypes.c_wchar * 80),
    ]


_shell32 = ctypes.windll.shell32
_shell32.SHGetFileInfoW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD,
    ctypes.POINTER(_SHFILEINFOW), wintypes.UINT, wintypes.UINT,
]
_shell32.SHGetFileInfoW.restype = ctypes.c_void_p

_user32 = ctypes.windll.user32
_gdi32 = ctypes.windll.gdi32


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


def _hicon_to_image(hicon: int, size: int = 128) -> Image.Image | None:
    """把 HICON 画到内存 DC 上再读成 PIL 图像。

    用 ``DrawIconEx`` 而不是 ``bitblt``：它会正确处理图标的遮罩通道，边缘不会
    出现黑框。

    ``GetDIBits`` 要求完整的 ``BITMAPINFOHEADER``（40 字节），且 ``biHeight``
    传负值才能得到自上而下的像素行——传正值需要手动垂直翻转。
    """

    if not hicon:
        return None

    hdc_screen = _user32.GetDC(0)
    hdc = _gdi32.CreateCompatibleDC(hdc_screen)
    hbmp = _gdi32.CreateCompatibleBitmap(hdc_screen, size, size)
    _gdi32.SelectObject(hdc, hbmp)

    # 铺深色底：图标多为透明底，读出来会变成纯黑方块；用主题背景色更融合。
    brush = _gdi32.CreateSolidBrush(0x00161A21)
    rect = wintypes.RECT(0, 0, size, size)
    _user32.FillRect(hdc, ctypes.byref(rect), brush)
    _gdi32.DeleteObject(brush)

    _user32.DrawIconEx(hdc, 0, 0, hicon, size, size, 0, 0, 0x0003)

    buf = ctypes.create_string_buffer(size * size * 4)
    header = _BITMAPINFOHEADER()
    header.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    header.biWidth = size
    header.biHeight = -size  # 负值 = 自上而下，省掉翻转
    header.biPlanes = 1
    header.biBitCount = 32
    header.biCompression = 0  # BI_RGB

    got = _gdi32.GetDIBits(hdc, hbmp, 0, size, buf, ctypes.byref(header), 0)

    _gdi32.DeleteObject(hbmp)
    _gdi32.DeleteDC(hdc)
    _user32.ReleaseDC(0, hdc_screen)

    if not got:
        return None
    return Image.frombuffer("RGBA", (size, size), buf.raw, "raw", "BGRA", 0, 1)


def _guess_target(lnk: Path) -> Path | None:
    """从 .lnk 二进制里找出目标 exe 路径。

    ``SHGetFileInfo`` 对快捷方式返回的图标**带左下角箭头叠加**，直接贴到面板上
    很难看。而取目标 exe 的图标没有箭头。解析 .lnk 正规做法要走 IShellLink
    COM，这里改用更省事的办法：.lnk 文件里保留了目标路径明文，扫一遍即可。

    找不到就返回 None，调用方退回用 .lnk 自身的图标（带箭头，但至少是图标）。
    """

    import re

    try:
        data = lnk.read_bytes()
    except OSError:
        return None

    pattern = re.compile(r"[A-Za-z]:\\[^\x00-\x1f\"<>|*?]{2,180}?\.exe", re.I)
    for encoding in ("utf-16-le", "mbcs"):
        try:
            text = data.decode(encoding, errors="ignore")
        except (UnicodeDecodeError, LookupError):
            continue
        for match in pattern.finditer(text):
            candidate = Path(match.group(0))
            if candidate.exists():
                return candidate
    return None


def _get_hicon(path: Path) -> int:
    """调 Shell 取文件图标句柄；取不到返回 0。"""

    info = _SHFILEINFOW()
    _shell32.SHGetFileInfoW(
        str(path), 0, ctypes.byref(info), ctypes.sizeof(info),
        SHGFI_ICON | SHGFI_LARGEICON,
    )
    if info.hIcon:
        return info.hIcon
    _shell32.SHGetFileInfoW(
        str(path), 0, ctypes.byref(info), ctypes.sizeof(info),
        SHGFI_ICON | SHGFI_LARGEICON | SHGFI_USEFILEATTRIBUTES,
    )
    return info.hIcon


def extract_icon(path: Path, size: int = 160) -> Image.Image | None:
    """提取文件/快捷方式的图标并缩放到 ``size``。失败返回 ``None``。"""

    try:
        hicon = 0
        if path.suffix.lower() == ".lnk":
            target = _guess_target(path)
            if target is not None:
                hicon = _get_hicon(target)
        if not hicon:
            hicon = _get_hicon(path)
        if not hicon:
            return None

        image = _hicon_to_image(hicon, 128)
        _user32.DestroyIcon(hicon)
        if image is None:
            return None
        return image.resize((size, size), Image.LANCZOS)
    except Exception:
        return None


def round_corners(image: Image.Image, radius_ratio: float = CORNER_RATIO) -> Image.Image:
    """套上圆角遮罩，返回带透明通道的图像。"""

    size = image.size[0]
    radius = int(size * radius_ratio)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255,
    )
    result = image.convert("RGBA")
    result.putalpha(mask)
    return result


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """取一个能渲染中文的字体。

    PIL 的默认位图字体不含中文字形，直接画中文会变成空心方块。雅黑在
    Windows 上必装，取不到再退回默认字体（英文/数字仍可用）。
    """

    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def placeholder_tile(size: int, color: str, letter: str) -> Image.Image:
    """取不到图标时的替代方案：纯色圆角方块 + 首字。

    比留空好：至少能靠颜色和首字辨认，而且视觉上是完整的一张卡片。
    """

    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    radius = int(size * CORNER_RATIO)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=color,
    )

    text = (letter or "")[:1]
    if text:
        font = _load_font(int(size * 0.44))
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(
            ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1] - size * 0.03),
            text, fill=(255, 255, 255, 240), font=font,
        )
    return image


class IconCache:
    """把 PIL 图像转成 tkinter 可用对象并缓存。

    ``ImageTk.PhotoImage`` 必须被引用住，否则会被 GC 回收，界面上表现为
    图标突然变空白——这是 tkinter 最常见的坑之一。
    """

    def __init__(self, size: int) -> None:
        self.size = size
        self._cache: dict[str, ImageTk.PhotoImage] = {}

    def get(
        self, path: Path | str, accent: str, label: str | None = None,
    ) -> ImageTk.PhotoImage:
        key = f"{path}|{accent}"
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        # shell: 别名（回收站等虚拟项）没有真实文件，取不到图标，直接占位。
        image = extract_icon(path, self.size) if isinstance(path, Path) else None
        if image is not None:
            image = round_corners(image, CORNER_RATIO)
        else:
            text = label or (path.stem if isinstance(path, Path) else str(path))
            image = placeholder_tile(self.size, accent, text)

        photo = ImageTk.PhotoImage(image)
        self._cache[key] = photo
        return photo
