"""在 OpenCV 画面上写中文（Pillow + 系统字体），以及 HUD 面板绘制。"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np

_FONT_CANDIDATES = (
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
)

# 颜色一律用 RGB 传参，内部转成 BGR 写入画面
DEFAULT_WHITE = (255, 255, 255)
DEFAULT_GRAY = (200, 200, 200)
DEFAULT_GREEN = (0, 255, 0)
DEFAULT_YELLOW = (0, 215, 255)
DEFAULT_RED = (255, 60, 60)


def _font_path() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


@lru_cache(maxsize=16)
def _font(size: int):
    from PIL import ImageFont

    path = _font_path()
    if path is None:
        return None
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return None


def estimate_width(text: str, size: int) -> float:
    """粗略估算文本宽度：中日韩字符按 1 个字宽，其余按 0.58 字宽。"""
    total = 0.0
    for ch in text:
        total += size if ord(ch) > 0x2E80 else size * 0.58
    return total


def draw_texts(frame_bgr: np.ndarray, items) -> None:
    """原地绘制多段文本。

    items: 可迭代的 (text, (x, y), size, (r, g, b))，坐标是文本左上角。
    只做一次 BGR<->RGB 转换，比逐条转换快得多。
    字体缺失时退化为 cv2.putText（此时中文会变成问号，但不会崩）。
    """
    items = list(items)
    if not items:
        return

    fonts = {}
    for _, _, size, _ in items:
        fonts.setdefault(size, _font(size))

    if any(font is None for font in fonts.values()):
        for text, (x, y), size, color in items:
            cv2.putText(
                frame_bgr,
                text,
                (int(x), int(y + size)),
                cv2.FONT_HERSHEY_SIMPLEX,
                size / 30.0,
                (color[2], color[1], color[0]),
                1,
                cv2.LINE_AA,
            )
        return

    from PIL import Image, ImageDraw

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    for text, (x, y), size, color in items:
        draw.text((x, y), text, font=fonts[size], fill=(color[0], color[1], color[2]))
    frame_bgr[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def draw_panel(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    alpha: float = 0.55,
    color: tuple = (20, 20, 20),
) -> None:
    """半透明底板，原地修改。color 是 BGR。"""
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_bgr.shape[1], int(x + width))
    y1 = min(frame_bgr.shape[0], int(y + height))
    if x1 <= x0 or y1 <= y0:
        return
    region = frame_bgr[y0:y1, x0:x1]
    overlay = np.empty_like(region)
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, region, 1.0 - alpha, 0.0, region)


def draw_hud(frame_bgr: np.ndarray, rows, origin=(12, 12), size=18, line_gap=24, padding=10) -> None:
    """rows: [(text, (r, g, b)), ...]，自动按最长一行撑开底板。"""
    rows = list(rows)
    if not rows:
        return
    x0, y0 = origin
    width = max(estimate_width(text, size) for text, _ in rows) + padding * 2
    height = line_gap * len(rows) + padding * 2 - (line_gap - size)

    draw_panel(frame_bgr, x0, y0, width, height)

    items = []
    for i, (text, color) in enumerate(rows):
        items.append((text, (x0 + padding, y0 + padding + i * line_gap), size, color))
    draw_texts(frame_bgr, items)
