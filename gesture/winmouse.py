"""Windows 鼠标 / DPI 封装。

直接用 ctypes 调 user32，不引入 pyautogui——少一个依赖，
而且 SetCursorPos 是"瞬移"式定位，没有 pyautogui 的缓动延迟。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

_user32 = ctypes.windll.user32

SM_CXSCREEN = 0
SM_CYSCREEN = 1

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

_user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
_user32.SetCursorPos.restype = wintypes.BOOL
_user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
_user32.GetCursorPos.restype = wintypes.BOOL
_user32.mouse_event.argtypes = [
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_size_t,
]
_user32.mouse_event.restype = None


def enable_dpi_awareness() -> None:
    """开启 DPI 感知，否则 125% / 150% 缩放下 GetSystemMetrics 返回的是逻辑像素，
    屏幕坐标会整体压缩，导致光标够不到右下角。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR_AWARE_V2
    except Exception:
        try:
            _user32.SetProcessDPIAware()
        except Exception:
            pass


def dpi_scale() -> float:
    """返回系统缩放倍数，用来把调试窗口放大回原来的视觉大小。"""
    try:
        dpi = int(_user32.GetDpiForSystem())
    except Exception:
        dpi = 96
    return max(1.0, dpi / 96.0)


def screen_size() -> tuple[int, int]:
    return int(_user32.GetSystemMetrics(SM_CXSCREEN)), int(_user32.GetSystemMetrics(SM_CYSCREEN))


def move_to(x: float, y: float) -> None:
    _user32.SetCursorPos(int(round(x)), int(round(y)))


def position() -> tuple[int, int]:
    pt = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return int(pt.x), int(pt.y)


def left_down() -> None:
    _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)


def left_up() -> None:
    _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)


def right_down() -> None:
    _user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)


def right_up() -> None:
    _user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
