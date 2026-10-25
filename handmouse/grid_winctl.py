"""对当前前台窗口执行最小化 / 最大化 / 关闭，并提供"某坐标处是哪个窗口"。

用途
----
系统的标题栏按钮只有 20 来像素，隔空指着点不到。这里接管"关窗"这类操作，
让用户用同一套大图标点。``window_at`` 则用于诊断"双击为什么没打到目标"。

安全约定
--------
* **关闭走 ``WM_CLOSE`` 而不是 ``DestroyWindow``**：让目标程序自己走正常退出
  流程（可以弹"是否保存"），不会被无声杀掉。
* 取前台窗口时会跳过自己的叠加层、无标题窗口（桌面/任务栏）和不可见窗口，
  避免把命令发到奇怪的地方。
* 所有操作先 ``IsWindow`` 校验，句柄失效直接返回 False，不抛异常。
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

_user32 = ctypes.windll.user32

#: 关闭窗口的窗口消息。
WM_CLOSE = 0x0010

#: ShowWindow 命令码。
SW_MAXIMIZE = 3
SW_MINIMIZE = 6
SW_RESTORE = 9

#: GetWindow 的取窗方向：Z 序下一个窗口。
GW_HWNDNEXT = 2


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


_user32.GetForegroundWindow.restype = wintypes.HWND
_user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
_user32.GetWindow.restype = wintypes.HWND
_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.IsZoomed.argtypes = [wintypes.HWND]
_user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
_user32.PostMessageW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
]
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
_user32.WindowFromPoint.argtypes = [_POINT]
_user32.WindowFromPoint.restype = wintypes.HWND


def window_title(hwnd: int) -> str:
    """取窗口标题；无效句柄返回空串。"""

    if not hwnd:
        return ""
    length = _user32.GetWindowTextLengthW(wintypes.HWND(hwnd))
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(wintypes.HWND(hwnd), buffer, length + 1)
    return buffer.value


def foreground_window(exclude: int = 0, max_skip: int = 12) -> int:
    """取应当被操作的那个窗口。

    从真实前台窗口开始沿 Z 序向下找，跳过 ``exclude``（自己的叠加层）、
    无标题窗口（桌面、任务栏、托盘）与不可见窗口。
    """

    hwnd = _user32.GetForegroundWindow()
    found = int(hwnd or 0)
    skipped = 0
    while found and skipped < max_skip:
        if (
            found != exclude
            and _user32.IsWindowVisible(wintypes.HWND(found))
            and window_title(found)
        ):
            return found
        nxt = _user32.GetWindow(wintypes.HWND(found), GW_HWNDNEXT)
        found = int(nxt or 0)
        skipped += 1
    return 0


def window_at(px: int, py: int, exclude: int = 0, max_skip: int = 8) -> int:
    """返回屏幕上 ``(px, py)`` 处可见的那个窗口（跳过自己的叠加层）。

    叠加层是 topmost + 点击穿透，但 ``WindowFromPoint`` 并不看穿透明层——它
    照样返回叠加层自己。所以要沿 Z 序往下找第一个真正的窗口，否则诊断出来
    的永远是"手势指针"这个窗口标题。
    """

    point = _POINT(int(px), int(py))
    hwnd = int(_user32.WindowFromPoint(point) or 0)
    skipped = 0
    while hwnd and skipped < max_skip:
        if hwnd != exclude and _user32.IsWindowVisible(wintypes.HWND(hwnd)):
            return hwnd
        hwnd = int(_user32.GetWindow(wintypes.HWND(hwnd), GW_HWNDNEXT) or 0)
        skipped += 1
    return 0


def minimize(hwnd: int) -> bool:
    if not hwnd or not _user32.IsWindow(wintypes.HWND(hwnd)):
        return False
    _user32.ShowWindow(wintypes.HWND(hwnd), SW_MINIMIZE)
    return True


def toggle_maximize(hwnd: int) -> bool:
    """最大化与还原之间切换。"""

    if not hwnd or not _user32.IsWindow(wintypes.HWND(hwnd)):
        return False
    if _user32.IsZoomed(wintypes.HWND(hwnd)):
        _user32.ShowWindow(wintypes.HWND(hwnd), SW_RESTORE)
    else:
        _user32.ShowWindow(wintypes.HWND(hwnd), SW_MAXIMIZE)
    return True


def close(hwnd: int) -> bool:
    """发 WM_CLOSE，让目标程序走自己的关闭流程。"""

    if not hwnd or not _user32.IsWindow(wintypes.HWND(hwnd)):
        return False
    return bool(_user32.PostMessageW(wintypes.HWND(hwnd), WM_CLOSE, 0, 0))


# ---- 窗口几何：读矩形 / 移动缩放（供"底部悬浮球"用）------------------------


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


_user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
_user32.SetWindowPos.argtypes = [
    wintypes.HWND, wintypes.HWND,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    wintypes.UINT,
]
_user32.IsZoomed.argtypes = [wintypes.HWND]
_user32.GetSystemMetrics.argtypes = [ctypes.c_int]
_user32.GetSystemMetrics.restype = ctypes.c_int

SWP_NOZORDER = 0x0004
SWP_NOACTIVATE = 0x0010

SM_CXSCREEN = 0
SM_CYSCREEN = 1

#: 窗口允许被缩到的最小尺寸（像素）。再小就没有意义，也容易让程序自己崩。
MIN_WINDOW_W = 320
MIN_WINDOW_H = 240


def window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """返回 ``(left, top, right, bottom)`` 屏幕像素；无效句柄返回 ``None``。"""

    if not hwnd or not _user32.IsWindow(wintypes.HWND(hwnd)):
        return None
    rect = _RECT()
    if not _user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
        return None
    return (int(rect.left), int(rect.top), int(rect.right), int(rect.bottom))


def screen_size() -> tuple[int, int]:
    """主屏尺寸（像素）。"""

    return (int(_user32.GetSystemMetrics(SM_CXSCREEN)), int(_user32.GetSystemMetrics(SM_CYSCREEN)))


def move_resize(hwnd: int, rect: tuple[int, int, int, int]) -> bool:
    """把窗口摆到 ``rect``。不改 Z 序、不抢焦点（免得把用户正在操作的窗口踢到后台）。"""

    if not hwnd or not _user32.IsWindow(wintypes.HWND(hwnd)):
        return False
    left, top, right, bottom = rect
    width = max(MIN_WINDOW_W, int(right - left))
    height = max(MIN_WINDOW_H, int(bottom - top))
    return bool(
        _user32.SetWindowPos(
            wintypes.HWND(hwnd),
            wintypes.HWND(0),
            int(left),
            int(top),
            width,
            height,
            SWP_NOZORDER | SWP_NOACTIVATE,
        )
    )


def is_maximized(hwnd: int) -> bool:
    """窗口是否处于最大化状态。"""

    if not hwnd or not _user32.IsWindow(wintypes.HWND(hwnd)):
        return False
    return bool(_user32.IsZoomed(wintypes.HWND(hwnd)))
