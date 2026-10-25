"""读取 Windows 桌面图标的真实位置与名称。

为什么要走 SysListView32
------------------------
桌面图标不是文件，而是 Explorer 托管的 ``SysListView32`` 控件里的 item。
位置由系统维护（拖动、自动排列、分辨率变化都会改），只能从该控件读。
读出来以后，手指坐标只要落进某个 item 的矩形就是命中——不需要另造一套
图标面板，用户看到的就是他自己桌面上的东西。

跨进程读取的必要性
------------------
``LVM_GETITEMRECT`` / ``LVM_GETITEMTEXT`` 的 lParam 是**指向调用方进程内存**
的指针。我们的进程和 explorer.exe 不是同一个，所以必须：
分配远程内存 -> 写入输入结构 -> SendMessage -> 读回结果。
直接传本地指针会静默失败或读到垃圾。

已知边界
--------
* 只处理有对应文件的图标（快捷方式/文件夹/文档）。"此电脑""回收站"这类
  虚拟项的 ``name`` 能读到，但没有可启动的文件，触发时会返回 False。
* 图标被拖成自定义位置时矩形仍然有效——位置本来就是从控件读的。
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

_user32 = ctypes.windll.user32
_kernel32 = ctypes.windll.kernel32

# ---- 窗口消息 ----
LVM_FIRST = 0x1000
LVM_GETITEMCOUNT = LVM_FIRST + 4
LVM_GETITEMRECT = LVM_FIRST + 14
LVM_GETITEMTEXTW = LVM_FIRST + 115

#: LVM_GETITEMRECT 的矩形类型（写进 RECT.left 作为入参）：
#: 0 = 整格（图标 + 文字），1 = 图标图案那一块，2 = 文字那一块。
LVIR_BOUNDS = 0
LVIR_ICON = 1
LVIR_LABEL = 2
#: LVITEM.mask：请求文本。
LVIF_TEXT = 0x0001

# ---- 进程权限 ----
PROCESS_VM_OPERATION = 0x0008
PROCESS_VM_READ = 0x0010
PROCESS_VM_WRITE = 0x0020
PROCESS_QUERY_INFORMATION = 0x0400

MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04

#: 每个图标名最多取这么多字符（资源管理器自身也用 260）。
MAX_TEXT = 260

#: 可启动的文件后缀，按优先级排列。
LAUNCHABLE_SUFFIXES = ("", ".lnk", ".url", ".exe", ".bat", ".cmd")


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class LVITEMW(ctypes.Structure):
    _fields_ = [
        ("mask", wintypes.UINT),
        ("iItem", ctypes.c_int),
        ("iSubItem", ctypes.c_int),
        ("state", wintypes.UINT),
        ("stateMask", wintypes.UINT),
        ("pszText", ctypes.c_void_p),
        ("cchTextMax", ctypes.c_int),
        ("iImage", ctypes.c_int),
        ("lParam", ctypes.c_void_p),
        ("iIndent", ctypes.c_int),
        ("iGroupId", ctypes.c_int),
        ("cColumns", wintypes.UINT),
        ("puColumns", ctypes.c_void_p),
        ("piColFmt", ctypes.c_void_p),
        ("iGroup", ctypes.c_int),
    ]


_user32.SendMessageW.argtypes = [
    wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
]
_user32.SendMessageW.restype = wintypes.LPARAM
_user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
_user32.FindWindowW.restype = wintypes.HWND
_user32.FindWindowExW.argtypes = [
    wintypes.HWND, wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR,
]
_user32.FindWindowExW.restype = wintypes.HWND
_user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(POINT)]
_user32.GetWindowThreadProcessId.argtypes = [
    wintypes.HWND, ctypes.POINTER(wintypes.DWORD),
]
_user32.GetWindowThreadProcessId.restype = wintypes.DWORD

_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.VirtualAllocEx.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, wintypes.DWORD,
]
_kernel32.VirtualAllocEx.restype = wintypes.LPVOID
_kernel32.VirtualFreeEx.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD,
]
_kernel32.WriteProcessMemory.argtypes = [
    wintypes.HANDLE, wintypes.LPVOID, wintypes.LPCVOID,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
]
_kernel32.ReadProcessMemory.argtypes = [
    wintypes.HANDLE, wintypes.LPCVOID, wintypes.LPVOID,
    ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t),
]


@dataclass(frozen=True, slots=True)
class DesktopIcon:
    """桌面上一个图标的屏幕矩形与名称。

    ``left/top/right/bottom`` 是**整格**（图案 + 文字），磁吸的命中判定用它；
    ``icon_top/icon_bottom`` 是**图案**那一块（``LVIR_ICON``），磁吸落点用它。
    两者差多少？2026-10-23 实测（96px 见方格子）：单行名字的格子中心比图案
    中心低 **11px**，双行名字低 **21px**——直接用格子中心就会"吸到名字上"，
    而且名字行数一变落点就跟着变，用户看到的就是"每次吸到图标不同位置"。
    """

    index: int
    name: str
    left: int
    top: int
    right: int
    bottom: int
    #: 图案区域的上下边界；读不到时回退成 -1，由 ``icon_center_y`` 兜底。
    icon_top: int = -1
    icon_bottom: int = -1

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def icon_center_y(self) -> int:
        """图案中心的 y。读不到图案矩形时退回格子中心。"""

        if self.icon_top >= 0 and self.icon_bottom > self.icon_top:
            return (self.icon_top + self.icon_bottom) // 2
        return (self.top + self.bottom) // 2

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def find_desktop_listview() -> int:
    """定位桌面图标控件。

    Windows 11 上 ``SHELLDLL_DefView`` 可能挂在 ``Progman`` 下，也可能被
    移到某个 ``WorkerW`` 下（壁纸幻灯片机制会移动它），所以两处都要找。
    """

    progman = _user32.FindWindowW("Progman", None)
    candidates: list[int] = []
    if progman:
        candidates.append(int(progman))

    worker = 0
    while True:
        worker = int(
            _user32.FindWindowExW(None, wintypes.HWND(worker), "WorkerW", None) or 0,
        )
        if not worker:
            break
        candidates.append(worker)

    for parent in candidates:
        defview = _user32.FindWindowExW(
            wintypes.HWND(parent), None, "SHELLDLL_DefView", None,
        )
        if defview:
            listview = _user32.FindWindowExW(defview, None, "SysListView32", None)
            if listview:
                return int(listview)
    return 0


class _RemoteMemory:
    """explorer 进程里的一块可读写内存，用于跨进程传结构体。"""

    def __init__(self, hwnd: int, size: int) -> None:
        self.hwnd = hwnd
        self.hproc = 0
        self.address = 0
        self.size = size

        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(pid))
        self.hproc = int(
            _kernel32.OpenProcess(
                PROCESS_VM_OPERATION | PROCESS_VM_READ | PROCESS_VM_WRITE
                | PROCESS_QUERY_INFORMATION,
                False,
                pid.value,
            )
            or 0,
        )
        if not self.hproc:
            raise OSError("无法打开 explorer 进程（权限不足）")
        self.address = int(
            _kernel32.VirtualAllocEx(
                wintypes.HANDLE(self.hproc), None, size,
                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
            )
            or 0,
        )
        if not self.address:
            _kernel32.CloseHandle(wintypes.HANDLE(self.hproc))
            raise OSError("无法在 explorer 进程里分配内存")

    def write(self, data, offset: int = 0) -> bool:
        written = ctypes.c_size_t(0)
        return bool(
            _kernel32.WriteProcessMemory(
                wintypes.HANDLE(self.hproc),
                ctypes.c_void_p(self.address + offset),
                ctypes.byref(data),
                ctypes.sizeof(data),
                ctypes.byref(written),
            ),
        )

    def read(self, data, offset: int = 0) -> bool:
        read_count = ctypes.c_size_t(0)
        return bool(
            _kernel32.ReadProcessMemory(
                wintypes.HANDLE(self.hproc),
                ctypes.c_void_p(self.address + offset),
                ctypes.byref(data),
                ctypes.sizeof(data),
                ctypes.byref(read_count),
            ),
        )

    def close(self) -> None:
        if self.address and self.hproc:
            _kernel32.VirtualFreeEx(
                wintypes.HANDLE(self.hproc), ctypes.c_void_p(self.address),
                0, MEM_RELEASE,
            )
        if self.hproc:
            _kernel32.CloseHandle(wintypes.HANDLE(self.hproc))
        self.address = 0
        self.hproc = 0

    def __enter__(self) -> _RemoteMemory:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _item_rect(
    memory: _RemoteMemory,
    listview: int,
    index: int,
    kind: int = LVIR_BOUNDS,
) -> RECT | None:
    rect = RECT(kind, 0, 0, 0)
    if not memory.write(rect):
        return None
    _user32.SendMessageW(
        wintypes.HWND(listview), LVM_GETITEMRECT, index, memory.address,
    )
    out = RECT()
    if not memory.read(out):
        return None
    return out


def _item_text(memory: _RemoteMemory, listview: int, index: int) -> str:
    text_offset = ctypes.sizeof(LVITEMW)
    item = LVITEMW()
    item.mask = LVIF_TEXT
    item.iItem = index
    item.iSubItem = 0
    item.pszText = ctypes.c_void_p(memory.address + text_offset)
    item.cchTextMax = MAX_TEXT
    if not memory.write(item):
        return ""
    # 先清空文本区，避免残留旧内容被读成名字。
    blank = ctypes.create_unicode_buffer(MAX_TEXT)
    if not memory.write(blank, text_offset):
        return ""
    _user32.SendMessageW(
        wintypes.HWND(listview), LVM_GETITEMTEXTW, index, memory.address,
    )
    buffer = ctypes.create_unicode_buffer(MAX_TEXT)
    if not memory.read(buffer, text_offset):
        return ""
    return buffer.value


def _icon_area(
    listview: int,
    memory: _RemoteMemory,
    index: int,
    cell: RECT,
) -> tuple[int, int]:
    """图标**图案**区域在屏幕坐标下的上下边界；读不到返回 ``(-1, -1)``。

    ``LVIR_ICON`` 给的是"图案 + 四周留白"那一块（实测 96px 格子里：图案区
    高 64、文字区高 22），所以它的正中心正好是图案正中。

    做两道合理性检查，宁可退回整格中心也不要乱吸：图案区必须落在整格之内，
    而且必须比整格矮——某些视图模式（列表/详细信息）不按"图案在上、
    文字在下"排布，那种布局下这个矩形没有意义。
    """

    rect = _item_rect(memory, listview, index, LVIR_ICON)
    if rect is None or rect.bottom <= rect.top:
        return -1, -1
    if rect.top < cell.top - 1 or rect.bottom > cell.bottom + 1:
        return -1, -1
    if rect.bottom - rect.top >= cell.bottom - cell.top:
        return -1, -1
    top = POINT(rect.left, rect.top)
    bottom = POINT(rect.right, rect.bottom)
    _user32.ClientToScreen(wintypes.HWND(listview), ctypes.byref(top))
    _user32.ClientToScreen(wintypes.HWND(listview), ctypes.byref(bottom))
    return top.y, bottom.y


def read_desktop_icons() -> list[DesktopIcon]:
    """读取桌面全部图标的屏幕矩形与名称。失败时返回空列表。"""

    listview = find_desktop_listview()
    if not listview:
        return []

    count = int(
        _user32.SendMessageW(wintypes.HWND(listview), LVM_GETITEMCOUNT, 0, 0),
    )
    if count <= 0:
        return []

    icons: list[DesktopIcon] = []
    try:
        with _RemoteMemory(
            listview, ctypes.sizeof(LVITEMW) + MAX_TEXT * 2 + 64,
        ) as memory:
            for index in range(count):
                rect = _item_rect(memory, listview, index)
                if rect is None:
                    continue
                # 控件坐标 -> 屏幕坐标。桌面控件通常铺满主屏，但不能假设。
                top_left = POINT(rect.left, rect.top)
                bottom_right = POINT(rect.right, rect.bottom)
                _user32.ClientToScreen(wintypes.HWND(listview), ctypes.byref(top_left))
                _user32.ClientToScreen(wintypes.HWND(listview), ctypes.byref(bottom_right))
                if bottom_right.x <= top_left.x or bottom_right.y <= top_left.y:
                    continue
                icon_top, icon_bottom = _icon_area(listview, memory, index, rect)
                icons.append(
                    DesktopIcon(
                        index=index,
                        name=_item_text(memory, listview, index),
                        left=top_left.x,
                        top=top_left.y,
                        right=bottom_right.x,
                        bottom=bottom_right.y,
                        icon_top=icon_top,
                        icon_bottom=icon_bottom,
                    ),
                )
    except OSError:
        return []
    return icons


def desktop_dirs() -> list[Path]:
    """当前用户桌面 + 公共桌面（中文系统上目录名通常仍是 Desktop）。"""

    folders: list[Path] = []
    home = Path.home()
    for name in ("Desktop", "桌面"):
        folder = home / name
        if folder.is_dir():
            folders.append(folder)
            break
    public = os.environ.get("PUBLIC")
    if public:
        for name in ("Desktop", "桌面"):
            folder = Path(public) / name
            if folder.is_dir():
                folders.append(folder)
                break
    return folders


def resolve_icon_path(name: str) -> Path | None:
    """把图标名解析成桌面上的实际路径。

    资源管理器显示的条目名通常去掉了 ``.lnk`` 后缀，所以要把常见后缀都试一遍。
    """

    if not name:
        return None
    for folder in desktop_dirs():
        direct = folder / name
        if direct.exists():
            return direct
        for suffix in LAUNCHABLE_SUFFIXES:
            if not suffix:
                continue
            candidate = folder / (name + suffix)
            if candidate.exists():
                return candidate
    return None


def launch_icon(icon: DesktopIcon) -> tuple[bool, str]:
    """启动图标对应的程序/文件夹。返回 ``(成功, 说明)``。

    优先按文件启动而不是模拟双击：双击要求桌面必须在前台，否则会点到挡在
    前面的窗口上。按文件启动不受当前前台窗口影响。
    """

    path = resolve_icon_path(icon.name)
    if path is None:
        return False, f"找不到「{icon.name}」对应的文件（可能是系统虚拟项）"
    try:
        os.startfile(str(path))  # noqa: S606
    except OSError as exc:
        return False, f"启动失败：{exc}"
    return True, f"已打开：{icon.name}"
