"""把主题面板放到「新建的虚拟桌面」上。

为什么用快捷键而不是 API
------------------------
``IVirtualDesktopManager`` 只有**查询**接口（窗口在不在当前桌面）；
真正"新建桌面 / 把窗口搬过去"走的是未公开的
``IVirtualDesktopManagerInternal``，其接口在每个 Windows 版本都会变
（Win10 21H2 / 22H2 / Win11 21H2 / 23H2 的 IID 与虚表都不同）。
为了不把项目绑死在某个系统版本上，这里用系统**自带快捷键** ``Win+Ctrl+D``
新建并切过去，然后就在那个桌面上启动本进程——窗口天然归属新桌面，
原来的 Windows 桌面完全不受影响。

所以「保留原版桌面 + 我们的主题桌面」是这样成立的：
新桌面上跑主题面板，``Win+Ctrl+←/→`` 随时切回去，两个桌面互不可见。

键序做成纯函数 ``hotkey_sequence``，无 Windows 也能单测。
"""

from __future__ import annotations

import time

VK_LWIN = 0x5B
VK_LCONTROL = 0xA2
VK_LMENU = 0xA4
KEYEVENTF_KEYUP = 0x0002

#: 动作 → 按键组合（顺序即按下顺序）。
SEQUENCES: dict[str, tuple[int, ...]] = {
    "create": (VK_LWIN, VK_LCONTROL, ord("D")),
    "left": (VK_LWIN, VK_LCONTROL, 0x25),   # VK_LEFT
    "right": (VK_LWIN, VK_LCONTROL, 0x27),  # VK_RIGHT
    "close": (VK_LWIN, VK_LCONTROL, 0x73),  # F4
}


def hotkey_sequence(action: str) -> tuple[int, ...]:
    """返回该动作要按下的虚拟键序列；未知动作抛 ``KeyError``。"""

    return SEQUENCES[action]


def press_sequence(keys: tuple[int, ...], sender=None, hold_s: float = 0.04) -> bool:
    """依次按下再逆序抬起。``sender`` 可注入，便于测试时断言调用。"""

    if sender is None:
        try:
            import ctypes

            user32 = ctypes.windll.user32

            def sender(vk: int, up: bool) -> None:  # type: ignore[misc]
                user32.keybd_event(vk, 0, KEYEVENTF_KEYUP if up else 0, 0)
        except Exception:
            return False

    try:
        for vk in keys:
            sender(vk, False)
            time.sleep(hold_s)
        for vk in reversed(keys):
            sender(vk, True)
            time.sleep(hold_s / 2)
    except Exception:
        return False
    return True


def open_new_desktop(sender=None, settle_s: float = 0.8) -> bool:
    """新建并切到一个空白虚拟桌面。之后启动的窗口就落在那个桌面上。"""

    if not press_sequence(hotkey_sequence("create"), sender=sender):
        return False
    if settle_s > 0:
        time.sleep(settle_s)
    return True
