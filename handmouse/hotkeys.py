"""无需额外依赖的Windows全局快捷键轮询。"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections.abc import Callable


class HotkeyMonitor(threading.Thread):
    """轮询 Ctrl+Alt 组合键，通过回调把意图交给主循环。"""

    VK_CONTROL = 0x11
    VK_MENU = 0x12
    VK_ESCAPE = 0x1B
    VK_G = 0x47
    VK_P = 0x50
    VK_Q = 0x51

    def __init__(
        self,
        on_toggle: Callable[[], None],
        on_emergency: Callable[[], None],
        on_quit: Callable[[], None],
        on_preview: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(name="global-hotkeys", daemon=True)
        self.on_toggle = on_toggle
        self.on_emergency = on_emergency
        self.on_quit = on_quit
        #: Ctrl+Alt+P：开关预览窗。**必须是全局键**——预览窗关掉之后窗口就没了，
        #: 窗口内的 P 键跟着一起失效，只有全局键能把画面叫回来。
        self.on_preview = on_preview
        self.stop_event = threading.Event()
        self._pressed: set[int] = set()

    def stop(self) -> None:
        self.stop_event.set()
        if self.is_alive():
            self.join(timeout=1.0)

    def run(self) -> None:
        if os.name != "nt":
            return
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        while not self.stop_event.is_set():
            control = bool(user32.GetAsyncKeyState(self.VK_CONTROL) & 0x8000)
            alt = bool(user32.GetAsyncKeyState(self.VK_MENU) & 0x8000)
            hotkeys = [
                (self.VK_G, self.on_toggle),
                (self.VK_ESCAPE, self.on_emergency),
                (self.VK_Q, self.on_quit),
            ]
            if self.on_preview is not None:
                hotkeys.append((self.VK_P, self.on_preview))
            for key, callback in hotkeys:
                down = control and alt and bool(user32.GetAsyncKeyState(key) & 0x8000)
                if down and key not in self._pressed:
                    self._pressed.add(key)
                    callback()
                elif not down:
                    self._pressed.discard(key)
            time.sleep(0.03)

