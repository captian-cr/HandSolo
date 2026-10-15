"""Windows 原生输入注入与安全释放。

鼠标与键盘都走 ``SendInput``：它比 ``mouse_event``/``SetCursorPos`` 更可靠
（能和真实输入一起进入队列，且不会覆盖用户自己的按键状态），也比
PyAutoGUI 少一个依赖、少一层缓动延迟。

绝对坐标的换算全部交给 ``ScreenMap``，本模块不读 ``GetSystemMetrics``，
避免在多屏机器上把坐标写死成主屏。
"""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from ctypes import wintypes
from typing import Protocol

from handmouse.config import AppConfig
from handmouse.contracts import MouseCommand, MouseCommandType
from handmouse.cursor_snap import CursorSnapService, WindowsUiAutomationProbe
from handmouse.screen_map import ScreenMap


LOGGER = logging.getLogger(__name__)


class MouseBackend(Protocol):
    """便于测试替换的鼠标后端接口。"""

    def move(self, x: float, y: float) -> None: ...

    def left_down(self) -> None: ...

    def left_up(self) -> None: ...

    def right_click(self) -> None: ...

    def left_click(self) -> None: ...

    def wheel(self, delta: int) -> None: ...

    def page_next(self) -> None: ...

    def page_previous(self) -> None: ...

    def release_all(self) -> None: ...

    def update_config(self, config: AppConfig) -> None: ...

    def shutdown(self) -> None: ...


if os.name == "nt":
    ULONG_PTR = wintypes.WPARAM

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        )

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ULONG_PTR),
        )

    class INPUT_UNION(ctypes.Union):
        _fields_ = (("mi", MOUSEINPUT), ("ki", KEYBDINPUT))

    class INPUT(ctypes.Structure):
        _anonymous_ = ("union",)
        _fields_ = (("type", wintypes.DWORD), ("union", INPUT_UNION))


class Win32MouseBackend:
    """通过 ``SendInput`` 向主显示器注入鼠标事件。"""

    INPUT_MOUSE = 0
    INPUT_KEYBOARD = 1
    MOVE = 0x0001
    LEFTDOWN = 0x0002
    LEFTUP = 0x0004
    RIGHTDOWN = 0x0008
    RIGHTUP = 0x0010
    WHEEL = 0x0800
    ABSOLUTE = 0x8000
    # 绝对坐标默认被钳在主显示器内；要覆盖整个虚拟桌面必须带上这个标志，
    # 且 0—65535 必须按虚拟桌面宽高（而非主屏宽高）归一化。
    VIRTUALDESK = 0x4000
    KEYUP = 0x0002
    VK_PRIOR = 0x21
    VK_NEXT = 0x22

    def __init__(
        self,
        config: AppConfig | None = None,
        screen_map: ScreenMap | None = None,
    ) -> None:
        if os.name != "nt":
            raise RuntimeError("手势鼠标的鼠标注入仅支持 Windows")
        config = config or AppConfig()
        self.screen_map = screen_map or ScreenMap(multi_monitor=False)
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (
            wintypes.UINT,
            ctypes.POINTER(INPUT),
            ctypes.c_int,
        )
        self._user32.SendInput.restype = wintypes.UINT
        self._left_is_down = False
        self._snap_suspended = False
        self._lock = threading.Lock()
        width, height = self.screen_map.size
        origin_x, origin_y = self.screen_map.origin
        self._snapper = CursorSnapService(
            WindowsUiAutomationProbe(),
            screen_width=width,
            screen_height=height,
            origin_x=origin_x,
            origin_y=origin_y,
            # 屏幕度量签名：DPI 模式/显示器数量/原点尺寸一变，磁吸缓存立刻作废。
            screen_signature=screen_map.describe(),
            enabled=config.cursor_snap_enabled,
            radius_px=config.cursor_snap_radius_px,
        )

    def _send(
        self,
        flags: int,
        dx: int = 0,
        dy: int = 0,
        data: int = 0,
    ) -> None:
        event = INPUT(
            type=self.INPUT_MOUSE,
            mi=MOUSEINPUT(
                dx=dx,
                dy=dy,
                mouseData=data,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            ),
        )
        sent = self._user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
        if sent != 1:
            code = ctypes.get_last_error()
            detail = (
                f"SendInput失败，Win32错误码={code}。"
                "若目标程序以管理员身份运行，请以同等权限启动Vimoose。"
            )
            raise OSError(code, detail)

    def _send_key(self, virtual_key: int, flags: int = 0) -> None:
        event = INPUT(
            type=self.INPUT_KEYBOARD,
            ki=KEYBDINPUT(
                wVk=virtual_key,
                wScan=0,
                dwFlags=flags,
                time=0,
                dwExtraInfo=0,
            ),
        )
        sent = self._user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
        if sent != 1:
            code = ctypes.get_last_error()
            raise OSError(code, f"键盘翻页注入失败，Win32错误码={code}。")

    def _press_key(self, virtual_key: int) -> None:
        with self._lock:
            self._send_key(virtual_key)
            self._send_key(virtual_key, self.KEYUP)

    def move(self, x: float, y: float) -> None:
        if not self._snap_suspended:
            x, y = self._snapper.transform(x, y)
        # 归一化以 ScreenMap 的注入坐标系为基准换算成 0—65535 绝对坐标；
        # 多屏时补上 VIRTUALDESK，否则会被系统钳回主屏。
        normalized_x, normalized_y = self.screen_map.absolute_coordinates(x, y)
        flags = self.MOVE | self.ABSOLUTE
        if self.screen_map.use_virtual_desk:
            flags |= self.VIRTUALDESK
        with self._lock:
            self._send(flags, normalized_x, normalized_y)

    def left_down(self) -> None:
        with self._lock:
            if not self._left_is_down:
                self._send(self.LEFTDOWN)
                self._left_is_down = True
                self._snap_suspended = True
                self._snapper.clear()

    def left_up(self) -> None:
        with self._lock:
            self._send(self.LEFTUP)
            self._left_is_down = False
            self._snap_suspended = False

    def left_click(self) -> None:
        with self._lock:
            self._send(self.LEFTDOWN)
            self._send(self.LEFTUP)
            self._left_is_down = False

    def right_click(self) -> None:
        with self._lock:
            self._send(self.RIGHTDOWN)
            self._send(self.RIGHTUP)

    def wheel(self, delta: int) -> None:
        with self._lock:
            self._send(self.WHEEL, data=delta)

    def page_next(self) -> None:
        self._press_key(self.VK_NEXT)

    def page_previous(self) -> None:
        self._press_key(self.VK_PRIOR)

    def release_all(self) -> None:
        # 即使内部状态未知也发送释放，确保异常退出不会留下按键。
        with self._lock:
            try:
                self._send(self.LEFTUP)
                self._send(self.RIGHTUP)
            finally:
                self._left_is_down = False
                self._snap_suspended = False

    @property
    def snap_target_name(self) -> str | None:
        if self._snap_suspended:
            return None
        return self._snapper.target_name

    def update_config(self, config: AppConfig) -> None:
        self._snapper.configure(
            enabled=config.cursor_snap_enabled,
            radius_px=config.cursor_snap_radius_px,
        )

    def shutdown(self) -> None:
        self.release_all()
        self._snapper.shutdown()


class NullMouseBackend:
    """不注入系统输入的安全后端，用于 ``--safe-mode``、测试和非Windows环境。

    记录收到的指令而不调用任何 Win32 API，因此 safe-mode 下可以完整跑通
    状态机、确认动作序列，同时保证系统光标不被接管。
    """

    def __init__(self, screen_map: ScreenMap | None = None) -> None:
        self.screen_map = screen_map or ScreenMap(dpi_aware=False)
        self.commands: list[tuple[str, tuple[object, ...]]] = []

    def _record(self, name: str, *args: object) -> None:
        self.commands.append((name, args))

    def move(self, x: float, y: float) -> None:
        self._record("move", x, y)

    def left_down(self) -> None:
        self._record("left_down")

    def left_up(self) -> None:
        self._record("left_up")

    def left_click(self) -> None:
        self._record("left_click")

    def right_click(self) -> None:
        self._record("right_click")

    def wheel(self, delta: int) -> None:
        self._record("wheel", delta)

    def page_next(self) -> None:
        self._record("page_next")

    def page_previous(self) -> None:
        self._record("page_previous")

    def release_all(self) -> None:
        self._record("release_all")

    @property
    def snap_target_name(self) -> str | None:
        return None

    def update_config(self, config: AppConfig) -> None:
        pass

    def shutdown(self) -> None:
        self.release_all()


class MouseController:
    """执行标准指令并统一记录系统注入错误。"""

    def __init__(self, backend: MouseBackend) -> None:
        self.backend = backend
        self.last_error: str | None = None

    def execute(self, command: MouseCommand) -> None:
        try:
            if command.kind == MouseCommandType.MOVE:
                if command.x is not None and command.y is not None:
                    self.backend.move(command.x, command.y)
            elif command.kind == MouseCommandType.LEFT_CLICK:
                self.backend.left_click()
            elif command.kind == MouseCommandType.LEFT_DOWN:
                self.backend.left_down()
            elif command.kind == MouseCommandType.LEFT_UP:
                self.backend.left_up()
            elif command.kind == MouseCommandType.RIGHT_CLICK:
                self.backend.right_click()
            elif command.kind == MouseCommandType.WHEEL:
                self.backend.wheel(command.delta)
            elif command.kind == MouseCommandType.PAGE_NEXT:
                self.backend.page_next()
            elif command.kind == MouseCommandType.PAGE_PREVIOUS:
                self.backend.page_previous()
            # PAUSE是状态通知，不产生系统输入。
            self.last_error = None
        except OSError as error:
            self.last_error = str(error)
            LOGGER.exception("鼠标指令执行失败：%s", command.kind.value)
            self.backend.release_all()

    def execute_many(self, commands: list[MouseCommand]) -> None:
        for command in commands:
            self.execute(command)

    def release_all(self) -> None:
        try:
            self.backend.release_all()
        except OSError:
            LOGGER.exception("释放鼠标按键失败")

    @property
    def snap_target_name(self) -> str | None:
        return getattr(self.backend, "snap_target_name", None)

    def update_config(self, config: AppConfig) -> None:
        self.backend.update_config(config)

    def shutdown(self) -> None:
        try:
            self.backend.shutdown()
        except OSError:
            LOGGER.exception("关闭鼠标后端失败")
