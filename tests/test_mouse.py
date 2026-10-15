import ctypes
import os

import pytest

from handmouse.config import AppConfig
from handmouse.contracts import MouseCommand, MouseCommandType
from handmouse.mouse import MouseController, NullMouseBackend, Win32MouseBackend
from handmouse.screen_map import ScreenMap


def test_controller_dispatches_without_system_injection() -> None:
    backend = NullMouseBackend()
    controller = MouseController(backend)
    controller.execute_many(
        [
            MouseCommand(MouseCommandType.MOVE, 1, x=0.25, y=0.75),
            MouseCommand(MouseCommandType.LEFT_DOWN, 2),
            MouseCommand(MouseCommandType.LEFT_UP, 3),
            MouseCommand(MouseCommandType.WHEEL, 4, delta=120),
            MouseCommand(MouseCommandType.PAGE_NEXT, 5),
            MouseCommand(MouseCommandType.PAGE_PREVIOUS, 6),
        ],
    )

    assert [name for name, _ in backend.commands] == [
        "move",
        "left_down",
        "left_up",
        "wheel",
        "page_next",
        "page_previous",
    ]


# ---------------------------------------------------------------------------
# SendInput 边界：结构体布局、绝对坐标归一化、越界钳制
# ---------------------------------------------------------------------------


class _CapturingUser32:
    """替换 ``user32``，把每次 SendInput 的 INPUT 事件原样留下。"""

    def __init__(self) -> None:
        self.events: list[object] = []

    def SendInput(self, _count: int, pointer: object, _size: int) -> int:
        # ``_send`` 传的是 ctypes.byref(event)，其 ``_obj`` 就是原始 INPUT 实例。
        event = getattr(pointer, "_obj", None)
        assert event is not None, "无法从 byref 取回原始 INPUT 对象"
        self.events.append(event)
        return 1


def _win32_backend() -> Win32MouseBackend:
    # 关掉磁吸：让 transform 走"原样返回"的短路分支，测试只关心注入 payload。
    return Win32MouseBackend(
        AppConfig(cursor_snap_enabled=False),
        ScreenMap(dpi_aware=False),
    )


@pytest.mark.skipif(os.name != "nt", reason="Win32 结构体仅在 Windows 上定义")
def test_win32_input_structs_match_documented_layout() -> None:
    from handmouse.mouse import INPUT, KEYBDINPUT, MOUSEINPUT

    pointer_size = ctypes.sizeof(ctypes.c_void_p)
    is_64 = pointer_size == 8

    # INPUT 含一个对齐到指针宽度的 union，整体尺寸随位数变化。
    assert ctypes.sizeof(INPUT) == (40 if is_64 else 28)
    assert ctypes.sizeof(MOUSEINPUT) == (32 if is_64 else 24)
    assert ctypes.sizeof(KEYBDINPUT) == (24 if is_64 else 16)

    assert INPUT.type.offset == 0
    assert INPUT.union.offset == (8 if is_64 else 4)

    # MOUSEINPUT 字段顺序/对齐必须与 Win32 定义一致，否则 SendInput 会错位。
    assert MOUSEINPUT.dx.offset == 0
    assert MOUSEINPUT.dy.offset == 4
    assert MOUSEINPUT.mouseData.offset == 8
    assert MOUSEINPUT.dwFlags.offset == 12
    assert MOUSEINPUT.time.offset == 16
    assert MOUSEINPUT.dwExtraInfo.offset == (24 if is_64 else 20)

    assert KEYBDINPUT.wVk.offset == 0
    assert KEYBDINPUT.wScan.offset == 2
    assert KEYBDINPUT.dwFlags.offset == 4
    assert KEYBDINPUT.time.offset == 8
    assert KEYBDINPUT.dwExtraInfo.offset == (16 if is_64 else 12)


@pytest.mark.skipif(os.name != "nt", reason="SendInput 仅在 Windows 上可用")
def test_move_normalizes_to_full_range_and_zeroes_unused_fields() -> None:
    backend = _win32_backend()
    spy = _CapturingUser32()
    backend._user32 = spy  # type: ignore[assignment]
    try:
        backend.move(0.5, 0.5)

        assert len(spy.events) == 1
        event = spy.events[0]
        assert event.type == Win32MouseBackend.INPUT_MOUSE  # type: ignore[attr-defined]
        mouse = event.mi  # type: ignore[attr-defined]
        # 0—65535 全域归一化：中点必须落在正中央。
        assert (mouse.dx, mouse.dy) == (32768, 32768)
        assert mouse.dwFlags & Win32MouseBackend.MOVE
        assert mouse.dwFlags & Win32MouseBackend.ABSOLUTE
        # 单屏不带 VIRTUALDESK；若带上会被系统按虚拟桌面重新解释。
        assert (mouse.dwFlags & Win32MouseBackend.VIRTUALDESK) == 0
        # time 与 dwExtraInfo 按 Win32 要求必须为 0，否则会污染系统输入队列。
        assert (mouse.mouseData, mouse.time, mouse.dwExtraInfo) == (0, 0, 0)
    finally:
        backend.shutdown()


@pytest.mark.skipif(os.name != "nt", reason="SendInput 仅在 Windows 上可用")
def test_move_clamps_out_of_range_coordinates() -> None:
    backend = _win32_backend()
    spy = _CapturingUser32()
    backend._user32 = spy  # type: ignore[assignment]
    try:
        backend.move(-5.0, 5.0)

        mouse = spy.events[0].mi  # type: ignore[attr-defined]
        assert (mouse.dx, mouse.dy) == (0, 65535)
    finally:
        backend.shutdown()
