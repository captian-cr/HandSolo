"""虚拟桌面：快捷键序列与发送顺序（不真的按键）。"""

from __future__ import annotations

import pytest

from handmouse import virtual_desktop as vd


def test_create_desktop_sequence() -> None:
    assert vd.hotkey_sequence("create") == (vd.VK_LWIN, vd.VK_LCONTROL, ord("D"))


def test_unknown_action_raises() -> None:
    with pytest.raises(KeyError):
        vd.hotkey_sequence("nope")


def test_press_then_release_in_reverse_order() -> None:
    events: list[tuple[int, bool]] = []

    ok = vd.press_sequence((1, 2, 3), sender=lambda vk, up: events.append((vk, up)), hold_s=0)

    assert ok
    assert [vk for vk, up in events if not up] == [1, 2, 3]
    assert [vk for vk, up in events if up] == [3, 2, 1]


def test_sender_failure_returns_false() -> None:
    def boom(vk: int, up: bool) -> None:
        raise RuntimeError("no input")

    assert vd.press_sequence((1,), sender=boom, hold_s=0) is False


def test_open_new_desktop_uses_the_create_hotkey() -> None:
    seen: list[int] = []

    ok = vd.open_new_desktop(sender=lambda vk, up: seen.append(vk), settle_s=0)

    assert ok
    assert seen[:3] == [vd.VK_LWIN, vd.VK_LCONTROL, ord("D")]
