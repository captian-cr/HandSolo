"""LAUNCH / 热键派发测试。

白名单是硬安全边界：轨迹识别错一次的代价不能是"打开任意 EXE"。
safe-mode（``dry_run=True``）下必须零副作用。
"""

from __future__ import annotations

import pytest

from handmouse.config import AppConfig
from handmouse.contracts import ActionCommand, ActionType
from handmouse.launch import LaunchDispatcher, LaunchRejected
from handmouse.mouse import parse_hotkey, resolve_virtual_key


def _dispatcher(**overrides) -> LaunchDispatcher:
    values: dict[str, object] = {"launch_enabled": True}
    values.update(overrides)
    config = AppConfig(**values)  # type: ignore[arg-type]
    config.validate()
    return LaunchDispatcher(config, dry_run=True)


def test_alias_resolves_through_whitelist() -> None:
    dispatcher = _dispatcher()

    assert dispatcher.resolve_launch_target("notepad") == "notepad.exe"
    assert dispatcher.resolve_launch_target("NOTEPAD") == "notepad.exe"


def test_full_command_inside_whitelist_is_allowed() -> None:
    assert _dispatcher().resolve_launch_target("calc.exe") == "calc.exe"


def test_target_outside_whitelist_raises() -> None:
    with pytest.raises(LaunchRejected):
        _dispatcher().resolve_launch_target("cmd.exe")


def test_dry_run_launch_records_without_side_effect() -> None:
    dispatcher = _dispatcher()

    result = dispatcher.dispatch(
        ActionCommand(ActionType.LAUNCH, 1, name="W", target="notepad"),
    )

    assert result.ok
    assert "dry-run" in result.message
    assert dispatcher.history == [result.message]


def test_dispatch_rejects_target_outside_whitelist() -> None:
    dispatcher = _dispatcher()

    result = dispatcher.dispatch(
        ActionCommand(ActionType.LAUNCH, 1, name="X", target="cmd.exe"),
    )

    assert not result.ok
    assert "launch_whitelist" in result.message
    assert dispatcher.history == [], "被拒绝的动作不能留下执行记录"


def test_launch_disabled_blocks_every_launch() -> None:
    dispatcher = _dispatcher(launch_enabled=False)

    result = dispatcher.dispatch(
        ActionCommand(ActionType.LAUNCH, 1, name="W", target="notepad"),
    )

    assert not result.ok
    assert "关闭" in result.message


def test_hotkey_dry_run_does_not_inject_keys() -> None:
    dispatcher = _dispatcher()

    result = dispatcher.dispatch(
        ActionCommand(ActionType.HOTKEY, 1, name="X", target="ctrl+w"),
    )

    assert result.ok
    assert "dry-run" in result.message


def test_cancel_is_always_accepted() -> None:
    assert _dispatcher().dispatch(ActionCommand(ActionType.CANCEL, 1)).ok


def test_unknown_action_kind_is_rejected() -> None:
    result = _dispatcher().dispatch(ActionCommand("bogus", 1))  # type: ignore[arg-type]

    assert not result.ok


def test_parse_hotkey_normalizes_names_and_case() -> None:
    assert parse_hotkey("ctrl+shift+w") == [0x11, 0x10, 0x57]
    assert parse_hotkey("Win+D") == [0x5B, 0x44]
    assert parse_hotkey("alt+tab") == [0x12, 0x09]


@pytest.mark.parametrize("combo", ["", "+", "ctrl+???", "nope", "f13"])
def test_bad_hotkeys_are_rejected(combo: str) -> None:
    with pytest.raises(ValueError):
        parse_hotkey(combo)


def test_resolve_virtual_key_supports_function_keys() -> None:
    assert resolve_virtual_key("f5") == 0x74
    assert resolve_virtual_key("escape") == 0x1B
