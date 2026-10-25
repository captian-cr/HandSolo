"""``ClickGate`` 双击门控测试。

背景：隔空捏合没有触觉反馈，抬手时误触代价大，所以执行类动作要求两次捏合。
"""

from __future__ import annotations

from handmouse.grid_judge import ClickGate


def test_first_pinch_only_arms() -> None:
    gate = ClickGate(window_ms=600.0)

    assert gate.register(0, 1000.0) is None
    assert gate.armed is True
    assert gate.pending_target == 0


def test_second_pinch_within_window_fires() -> None:
    gate = ClickGate(window_ms=600.0)

    gate.register(0, 1000.0)
    assert gate.register(0, 1400.0) == "double"
    assert gate.armed is False


def test_single_falls_back_after_window() -> None:
    gate = ClickGate(window_ms=600.0)

    gate.register(0, 1000.0)
    assert gate.poll(1300.0) is None
    assert gate.poll(1700.0) == 0
    # 取出后必须复位，避免重复上报。
    assert gate.poll(1800.0) is None
    assert gate.armed is False


def test_different_target_restarts_count() -> None:
    gate = ClickGate(window_ms=600.0)

    gate.register(0, 1000.0)
    # 指到另一个图标再捏，不能算作对第一个图标的双击。
    assert gate.register(1, 1200.0) is None
    assert gate.pending_target == 1
    assert gate.register(1, 1500.0) == "double"


def test_late_second_pinch_is_not_double() -> None:
    gate = ClickGate(window_ms=600.0)

    gate.register(0, 1000.0)
    assert gate.register(0, 1800.0) is None
    assert gate.armed is True


def test_reset_clears_state() -> None:
    gate = ClickGate(window_ms=600.0)

    gate.register(0, 1000.0)
    gate.reset()

    assert gate.armed is False
    assert gate.pending_target is None
    assert gate.poll(5000.0) is None
