"""``uia_targets`` 的去重与过滤逻辑测试（不依赖真实窗口）。"""

from __future__ import annotations

from handmouse.uia_targets import UITarget, _dedupe


def make(name: str, ctype: str, left: int, top: int, right: int, bottom: int) -> UITarget:
    return UITarget(
        name=name, control_type=ctype,
        left=left, top=top, right=right, bottom=bottom,
    )


def test_inner_text_is_dropped_in_favour_of_menu_item() -> None:
    """菜单项和它内部的文字只留一个——留能点的那个。"""

    targets = [
        make("编辑(E)", "MenuItemControl", 0, 0, 100, 40),
        make("编辑(", "TextControl", 10, 10, 50, 30),
    ]

    result = _dedupe(targets)

    assert [item.name for item in result] == ["编辑(E)"]


def test_inner_button_replaces_container() -> None:
    """容器里包着真按钮时，容器让位。"""

    targets = [
        make("面板", "GroupControl", 0, 0, 200, 200),
        make("确定", "ButtonControl", 50, 50, 100, 80),
    ]

    result = _dedupe(targets)

    assert [item.name for item in result] == ["确定"]


def test_disjoint_targets_both_kept() -> None:
    targets = [
        make("A", "ButtonControl", 0, 0, 40, 40),
        make("B", "ButtonControl", 100, 0, 140, 40),
    ]

    result = _dedupe(targets)

    assert len(result) == 2


def test_nested_text_kept_when_no_interactive_parent() -> None:
    """两个都不是交互类型时，保留内层（更具体）。"""

    targets = [
        make("整块", "GroupControl", 0, 0, 300, 300),
        make("一行字", "TextControl", 20, 20, 200, 40),
    ]

    result = _dedupe(targets)

    assert [item.name for item in result] == ["一行字"]


def test_center_and_area() -> None:
    target = make("X", "ButtonControl", 10, 20, 110, 70)

    assert target.center == (60, 45)
    assert target.area == 5000


def test_contains() -> None:
    outer = make("outer", "GroupControl", 0, 0, 100, 100)
    inner = make("inner", "ButtonControl", 10, 10, 50, 50)

    assert outer.contains(inner) is True
    assert inner.contains(outer) is False


def test_plain_text_is_never_a_snap_target() -> None:
    """纯文字不能当磁吸目标（2026-10-22 用户反馈「停留在字上也被吸住」）。

    这是策略常量级的锁：真正的过滤发生在枚举 UIA 控件的循环里，需要真实
    窗口才能跑；但"哪些类型永远排除"是这个模块的**契约**，改动了就该被拦。
    """

    from handmouse.uia_targets import INTERACTIVE_TYPES, SKIP_TYPES

    assert "TextControl" in SKIP_TYPES, "纯文字又被当成可点目标了"
    assert "TextControl" not in INTERACTIVE_TYPES


def test_container_types_are_not_preferred_over_inner_controls() -> None:
    """容器类型不能进"可交互优先"，否则会用一个整块容器盖掉里面的小按钮。"""

    from handmouse.uia_targets import INTERACTIVE_TYPES

    for container in ("GroupControl", "PaneControl", "WindowControl"):
        assert container not in INTERACTIVE_TYPES, f"{container} 是容器，不该被优先保留"
