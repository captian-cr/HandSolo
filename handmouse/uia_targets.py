"""枚举当前窗口里"能点的地方"，不依赖控件类型分类。

设计思路
--------
用户的原话是「把他们都当成按钮不就好了吗」——这是对的。UIA 的
``ControlType`` 分类靠不住：Electron 应用里"编辑/窗口/帮助"这些菜单项
虽然标成了 ``MenuItemControl``，但侧边栏条目可能是 ``TabItem``、
``Button``，甚至只是一个带文字的 ``Group``。所以判定标准改成三条：

1. **有名字**（没有文字的容器基本不是给人点的）
2. **有尺寸**（面积过小的是装饰线或图标碎片）
3. **不与其他候选重复**（父子嵌套时保留更具体的那个）

只有少数明确是"可交互"的类型享有优先级：当父子矩形重叠时，保留可交互的
那一个，把包着它的容器丢掉——否则"编辑(E)"和它里面的"编辑("会各占一个框。

为什么点击不用 UIA 的 InvokePattern
-----------------------------------
Invoke 需要保留 COM 控件引用，跨线程使用会失效，且 ``uiautomation`` 的
控件对象绑定在创建它的线程上。这里改成只保留**纯坐标数据**，点击时用
``SetCursorPos`` + ``mouse_event`` 打到矩形中心——跨线程安全，也不怕元素
在两次刷新之间重建。
"""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass

_user32 = ctypes.windll.user32

#: 明确可交互的控件类型。这些在父子重叠时优先保留。
INTERACTIVE_TYPES = {
    "ButtonControl",
    "MenuItemControl",
    "TabItemControl",
    "HyperlinkControl",
    "ListItemControl",
    "CheckBoxControl",
    "RadioButtonControl",
    "EditControl",
    "ComboBoxControl",
    "SplitButtonControl",
    "TreeItemControl",
    "SliderControl",
    #: 2026-10-22 用户要求"能操作的一切都吸"：很多应用（尤其 Electron / 网页套壳）
    #: 把可点区域标成 ImageControl，原来被当装饰忽略了。
    #: ⚠️ **不要把 GroupControl 加进来**：它是容器，一旦进了"可交互优先"，
    #: 父子去重时会用一个整块 Group 盖掉里面真实的小按钮，落点变成容器中心
    #: ——正是"整块容器不是可点条目"这条注释要避免的事。
    "ImageControl",
}

#: 永不当作目标的类型：窗口/文档根节点、纯装饰、**以及纯文字**。
#:
#: 🔴 2026-10-22 用户反馈：「停留在一些字上面也会被吸住」——
#: 纯 ``TextControl`` 点了没有任何作用，把它当磁吸目标纯属噪声，还会
#: 把光标从旁边的真按钮上拽走。真正可点的按钮/菜单项都有各自的
#: ControlType（Button/MenuItem/...），不靠 Text 兜底。
SKIP_TYPES = {
    "WindowControl",
    "DocumentControl",
    "TitleBarControl",
    "ScrollBarControl",
    "PaneControl",
    "MenuBarControl",
    "TabControl",
    "ToolBarControl",
    "TextControl",
}

#: 目标最小边长（像素）。低于这个值基本按不中，留着只会干扰。
MIN_SIZE = 14

#: 单个目标最大面积占比。超过说明是整块容器，不是可点条目。
#: 2026-10-22 由 0.20 提到 0.35：他抱怨"放大缩小这类控件很难瞄准"，而那类
#: 按钮/滑动区往往占窗口面积的两三成，旧阈值把它们当"整块容器"排除了。
MAX_AREA_RATIO = 0.35

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


@dataclass(frozen=True, slots=True)
class UITarget:
    name: str
    control_type: str
    left: int
    top: int
    right: int
    bottom: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.left + self.right) // 2, (self.top + self.bottom) // 2)

    @property
    def area(self) -> int:
        return max(0, self.right - self.left) * max(0, self.bottom - self.top)

    def contains(self, other: UITarget) -> bool:
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )


def _screen_size() -> tuple[int, int]:
    return int(_user32.GetSystemMetrics(0)), int(_user32.GetSystemMetrics(1))


def collect_targets(hwnd: int, max_depth: int = 14) -> list[UITarget]:
    """读取指定窗口内的所有可点目标。失败时返回空列表。"""

    import uiautomation as auto

    auto.SetGlobalSearchTimeout(3.0)
    root = auto.ControlFromHandle(wintypes.HWND(hwnd))
    if root is None:
        return []

    screen_w, screen_h = _screen_size()
    max_area = int(screen_w * screen_h * MAX_AREA_RATIO)

    raw: list[UITarget] = []
    for control, depth in auto.WalkControl(root, maxDepth=max_depth):
        if depth == 0:
            continue
        try:
            control_type = control.ControlTypeName
            if control_type in SKIP_TYPES:
                continue
            name = (control.Name or "").strip()
            if not name:
                continue
            rect = control.BoundingRectangle
        except Exception:
            continue

        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width < MIN_SIZE or height < MIN_SIZE:
            continue
        # 完全在屏幕外的元素（最小化的子窗口等）不要。
        if rect.right <= 0 or rect.bottom <= 0 or rect.left >= screen_w or rect.top >= screen_h:
            continue

        target = UITarget(
            name=name,
            control_type=control_type,
            left=rect.left,
            top=rect.top,
            right=rect.right,
            bottom=rect.bottom,
        )
        if target.area > max_area:
            continue
        raw.append(target)

    return _dedupe(raw)


def _dedupe(targets: list[UITarget]) -> list[UITarget]:
    """去掉父子重复，保留"更该被点"的那一个。

    三条规则：
    * 外层是容器、内层是交互控件 -> 留内层
    * 两者交互性相同 -> 留内层（更具体，面积更小）
    * 外层是交互控件、内层只是它的一段文字 -> 留外层
    """

    kept: list[UITarget] = []
    for candidate in targets:
        candidate_interactive = candidate.control_type in INTERACTIVE_TYPES
        dominated = False
        replaced: list[UITarget] = []

        for existing in list(kept):
            existing_interactive = existing.control_type in INTERACTIVE_TYPES
            if existing.contains(candidate):
                if existing_interactive and not candidate_interactive:
                    dominated = True
                    break
                replaced.append(existing)
                continue
            if candidate.contains(existing):
                if existing_interactive and not candidate_interactive:
                    dominated = True
                    break
                replaced.append(existing)

        if dominated:
            continue
        for item in replaced:
            if item in kept:
                kept.remove(item)
        kept.append(candidate)
    return kept


def list_targets(hwnd: int | None = None, max_depth: int = 14) -> list[UITarget]:
    """枚举前台窗口（或指定窗口）的可点目标。"""

    if hwnd is None:
        hwnd = int(_user32.GetForegroundWindow() or 0)
    if not hwnd:
        return []
    return collect_targets(hwnd, max_depth=max_depth)


def click_at(x: int, y: int, double: bool = False) -> None:
    """把光标移到 ``(x, y)`` 并左键单击/双击。"""

    _user32.SetCursorPos(int(x), int(y))
    time.sleep(0.03)
    repeats = 2 if double else 1
    for index in range(repeats):
        _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        if index == 0 and repeats == 2:
            time.sleep(0.04)


def click_target(target: UITarget) -> None:
    x, y = target.center
    click_at(x, y, double=False)
