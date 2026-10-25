"""Windows控件磁吸的异步探测与纯几何状态机。

热路径 ``transform()`` 只做「加锁 + 读不可变快照 + 整数换算」，是常数时间；
真正跨进程的 UI Automation 查询只发生在 ``cursor-snap-uia`` 守护线程里，
并且由该线程自己完成 COM 初始化——COM 对象不跨线程传递。缺少
``uiautomation`` 或初始化失败时自动降级为纯坐标点击，点击照常可用。
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
import os
import threading
import time
from dataclasses import dataclass
from math import hypot, inf
from typing import Callable, Protocol

from handmouse.filters import clamp


LOGGER = logging.getLogger(__name__)

# 缓存候选的最长存活时间。worker 若卡死或退出，快照必须在 1 秒内自然失效，
# 否则光标会一直被吸在一个早已不存在的控件上。
#: 吸附目标横跳的判定参数：1 秒内吸住这么多个不同目标，就临时松手，
#: 避免光标在两个挨得很近的目标之间被来回拽（用户反馈"很容易被别的软件吸过去"）。
_SWITCH_WINDOW_SECONDS = 1.0
_SWITCH_MAX = 3
_SUPPRESS_SECONDS = 0.6

CANDIDATE_TTL_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class SnapCandidate:
    """一个可点击控件及其推荐落点。

    坐标均为物理屏幕像素：``x``/``y`` 是推荐点击点，``left``/``top``/
    ``right``/``bottom`` 是控件矩形。``target_id`` 是 UI Automation 的
    RuntimeId（控件身份），``expires_at`` 是 ``time.monotonic()`` 时间戳。
    对象不可变，所以热路径可以直接读，无需担心被 worker 改到一半。
    """

    x: int
    y: int
    left: int
    top: int
    right: int
    bottom: int
    name: str = ""
    control_type: str = ""
    target_id: str = ""
    expires_at: float = inf
    #: 显式吸附落点。为 ``None`` 时按 ``cursor_snap_to_center`` 决定用矩形中心
    #: 还是 UIA 点击点；桌面图标会填这里——它的"矩形"是**图标+文字的格子**，
    #: 格子中心落在文字上而不是图案上（2026-10-23 实测：单行名字偏下 11px、
    #: 双行名字偏下 21px，这正是"每次吸到图标不同位置"的来源）。
    snap_x: int | None = None
    snap_y: int | None = None


@dataclass(frozen=True, slots=True)
class DesktopContext:
    """探测时刻的桌面上下文，任一项变化都让缓存候选立即失效。

    - ``foreground`` / ``title``：前台窗口句柄与标题（切窗口、切标签页）
    - ``screen``：屏幕度量签名（DPI 模式、显示器数量、注入坐标系原点与尺寸），
      由 ``ScreenMap.describe()`` 提供
    """

    foreground: int = 0
    title: str = ""
    screen: str = ""


def _win32_foreground() -> tuple[int, str]:
    """读取前台窗口句柄与标题；非 Windows 或调用失败时返回空值。"""

    if os.name != "nt":
        return 0, ""
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        handle = int(user32.GetForegroundWindow())
        if not handle:
            return 0, ""
        length = int(user32.GetWindowTextLengthW(handle))
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(handle, buffer, length + 1)
        return handle, buffer.value
    except Exception:
        LOGGER.debug("读取前台窗口失败", exc_info=True)
        return 0, ""


def _distance_to_rect(x: int, y: int, candidate: SnapCandidate) -> float:
    nearest_x = min(max(x, candidate.left), candidate.right)
    nearest_y = min(max(y, candidate.top), candidate.bottom)
    return hypot(x - nearest_x, y - nearest_y)


def _inside(x: int, y: int, candidate: SnapCandidate) -> bool:
    """点在不在控件矩形内（含边界）。用于"进框才吸"的判据。"""

    return (
        candidate.left <= x <= candidate.right
        and candidate.top <= y <= candidate.bottom
    )


#: 自绘 UI 兜底时接受的容器类型：自定义绘制 / 分组 / 面板。
#: 这些类型本身并非"可点控件"，但不少自绘按钮就只暴露成这样。
_FALLBACK_TYPES = {
    50025,  # Custom
    50026,  # Group
    50033,  # Pane
}

#: 纯图标控件：播放器控制条的「−」「全屏」这类按钮往往没有 Button 节点，
#: 只有一层 ``ImageControl``（图标本身）盖在按钮上。
#:
#: 2026-10-20 用户反馈"缩小键没有磁吸，同一条控制条上滑块和全屏都有"。
#: 原因不是半径太小，而是 ``_candidate_from_control`` 沿父链找 Button 时，
#: 图标节点的父链上如果没有 Button（自绘控件条很常见），就会一路走到覆盖整条
#: 控制条的大容器，被 ``_looks_like_button`` 的尺寸上限否掉 → 这个图标按钮
#: 整个消失；而旁边的滑块自身是 Slider/Button 节点，所以照常能吸。
#:
#: 修法：图标尺寸小、形状规整时，**它自己就是可吸落点**，不必等父链给答案。
_ICON_TYPES = {
    50006,  # Image
}


def _looks_like_icon(control: object) -> bool:
    """纯图标控件是否像"一个按钮上的图标"。

    比 ``_looks_like_button`` 更严：图标必须够小（<=48px 见方），否则会把
    整块画布/背景当成图标吸过去。
    """

    try:
        rect = control.BoundingRectangle
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
    except Exception:
        return False
    if width < 6 or height < 6:
        return False
    if width > 48 or height > 48:
        return False
    return True


def _looks_like_button(control: object) -> bool:
    """矩形像不像一个按钮：尺寸适中、长宽比不极端。

    判据刻意保守（宽 12~220、高 12~120、长宽比 0.3~8）。宁可漏掉一个自绘
    按钮，也不要把整块窗口面板吸过来——那会让光标往窗口中间乱跳。
    """

    try:
        rect = control.BoundingRectangle
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
    except Exception:
        return False
    if width < 12 or height < 12:
        return False
    if width > 220 or height > 120:
        return False
    ratio = width / height
    return 0.3 <= ratio <= 8.0


#: 标题栏按钮的标准几何（Win10/11）：每个 46px 宽，从右边界往左排。
_TITLEBAR_BUTTON_WIDTH = 46
_TITLEBAR_HEIGHT = 46
#: 标题栏按钮的滞回带宽（像素）：光标已经从"最大化"挪到"关闭"上、但仍在
#: 旧按钮外扩这么多像素以内时，**继续用旧按钮**，不跟着最近距离跳。
#: 2026-10-22 用户反馈「夹在放大和叉叉的中间反跳」就是没有这一层。
_TITLEBAR_STICKY_PX = 9


def _inside_expanded(x: int, y: int, candidate: SnapCandidate, margin: int) -> bool:
    """点在不在矩形外扩 ``margin`` 之后的范围内。"""

    return (
        candidate.left - margin <= x <= candidate.right + margin
        and candidate.top - margin <= y <= candidate.bottom + margin
    )


def _best_candidate(
    x: int,
    y: int,
    nearby: list[SnapCandidate],
) -> SnapCandidate:
    """在若干候选里选一个吸附目标。

    排序键（依次比较，越小越优先）：

    1. **被别的候选包住** → 优先。光标落在整行 TabItem 里、旁边正好是它的
       "关闭"按钮时，要选那个按钮，而不是把整行吸到中心。
    2. **到矩形边界的距离** → 近者优先（矩形内一律算 0）。
    3. **到吸附落点（正中心）的距离** → 见下。
    4. 面积 → 小者优先（同分时的兜底）。

    🔴 第 3 维是 2026-10-22 为他加的（原话：「吸在这些小标的左右上下，
    会离另外一个小标很近，然后有时候就会被其他小标吸走，就是要吸在这些
    小标的中央才行」）。桌面图标的"图标 + 文字"矩形又宽又高，相邻两个
    常常**互相重叠**——此时第 2 维两边都是 0，若直接落到面积裁决，光标
    明明在 A 上也会被 B 抢走。加一层"谁的中心更近"就正好对应"吸在中央"。

    单独抽成函数是为了能直接单测排序规则，不必伪造整个 UIA 树。
    """

    def rank(item: SnapCandidate) -> tuple[int, float, float, int]:
        nested_in_other = any(
            other is not item and _contains(other, item) for other in nearby
        )
        snap_point_distance = hypot(item.x - x, item.y - y)
        area = max(0, item.right - item.left) * max(0, item.bottom - item.top)
        return (
            0 if nested_in_other else 1,
            _activation_distance(x, y, item),
            snap_point_distance,
            area,
        )

    return min(nearby, key=rank)


def _titlebar_fallback(x: int, y: int, radius_px: int) -> SnapCandidate | None:
    """UIA 完全读不到标题栏按钮时，按 Windows 标准位置推算三个按钮。

    为什么需要这一层：Electron 应用（豆包、部分桌面版客户端）用 CSS 画标题栏，
    那 — □ × 在 UIA 里**完全没有节点**。2026-10-20 实测豆包：整条标题栏从右往左
    扫 200px，每一点都只返回一个 1453×1043 的 Pane（覆盖整个窗口内容区），
    整条父链找不到任何按钮。这类只能按标准位置猜。

    只在 UIA 一个候选都没找到时才启用，所以有真正系统标题栏的应用不受影响。

    🔴 上面这句是 2026-10-20 的旧说法，**已过期**：现在 ``find_target`` 里它经
    ``_titlebar_pick`` **优先**采用（同一天修的"缩小键吸不了"）。它之所以
    仍然安全，是因为下面用 ``WindowFromPoint`` + ``GA_ROOT`` 拿的是**光标正下方**
    那个窗口，而不是 ``GetForegroundWindow``——光标在桌面图标上时，算出来的
    按钮必然落在那个图标所在的窗口里，不会把光标甩到别的窗口去。
    """

    if os.name != "nt":
        return None
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        # 必须用**光标所在**的窗口，不能用 GetForegroundWindow：
        # 光标在桌面上时前台窗口可能是别的程序，那样就会按那个程序的标题栏
        # 位置去猜，把光标吸到毫不相干的地方（用户反馈"很容易被别的软件
        # 吸过去"）。WindowFromPoint 拿到的才是光标正下方的窗口。
        user32.WindowFromPoint.restype = ctypes.c_void_p
        user32.WindowFromPoint.argtypes = [_POINT]
        hwnd = int(user32.WindowFromPoint(_POINT(int(x), int(y))) or 0)
        if not hwnd:
            return None
        # 命中的可能是子窗口（如 Chrome 的渲染宿主窗口），要上溯到顶层窗口
        # 才能拿到含标题栏的矩形。
        user32.GetAncestor.restype = ctypes.c_void_p
        user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        root = int(user32.GetAncestor(hwnd, 2) or 0)  # GA_ROOT
        if root:
            hwnd = root
        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        # 光标必须在标题栏高度带内
        if y < rect.top or y > rect.top + _TITLEBAR_HEIGHT:
            return None
        # 光标必须在右侧按钮区（留给「关闭/最大化/最小化」三个）
        if x < rect.right - _TITLEBAR_BUTTON_WIDTH * 4:
            return None

        best: SnapCandidate | None = None
        best_distance = float("inf")
        for name, index in (("关闭", 0), ("最大化", 1), ("最小化", 2)):
            right = rect.right - index * _TITLEBAR_BUTTON_WIDTH
            left = right - _TITLEBAR_BUTTON_WIDTH
            top = rect.top
            bottom = rect.top + _TITLEBAR_HEIGHT
            centre_x = (left + right) // 2
            centre_y = (top + bottom) // 2
            probe = SnapCandidate(
                x=centre_x,
                y=centre_y,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
            distance = _distance_to_rect(x, y, probe)
            if distance <= radius_px and distance < best_distance:
                best_distance = distance
                best = SnapCandidate(
                    x=centre_x,
                    y=centre_y,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    name=name,
                    control_type="TitleBarButton",
                    target_id=f"titlebar:{name}",
                )
        return best
    except Exception:
        return None


def _same_rect(left: SnapCandidate, right: SnapCandidate) -> bool:
    return (
        left.left == right.left
        and left.top == right.top
        and left.right == right.right
        and left.bottom == right.bottom
    )


def _contains(outer: SnapCandidate, inner: SnapCandidate) -> bool:
    """``inner`` 的矩形是否被 ``outer`` 完全包住（用于"内部小按钮优先"）。"""

    return (
        outer.left <= inner.left
        and outer.top <= inner.top
        and outer.right >= inner.right
        and outer.bottom >= inner.bottom
    )


def _nested(inner: SnapCandidate, outer: SnapCandidate) -> bool:
    return not _same_rect(inner, outer) and _contains(outer, inner)


def _activation_distance(x: int, y: int, candidate: SnapCandidate) -> float:
    """返回磁吸激活距离 = 光标到控件矩形的距离。

    TabItem 也一律按矩形算。WorkBuddy 侧边栏那种 318×40 的整行条目，若按
    中心锚点算（旧实现），两端各有一百多像素永远够不到，用户反馈"移不过
    去"就是这个原因。

    整条可吸之后，"标签页里的关闭按钮点不到"由两层保证，而不是靠缩小平
    吸附区域：
    * ``WindowsUiAutomationProbe.find_target`` 让**被其他候选包住的小控件
      优先**；
    * ``MagneticSnapState.resolve`` 在锁存期间遇到更贴身的嵌套控件就切换。
    """

    return _distance_to_rect(x, y, candidate)


class UiTargetProbe(Protocol):
    """控件探测后端；Windows实现必须在其专用线程内初始化COM。"""

    def initialize(self) -> None: ...

    def close(self) -> None: ...

    def find_target(
        self,
        x: int,
        y: int,
        radius_px: int,
    ) -> SnapCandidate | None: ...


class MagneticSnapState:
    """带释放滞回的磁吸状态，避免光标在控件边缘反复跳动。"""

    def __init__(
        self,
        radius_px: int,
        release_multiplier: float = 1.8,
        inside_only: bool = False,
        to_center: bool = False,
    ) -> None:
        self.radius_px = max(1, int(radius_px))
        self.release_multiplier = max(1.05, float(release_multiplier))
        #: 只在光标位于控件矩形内部时才吸附（进框才吸）。
        self.inside_only = bool(inside_only)
        #: 吸附落点取控件正中心，而不是 UIA 的 clickable point。
        self.to_center = bool(to_center)
        self._latched: SnapCandidate | None = None

    @property
    def target_name(self) -> str | None:
        if self._latched is None:
            return None
        return self._latched.name or self._latched.control_type or "可点击控件"

    @property
    def latched_rect(self) -> tuple[int, int, int, int] | None:
        """当前锁存目标的屏幕矩形，供全屏高亮层画光环用。"""

        if self._latched is None:
            return None
        candidate = self._latched
        return (candidate.left, candidate.top, candidate.right, candidate.bottom)

    def configure(self, radius_px: int) -> None:
        self.radius_px = max(1, int(radius_px))
        self._latched = None

    def clear(self) -> None:
        self._latched = None

    def resolve(
        self,
        x: int,
        y: int,
        candidate: SnapCandidate | None,
    ) -> tuple[int, int, bool]:
        if self._latched is not None:
            release_radius = self.radius_px * self.release_multiplier
            latched_distance = _activation_distance(x, y, self._latched)
            if latched_distance <= release_radius:
                # 锁存期间若光标已进入另一个更贴身的控件（典型场景：整行
                # 宽条目的 TabItem 里还嵌着一个更小的按钮），必须切换过去，
                # 否则大人质会把人一直按在中心，里面那个小按钮永远点不到。
                if candidate is not None and _nested(candidate, self._latched):
                    self._latched = candidate
                    return self._snap_point(candidate)
                return self._snap_point(self._latched)
            self._latched = None

        if candidate is not None and self._activates(x, y, candidate):
            self._latched = candidate
            return self._snap_point(candidate)
        return x, y, False

    def _snap_point(self, candidate: SnapCandidate) -> tuple[int, int, bool]:
        """吸附落点：优先用候选自带的落点，否则 ``to_center`` 取矩形中心。

        候选自带落点只有一种情况——桌面图标要落在**图标图案**中心，而不是
        "图标+文字"格子的中心（格子中心在文字上，看着就是"没吸到中央"）。
        """

        if candidate.snap_x is not None and candidate.snap_y is not None:
            return int(candidate.snap_x), int(candidate.snap_y), True
        if self.to_center:
            return (
                int(round((candidate.left + candidate.right) / 2)),
                int(round((candidate.top + candidate.bottom) / 2)),
                True,
            )
        return candidate.x, candidate.y, True

    def _activates(self, x: int, y: int, candidate: SnapCandidate) -> bool:
        """这一刻是否该吸上这个控件。"""

        if _inside(x, y, candidate):
            # 已经在控件内部：无论如何都吸（这就是"进框才吸"的正向情形）。
            return True
        if self.inside_only:
            # 进框才吸：框外一律不拽。
            return False
        return _activation_distance(x, y, candidate) <= self.radius_px


class WindowsUiAutomationProbe:
    """使用Windows UI Automation查找光标附近的可点击控件。"""

    #: 可作为磁吸落点的 UIA 控件类型。
    #:
    #: 2026-10-19 实测补齐：原先只有 Button/CheckBox/Hyperlink/MenuItem/
    #: RadioButton/SplitButton 六种，导致用户点不到的正是最常见的几种——
    #:   * WorkBuddy 左侧栏条目是 **TabItemControl**（"新建任务""资料库"…），
    #:     而 `_activation_distance` 里明明写了 TabItem 专门分支，白名单却没这个
    #:     类型，那段逻辑永远走不到；
    #:   * 资源管理器/设置页里的行是 ListItemControl / TreeItemControl / DataItem。
    #: 另外标题栏的 — □ × 实测是 ButtonControl（有精确矩形），本来就在范围内。
    _CONTROL_TYPES = {
        50000,  # Button
        50002,  # CheckBox
        50005,  # Hyperlink
        50007,  # ListItem
        50011,  # MenuItem
        50013,  # RadioButton
        50019,  # TabItem
        50024,  # TreeItem
        50029,  # DataItem
        50031,  # SplitButton
    }

    def __init__(self) -> None:
        self._auto: object | None = None
        self._initializer: object | None = None
        self._unavailable = False
        #: 上一次选中的标题栏按钮，用于边界滞回（见 `_titlebar_pick`）。
        self._titlebar_last: SnapCandidate | None = None

    @property
    def unavailable(self) -> bool:
        """True 表示此后端**永久**不可用，调用方应降级为纯坐标点击。"""

        return self._unavailable

    def initialize(self) -> None:
        # import 与 COM 初始化都在调用本方法的线程里完成：COM 对象不能跨线程
        # 使用，因此 initialize/find_target/close 必须始终由同一个线程调用。
        if self._unavailable or self._auto is not None:
            return
        try:
            from uiautomation import uiautomation as auto
        except Exception:
            self._unavailable = True
            LOGGER.info("未安装 uiautomation，控件磁吸降级为纯坐标点击")
            return
        try:
            self._initializer = auto.UIAutomationInitializerInThread()
        except Exception:
            self._unavailable = True
            LOGGER.info("UI Automation 线程初始化失败，控件磁吸降级为纯坐标点击", exc_info=True)
            return
        self._auto = auto

    def close(self) -> None:
        initializer = self._initializer
        self._initializer = None
        self._auto = None
        if initializer is not None:
            try:
                initializer.Uninitialize()
            except Exception:
                LOGGER.debug("释放UI Automation线程失败", exc_info=True)

    @staticmethod
    def _probe_offsets(radius_px: int) -> tuple[tuple[int, int], ...]:
        half = max(4, radius_px // 2)
        diagonal = max(4, round(radius_px * 0.70))
        return (
            (0, 0),
            (half, 0),
            (-half, 0),
            (0, half),
            (0, -half),
            (radius_px, 0),
            (-radius_px, 0),
            (0, radius_px),
            (0, -radius_px),
            (diagonal, diagonal),
            (diagonal, -diagonal),
            (-diagonal, diagonal),
            (-diagonal, -diagonal),
        )

    def _candidate_from_control(
        self,
        control: object,
    ) -> SnapCandidate | None:
        # 沿着父链找最近的"可点控件"。找不到真正的 Button 时，用一个尺寸像
        # 按钮的自绘容器兜底——很多工具（国产小工具、Qt、Electron 自定义
        # 标题栏）的按钮在 UIA 里根本没有 Button 节点，整条父链只有
        # Custom/Pane，旧实现到这里就返回 None，表现为"这个软件吸不到"。
        matched: object | None = None
        fallback: object | None = None
        icon: object | None = None
        cursor = control
        for _ in range(6):
            if cursor is None:
                break
            try:
                control_type = int(cursor.ControlType)
            except Exception:
                break
            if control_type in self._CONTROL_TYPES:
                matched = cursor
                break
            if (
                fallback is None
                and control_type in _FALLBACK_TYPES
                and _looks_like_button(cursor)
            ):
                fallback = cursor
            if (
                icon is None
                and control_type in _ICON_TYPES
                and _looks_like_icon(cursor)
            ):
                icon = cursor
            try:
                cursor = cursor.GetParentControl()
            except Exception:
                break
        # 优先级：真正的可点控件 > 尺寸像按钮的自绘容器 > 纯图标。
        # 图标排最后是有意的——若父链上已有 Button，走Button 的矩形更准
        # （按钮的点击区通常比图标大）；只有父链完全没有可点控件时，
        # 才退到图标本身，避免「−」这类按钮整个消失。
        control = matched if matched is not None else fallback
        if control is None:
            control = icon
        if control is None:
            return None

        try:
            if not bool(control.IsEnabled) or bool(control.IsOffscreen):
                return None
            rect = control.BoundingRectangle
            left = int(rect.left)
            top = int(rect.top)
            right = int(rect.right)
            bottom = int(rect.bottom)
            if right <= left or bottom <= top:
                return None
            x, y, got_clickable = control.GetClickablePoint()
            if not got_clickable:
                x = round((left + right) / 2)
                y = round((top + bottom) / 2)
            try:
                name = str(control.Name or "")
            except Exception:
                name = ""
            try:
                type_name = str(control.ControlTypeName or "")
            except Exception:
                type_name = "ButtonControl"
            try:
                runtime_id = control.GetRuntimeId()
                target_id = "-".join(str(part) for part in runtime_id)
            except Exception:
                target_id = ""
            return SnapCandidate(
                x=int(x),
                y=int(y),
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                name=name,
                control_type=type_name,
                target_id=target_id,
                expires_at=time.monotonic() + CANDIDATE_TTL_SECONDS,
            )
        except Exception:
            return None

    def find_target(
        self,
        x: int,
        y: int,
        radius_px: int,
    ) -> SnapCandidate | None:
        self.initialize()
        auto = self._auto
        if auto is None:
            return None

        candidates: dict[tuple[int, int, int, int], SnapCandidate] = {}
        for dx, dy in self._probe_offsets(radius_px):
            try:
                control = auto.ControlFromPoint(int(x + dx), int(y + dy))
                candidate = self._candidate_from_control(control)
            except Exception:
                # UI元素可能在探测期间被销毁；按Windows文档要求优雅降级。
                continue
            if candidate is None:
                continue
            key = (
                candidate.left,
                candidate.top,
                candidate.right,
                candidate.bottom,
            )
            candidates[key] = candidate

        nearby = [
            candidate
            for candidate in candidates.values()
            if _activation_distance(x, y, candidate) <= radius_px
        ]

        # 🔴 标题栏三个按钮**优先**，不能只在"一个候选都没有"时才兜底。
        # 2026-10-22 用户反馈「有些软件的缩小键吸不了」：那些应用（Chrome、
        # 各种套壳客户端）的标题栏按钮 UIA 里确实有节点，但同一条标题栏上
        # 还有别的候选（菜单、标签），`nearby` 非空，于是永远走不到兜底那一步，
        # 最右那三个按钮就吸不到。它们的位置是 Windows 标准几何，可靠度高于
        # 附近的容器/文字，所以判定命中就直接采用。
        titlebar = self._titlebar_pick(x, y, radius_px)
        if titlebar is not None:
            return titlebar
        # 离开标题栏按钮带就清掉滞回记忆，免得下次进来自动沿用旧按钮。
        self._titlebar_last = None

        if not nearby:
            return None

        return _best_candidate(x, y, nearby)

    def _titlebar_pick(
        self,
        x: int,
        y: int,
        radius_px: int,
    ) -> SnapCandidate | None:
        """带**滞回**的标题栏按钮选择。

        2026-10-22 用户反馈「在文件夹这里会夹在放大和叉叉的中间反跳」：
        三个按钮是 46px 宽的紧邻矩形，光标在边界上来回抖就会在两个按钮之间
        反复切换，光环和目标名一起跳。所以只要光标还留在**上一个按钮的
        粘滞带**（外扩 ``_TITLEBAR_STICKY_PX``）里，就继续用上一个，
        不跟着最近距离走。
        """

        candidate = _titlebar_fallback(x, y, radius_px)
        previous = self._titlebar_last
        if candidate is not None and previous is not None:
            if candidate.target_id != previous.target_id and _inside_expanded(
                x, y, previous, _TITLEBAR_STICKY_PX,
            ):
                return previous
        if candidate is not None:
            self._titlebar_last = candidate
        return candidate


#: 桌面图标所在的窗口类名：图标在 SysListView32 里，父窗口是 Progman/WorkerW。
_DESKTOP_CLASSES = {"SysListView32", "Progman", "WorkerW"}


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _desktop_at(x: int, y: int) -> bool:
    """判断屏幕点 ``(x, y)`` 上露出来的窗口是不是桌面本身。

    只有桌面真露出来（没被别的窗口盖住）才允许把桌面图标当候选，否则
    窗口背后那些图标会被吸，光标当场跑到看不见的地方。
    """

    if os.name != "nt":
        return False
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.WindowFromPoint.restype = ctypes.c_void_p
        user32.WindowFromPoint.argtypes = [_POINT]
        handle = user32.WindowFromPoint(_POINT(int(x), int(y)))
        if not handle:
            return False
        user32.GetClassNameW.restype = ctypes.c_int
        user32.GetClassNameW.argtypes = [
            ctypes.c_void_p,
            ctypes.c_wchar_p,
            ctypes.c_int,
        ]
        buffer = ctypes.create_unicode_buffer(64)
        user32.GetClassNameW(handle, buffer, 64)
        return buffer.value in _DESKTOP_CLASSES
    except Exception:
        return False


class DesktopIconProbe:
    """把桌面图标当作磁吸候选源。

    为什么绕开 UIA：桌面图标是 ``SysListView32`` 的列表项，2026-10-19 实测
    在图标中心点做 ``ControlFromPoint`` 返回的却是任务栏的"搜索"按钮——
    坐标体系根本对不上，拿不到稳定矩形。而 ``read_desktop_icons()`` 跨进程
    读列表项矩形是可靠的，所以直接拿它的坐标当候选。

    图标位置不常变，按秒缓存；只有光标真的落在桌面上时才返回候选。
    """

    def __init__(self, cache_seconds: float = 1.0) -> None:
        self.cache_seconds = max(0.2, float(cache_seconds))
        self._icons: list[SnapCandidate] = []
        self._loaded_at = 0.0

    @property
    def icon_count(self) -> int:
        return len(self._icons)

    def _refresh(self) -> None:
        now = time.monotonic()
        if self._icons and now - self._loaded_at < self.cache_seconds:
            return
        self._loaded_at = now
        try:
            from handmouse.grid_desktop import read_desktop_icons
        except Exception:
            self._icons = []
            return
        try:
            icons = read_desktop_icons()
        except Exception:
            LOGGER.debug("读取桌面图标失败", exc_info=True)
            icons = []
        self._icons = [
            SnapCandidate(
                x=(icon.left + icon.right) // 2,
                y=(icon.top + icon.bottom) // 2,
                left=icon.left,
                top=icon.top,
                right=icon.right,
                bottom=icon.bottom,
                name=icon.name,
                control_type="DesktopIcon",
                target_id=f"desktop:{icon.name}",
                # 落点用**图标图案**中心：格子中心在名字上（单行偏下 11px、
                # 双行偏下 21px），这就是"每次吸到图标不同位置"的根因。
                snap_x=(icon.left + icon.right) // 2,
                snap_y=icon.icon_center_y,
            )
            for icon in icons
        ]

    def initialize(self) -> None:
        """与 UiTargetProbe 协议对齐；本候选源不需要初始化。"""

    def close(self) -> None:
        self._icons = []

    def find_target(self, x: int, y: int, radius_px: int) -> SnapCandidate | None:
        if not _desktop_at(x, y):
            return None
        self._refresh()
        best: SnapCandidate | None = None
        best_distance = float("inf")
        for candidate in self._icons:
            distance = _distance_to_rect(x, y, candidate)
            if distance <= radius_px and distance < best_distance:
                best_distance = distance
                best = candidate
        return best


class CursorSnapService:
    """异步探测控件并在鼠标线程内以常数时间应用磁吸。"""

    def __init__(
        self,
        probe: UiTargetProbe,
        *,
        screen_width: int,
        screen_height: int,
        enabled: bool,
        radius_px: int,
        origin_x: int = 0,
        origin_y: int = 0,
        screen_signature: str = "",
        foreground_provider: Callable[[], tuple[int, str]] | None = None,
        start_worker: bool = True,
        poll_interval_seconds: float = 0.04,
        fallback_probe: UiTargetProbe | None = None,
        max_speed_px_s: float = 1500.0,
        inside_only: bool = False,
        to_center: bool = False,
        release_grace_seconds: float = 0.35,
    ) -> None:
        #: 速度门：光标比这更快时不做吸附。
        #: 用户 2026-10-19 反馈"磁吸力度太高，一直吸这个吸那个"——快速掠过
        #: 时若照样吸附，光标会被一串目标来回拽。慢下来精调才该吸。
        self.max_speed_px_s = max(1.0, float(max_speed_px_s))
        self._last_sample_x = 0
        self._last_sample_y = 0
        self._last_sample_at = 0.0
        self._speed_px_s = 0.0
        # 吸附目标频繁横跳时临时松手，避免在相邻目标之间来回拽。
        self._switch_times: list[float] = []
        self._last_target_rect: tuple[int, int, int, int] | None = None
        self._suppress_until = 0.0
        self.probe = probe
        #: 主探测源找不到时的备用源（当前用于桌面图标：它们不是 UIA 可点控件）。
        self.fallback_probe = fallback_probe
        # 像素坐标系必须与 ScreenMap 完全一致：``screen_*`` 是注入坐标系尺寸，
        # ``origin_*`` 是该坐标系原点（多屏时为虚拟桌面左上角，可能为负值）。
        # UIA 返回的 BoundingRectangle 就是这套物理像素，所以这里不能默认 0,0。
        self.screen_width = max(2, int(screen_width))
        self.screen_height = max(2, int(screen_height))
        self.origin_x = int(origin_x)
        self.origin_y = int(origin_y)
        # 屏幕度量签名（DPI 模式/显示器数量/原点尺寸）。一旦变化就丢弃缓存候选，
        # 因为之前算出的像素点已经不在同一套坐标系里了。
        self.screen_signature = str(screen_signature)
        self.enabled = bool(enabled)
        self.radius_px = max(1, int(radius_px))
        #: 是否处于"预捏合窗口"（仅 cursor_snap_windowed=True 时有意义）。
        #: 默认 True（非窗口化旧行为、以及启动首帧之前都保持"会吸"）。
        #: 窗口化后由 MouseController.set_snap_engaged 每帧写入：临近点击才 True，
        #: 平时 False —— 后台 UIA 探测因此完全静默，不再每 50ms 扫一遍桌面，
        #: 消除光标发顿/系统卡顿（2026-10-21 性能修复）。
        self._engaged = True
        #: 磁吸窗口**关掉后**的宽限期（秒）。见 ``snap_window_open``。
        #: 2026-10-23 加：他反馈"夹在两个小图标中间左右弹跳"。窗口一关就
        #: 立刻放弃锁存 → 光标从"控件中心"弹回"随手位置"，而他的手仍在
        #: 漂移，下一帧窗口又开、又锁到旁边那个图标 —— 一开一关就是他看到的
        #: 左右弹跳。所以关窗后先留着锁存，等手真的动了再松。
        self._release_grace_s = max(0.0, float(release_grace_seconds))
        self._grace_until = 0.0
        self._grace_px: tuple[int, int] | None = None
        self.poll_interval_seconds = max(0.02, float(poll_interval_seconds))
        self._foreground_provider = foreground_provider or _win32_foreground
        self._state = MagneticSnapState(
            self.radius_px,
            inside_only=inside_only,
            to_center=to_center,
        )
        self._candidate: SnapCandidate | None = None
        self._context: DesktopContext | None = None
        self._pending: tuple[int, int] | None = None
        self._lock = threading.RLock()
        self._request_event = threading.Event()
        self._stop_event = threading.Event()
        self._probe_initialized = False
        self._probe_disabled = False
        self._thread: threading.Thread | None = None
        if start_worker:
            self._thread = threading.Thread(
                target=self._run,
                name="cursor-snap-uia",
                daemon=True,
            )
            self._thread.start()

    @property
    def target_name(self) -> str | None:
        with self._lock:
            return self._state.target_name

    @property
    def target_id(self) -> str | None:
        """当前锁存控件的 UIA RuntimeId，供日志/HUD 标识"是哪个控件"。"""

        with self._lock:
            candidate = self._candidate
        return candidate.target_id if candidate is not None else None

    @property
    def target_rect(self) -> tuple[int, int, int, int] | None:
        """当前锁存目标的矩形；没有吸附时为 None。"""

        with self._lock:
            return self._state.latched_rect

    def invalidate(self) -> None:
        """环境变化（前台窗口切换、DPI/虚拟屏变化）时立刻丢弃缓存与锁存。"""

        with self._lock:
            self._candidate = None
            self._context = None
            self._state.clear()

    def _desktop_context(self) -> DesktopContext:
        foreground, title = self._foreground_provider()
        return DesktopContext(
            foreground=foreground,
            title=title,
            screen=self.screen_signature,
        )

    def configure(self, *, enabled: bool, radius_px: int) -> None:
        with self._lock:
            self.enabled = bool(enabled)
            self.radius_px = max(1, int(radius_px))
            self._candidate = None
            self._state.configure(self.radius_px)
        if self.enabled:
            self._request_event.set()

    def clear(self) -> None:
        with self._lock:
            self._candidate = None
            self._state.clear()

    def _drop_latch(self) -> None:
        """丢弃候选与锁存。调用方必须已持有锁。"""

        self._candidate = None
        self._state.clear()
        self._grace_until = 0.0
        self._grace_px = None

    def set_engaged(self, engaged: bool) -> None:
        """由 MouseController 每帧写入：是否处于"预捏合窗口"。

        仅 cursor_snap_windowed=True 时有意义。False 时令后台探测线程彻底静默
        （不再调 UI Automation），避免持续扫描桌面造成的光标发顿/系统卡顿。

        🔴 2026-10-23：**关窗不再立刻丢锁存**。原来 False 分支直接
        ``_candidate = None; _state.clear()``，于是"开—关—开"会把光标在
        "吸附点"和"随手位置"之间来回甩；配合捏合趋向锚定的开合抖动，
        表现就是他在两个相邻小图标之间看到的左右弹跳。现在改成延迟释放
        （``_release_grace_s``），真正的丢弃交给 ``snap_window_open``
        在宽限期结束时做。
        """

        with self._lock:
            if engaged:
                self._engaged = True
                self._grace_until = 0.0
                self._grace_px = None
                return
            was_engaged = self._engaged
            self._engaged = False
            if (
                was_engaged
                and self._release_grace_s > 0.0
                and self._candidate is not None
            ):
                self._grace_until = time.monotonic() + self._release_grace_s
                self._grace_px = self._pending
                return
            self._drop_latch()

    def snap_window_open(self, x: float, y: float) -> bool:
        """这一刻是否让 ``transform`` 参与（即磁吸窗口是否有效）。

        与 ``_engaged`` 的区别：``_engaged`` 是引擎给的**原始**开关，它还会
        决定后台探测线程是否工作；这里多一层**释放宽限**——窗口刚关掉的
        ``release_grace_seconds`` 之内、而且手没真正离开（原始位置离关窗点
        不超过 ``radius_px``），就继续沿用锁存，避免"关窗即弹回"。

        ``x``/``y`` 是归一化坐标；顺手把 ``_pending`` 刷新到最新，这样宽限期
        内探测线程即使被 ``_engaged`` 关掉，重新开窗时也能拿到新鲜位置。
        """

        with self._lock:
            if not self.enabled:
                return False
            pixel_x = self.origin_x + round(clamp(x) * (self.screen_width - 1))
            pixel_y = self.origin_y + round(clamp(y) * (self.screen_height - 1))
            self._pending = (pixel_x, pixel_y)
            if self._engaged:
                return True
            if self._grace_until <= 0.0:
                return False
            if time.monotonic() > self._grace_until:
                self._drop_latch()
                return False
            if self._grace_px is not None and hypot(
                pixel_x - self._grace_px[0],
                pixel_y - self._grace_px[1],
            ) > float(self.radius_px):
                # 手已经明确离开了：宽限立刻作废，不再拽着光标。
                self._drop_latch()
                return False
            return True

    def transform(self, x: float, y: float) -> tuple[float, float]:
        raw_x = clamp(x)
        raw_y = clamp(y)
        with self._lock:
            if not self.enabled:
                return raw_x, raw_y
            pixel_x = self.origin_x + round(raw_x * (self.screen_width - 1))
            pixel_y = self.origin_y + round(raw_y * (self.screen_height - 1))
            self._pending = (pixel_x, pixel_y)

            # 估算当前移动速度（指数平滑，避免单帧抖动误判）。
            now = time.monotonic()
            if self._last_sample_at:
                elapsed = now - self._last_sample_at
                if elapsed > 1e-4:
                    moved = hypot(
                        pixel_x - self._last_sample_x,
                        pixel_y - self._last_sample_y,
                    )
                    instant = moved / elapsed
                    self._speed_px_s = (
                        instant
                        if self._speed_px_s <= 0.0
                        else self._speed_px_s * 0.6 + instant * 0.4
                    )
            self._last_sample_x = pixel_x
            self._last_sample_y = pixel_y
            self._last_sample_at = now

            # 刚刚判定"在横跳"：先别吸，让用户把光标落到想要的位置。
            if now < self._suppress_until:
                return raw_x, raw_y

            if self._speed_px_s > self.max_speed_px_s:
                # 快速掠过：不吸附，同时松开锁存，避免"被拽住不放"。
                self._state.clear()
                return raw_x, raw_y

            candidate = self._candidate
            if candidate is not None and candidate.expires_at <= time.monotonic():
                # 快照过期：只丢弃候选，不动锁存——按下左键的瞬间若强行松锁，
                # 光标会当场跳走。锁存交给下面的滞回逻辑自然释放。
                candidate = None
                self._candidate = None
            snapped_x, snapped_y, snapped = self._state.resolve(
                pixel_x,
                pixel_y,
                candidate,
            )
            current_rect = self._state.latched_rect
            if current_rect != self._last_target_rect:
                self._last_target_rect = current_rect
                if current_rect is not None:
                    self._switch_times.append(now)
                    self._switch_times = [
                        moment
                        for moment in self._switch_times
                        if now - moment <= _SWITCH_WINDOW_SECONDS
                    ]
                    # 1 秒内吸住过 3 个不同目标 = 在横跳，松手 0.6 秒。
                    if len(self._switch_times) >= _SWITCH_MAX:
                        self._suppress_until = now + _SUPPRESS_SECONDS
                        self._switch_times.clear()
                        self._state.clear()
        self._request_event.set()
        if not snapped:
            return raw_x, raw_y
        return (
            clamp((snapped_x - self.origin_x) / (self.screen_width - 1)),
            clamp((snapped_y - self.origin_y) / (self.screen_height - 1)),
        )

    def refresh_once(self) -> None:
        """在线程内做一轮探测。只应由 worker 线程调用（COM 不能跨线程）。"""

        with self._lock:
            # _engaged=False（窗口化且未临近点击）时彻底跳过探测：不让后台线程
            # 每 50ms 扫一遍桌面 UI Automation，那是光标发顿/系统卡顿的元凶。
            if (
                not self.enabled
                or not self._engaged
                or self._pending is None
                or self._probe_disabled
            ):
                return
            x, y = self._pending
            radius_px = self.radius_px
        if not self._probe_initialized:
            self.probe.initialize()
            self._probe_initialized = True
            if getattr(self.probe, "unavailable", False):
                # 缺 uiautomation 或 COM 初始化失败：永久降级为纯坐标点击，
                # 不再每 60ms 重试一次（否则日志会被刷爆）。
                self._probe_disabled = True
                LOGGER.info("控件磁吸已降级为纯坐标点击，点击功能不受影响")
                return

        context = self._desktop_context()
        # 前台窗口/标题、DPI 模式或虚拟屏变了 → 旧候选与锁存一概作废。
        context_changed = self._context is not None and context != self._context
        try:
            candidate = self.probe.find_target(x, y, radius_px)
        except Exception:
            # UI 元素可能在探测期间被销毁；按 Windows 文档要求优雅降级。
            LOGGER.debug("磁吸探测失败，本轮降级", exc_info=True)
            candidate = None
        if self.fallback_probe is not None:
            # 🔴 桌面图标源**优先**，不是兜底（2026-10-23 改）。
            # UIA 把桌面图标认成 ListItemControl 并返回"图标+文字"整格矩形，
            # 用它算中心会落在名字上；只有直读列表项才能拿到"图案"那一块。
            # 它由 `_desktop_at()` 把关：光标不在桌面上时直接返回 None。
            try:
                desktop_candidate = self.fallback_probe.find_target(x, y, radius_px)
            except Exception:
                LOGGER.debug("桌面图标源失败，本轮降级", exc_info=True)
                desktop_candidate = None
            if desktop_candidate is not None:
                candidate = desktop_candidate
        if candidate is not None and candidate.expires_at <= time.monotonic():
            candidate = None
        with self._lock:
            if context_changed:
                self._state.clear()
            self._candidate = candidate
            self._context = context

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                self._request_event.wait(timeout=0.05)
                self._request_event.clear()
                if self._stop_event.is_set():
                    break
                try:
                    self.refresh_once()
                except Exception:
                    LOGGER.debug("UI Automation磁吸探测失败，已临时降级", exc_info=True)
                self._stop_event.wait(self.poll_interval_seconds)
        finally:
            if self._probe_initialized:
                self.probe.close()
                self._probe_initialized = False

    def shutdown(self) -> None:
        self._stop_event.set()
        self._request_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._thread is None and self._probe_initialized:
            self.probe.close()
            self._probe_initialized = False
