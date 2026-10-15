"""Windows控件磁吸的异步探测与纯几何状态机。

热路径 ``transform()`` 只做「加锁 + 读不可变快照 + 整数换算」，是常数时间；
真正跨进程的 UI Automation 查询只发生在 ``cursor-snap-uia`` 守护线程里，
并且由该线程自己完成 COM 初始化——COM 对象不跨线程传递。缺少
``uiautomation`` 或初始化失败时自动降级为纯坐标点击，点击照常可用。
"""

from __future__ import annotations

import ctypes
import logging
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


def _activation_distance(x: int, y: int, candidate: SnapCandidate) -> float:
    """返回磁吸激活距离；标签页只在中心锚点附近吸附。

    浏览器的TabItem矩形通常覆盖整张标签。若按矩形距离判断，光标一进入
    标签右侧就会被拉回中心，从而永远到不了独立的“关闭标签页”按钮。
    """

    if candidate.control_type == "TabItemControl":
        return hypot(x - candidate.x, y - candidate.y)
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
    ) -> None:
        self.radius_px = max(1, int(radius_px))
        self.release_multiplier = max(1.05, float(release_multiplier))
        self._latched: SnapCandidate | None = None

    @property
    def target_name(self) -> str | None:
        if self._latched is None:
            return None
        return self._latched.name or self._latched.control_type or "可点击控件"

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
            if _activation_distance(x, y, self._latched) <= release_radius:
                return self._latched.x, self._latched.y, True
            self._latched = None

        if (
            candidate is not None
            and _activation_distance(x, y, candidate) <= self.radius_px
        ):
            self._latched = candidate
            return candidate.x, candidate.y, True
        return x, y, False


class WindowsUiAutomationProbe:
    """使用Windows UI Automation查找光标附近的可点击控件。"""

    _CONTROL_TYPES = {
        50000,  # Button
        50002,  # CheckBox
        50005,  # Hyperlink
        50011,  # MenuItem
        50013,  # RadioButton
        50031,  # SplitButton
    }

    def __init__(self) -> None:
        self._auto: object | None = None
        self._initializer: object | None = None
        self._unavailable = False

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
        for _ in range(5):
            if control is None:
                return None
            try:
                control_type = int(control.ControlType)
            except Exception:
                return None
            if control_type in self._CONTROL_TYPES:
                break
            try:
                control = control.GetParentControl()
            except Exception:
                return None
        else:
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
        if not nearby:
            return None
        return min(
            nearby,
            key=lambda item: (
                _activation_distance(x, y, item),
                hypot(x - item.x, y - item.y),
            ),
        )


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
        poll_interval_seconds: float = 0.06,
    ) -> None:
        self.probe = probe
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
        self.poll_interval_seconds = max(0.02, float(poll_interval_seconds))
        self._foreground_provider = foreground_provider or _win32_foreground
        self._state = MagneticSnapState(self.radius_px)
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

    def transform(self, x: float, y: float) -> tuple[float, float]:
        raw_x = clamp(x)
        raw_y = clamp(y)
        with self._lock:
            if not self.enabled:
                return raw_x, raw_y
            pixel_x = self.origin_x + round(raw_x * (self.screen_width - 1))
            pixel_y = self.origin_y + round(raw_y * (self.screen_height - 1))
            self._pending = (pixel_x, pixel_y)
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
            if not self.enabled or self._pending is None or self._probe_disabled:
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
                self._request_event.wait(timeout=0.10)
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
