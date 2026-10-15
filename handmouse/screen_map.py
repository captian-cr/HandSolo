"""屏幕坐标抽象：DPI 感知 + 单屏/多屏统一映射。

引擎内部一律使用 0..1 归一化坐标（``map_activity_region`` 的输出）。本模块
是唯一负责"归一化 ↔ 真实桌面像素"换算的地方，目的是让下面三处共享同一套
坐标基准，避免任何一处写死主屏：

* 鼠标注入（``mouse.Win32MouseBackend``）
* 控件磁吸命中判定（``cursor_snap.CursorSnapService``，UIA 返回物理像素）
* 调试窗口的屏幕矩形绘制（``debug_view``）

多屏策略
--------
``multi_monitor=False``（默认）时只覆盖主显示器，行为与旧实现一致；
``multi_monitor=True`` 时覆盖虚拟桌面（所有显示器并集），并使用
``MOUSEEVENTF_VIRTUALDESK`` 发送绝对坐标——这是 Windows 官方要求的搭配，
否则绝对坐标会被钳在主屏内。

绝对坐标必须按虚拟桌面宽高归一化到 0—65535；原点不是 0,0 而是
``SM_XVIRTUALSCREEN/SM_YVIRTUALSCREEN``（副屏在主屏左侧时为负值），
所以换算里必须减去原点，否则副屏会整体偏移。

DPI
---
必须先 ``SetProcessDpiAwarenessContext`` 再读 ``GetSystemMetrics``，否则在
125%/150% 缩放下拿到的是逻辑像素，1200x800 逻辑屏换成物理坐标会整体压缩，
光标够不到右下角；多屏下还会因为各屏缩放不同而错位。
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass

IS_WINDOWS = os.name == "nt"

# GetSystemMetrics 索引
SM_CXSCREEN = 0
SM_CYSCREEN = 1
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
_DPI_PER_MONITOR_V2 = -4

_dpi_state: str = "未初始化"


def enable_dpi_awareness() -> str:
    """进程级 DPI 感知，必须在读取屏幕尺寸和注入坐标之前调用一次。

    返回最终生效的模式，便于日志排查（不同 Windows 版本支持的入口不同）。
    """

    global _dpi_state
    if not IS_WINDOWS:
        _dpi_state = "非Windows平台"
        return _dpi_state
    try:
        context = ctypes.c_void_p(_DPI_PER_MONITOR_V2)
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(context):
            _dpi_state = "PerMonitorV2"
            return _dpi_state
    except (AttributeError, OSError):
        pass
    try:
        # SHCore：0=未感知 1=系统 2=每显示器。旧系统没有 V2 上下文。
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        _dpi_state = "PerMonitor"
        return _dpi_state
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        _dpi_state = "System"
        return _dpi_state
    except (AttributeError, OSError):
        pass
    _dpi_state = "无DPI感知"
    return _dpi_state


def dpi_state() -> str:
    return _dpi_state


def system_dpi_scale() -> float:
    """系统缩放倍数，调试窗口用它把画面放大回视觉大小。"""

    if not IS_WINDOWS:
        return 1.0
    try:
        dpi = int(ctypes.windll.user32.GetDpiForSystem())
    except (AttributeError, OSError):
        dpi = 96
    return max(1.0, dpi / 96.0)


@dataclass(frozen=True, slots=True)
class ScreenMetrics:
    """一次采样得到的桌面几何（物理像素）。"""

    origin_x: int
    origin_y: int
    width: int
    height: int
    primary_width: int
    primary_height: int
    monitor_count: int


def _get_metrics(get: "ctypes._CFuncPtr") -> ScreenMetrics:
    primary_width = max(1, int(get(SM_CXSCREEN)))
    primary_height = max(1, int(get(SM_CYSCREEN)))
    virtual_x = int(get(SM_XVIRTUALSCREEN))
    virtual_y = int(get(SM_YVIRTUALSCREEN))
    virtual_width = max(1, int(get(SM_CXVIRTUALSCREEN)))
    virtual_height = max(1, int(get(SM_CYVIRTUALSCREEN)))
    monitors = 1
    try:
        monitors = max(1, int(ctypes.windll.user32.GetSystemMetrics(80)))
    except (AttributeError, OSError, ValueError):
        pass
    return ScreenMetrics(
        origin_x=virtual_x,
        origin_y=virtual_y,
        width=virtual_width,
        height=virtual_height,
        primary_width=primary_width,
        primary_height=primary_height,
        monitor_count=monitors,
    )


class ScreenMap:
    """把 0..1 归一化坐标换算为真实桌面像素，并给出绝对注入所需参数。

    全局只应存在一个实例，由 ``app`` 构造后注入鼠标后端与磁吸服务。
    """

    def __init__(
        self,
        *,
        multi_monitor: bool = False,
        virtual_desk_experimental: bool = False,
        dpi_aware: bool = True,
    ) -> None:
        self.multi_monitor = bool(multi_monitor)
        # 实验开关：即使 multi_monitor=False，也允许用 VIRTUALDESK 发送坐标，
        # 用于在多屏机器上先验证官方推荐路径再决定是否切换默认值。
        self.virtual_desk_experimental = bool(virtual_desk_experimental)
        if dpi_aware:
            enable_dpi_awareness()
        self._metrics = (
            _get_metrics(ctypes.windll.user32.GetSystemMetrics) if IS_WINDOWS
            else ScreenMetrics(0, 0, 1920, 1080, 1920, 1080, 1)
        )
        self.refresh()

    # ---------------- 采样 ----------------
    def refresh(self) -> None:
        if IS_WINDOWS:
            self._metrics = _get_metrics(ctypes.windll.user32.GetSystemMetrics)

    @property
    def use_virtual_desk(self) -> bool:
        """是否按虚拟桌面发送绝对坐标。"""

        return self.multi_monitor or self.virtual_desk_experimental

    @property
    def metrics(self) -> ScreenMetrics:
        return self._metrics

    @property
    def primary_size(self) -> tuple[int, int]:
        return self._metrics.primary_width, self._metrics.primary_height

    @property
    def origin(self) -> tuple[int, int]:
        """注入坐标系原点。单屏固定为 0,0；多屏取虚拟桌面左上角。"""

        if self.use_virtual_desk:
            return self._metrics.origin_x, self._metrics.origin_y
        return 0, 0

    @property
    def size(self) -> tuple[int, int]:
        """注入坐标系尺寸。归一化与磁吸像素换算都以它为基准。"""

        if self.use_virtual_desk:
            return self._metrics.width, self._metrics.height
        return self._metrics.primary_width, self._metrics.primary_height

    @property
    def monitor_count(self) -> int:
        return self._metrics.monitor_count

    # ---------------- 换算 ----------------
    def to_pixels(self, normalized_x: float, normalized_y: float) -> tuple[float, float]:
        """归一化 → 真实桌面像素（含原点偏移）。"""

        width, height = self.size
        origin_x, origin_y = self.origin
        x = min(max(float(normalized_x), 0.0), 1.0)
        y = min(max(float(normalized_y), 0.0), 1.0)
        return (
            origin_x + x * (width - 1),
            origin_y + y * (height - 1),
        )

    def to_normalized(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """真实桌面像素 → 归一化（供磁吸/UIA 结果回灌引擎）。"""

        width, height = self.size
        origin_x, origin_y = self.origin
        return (
            min(max((float(pixel_x) - origin_x) / max(1, width - 1), 0.0), 1.0),
            min(max((float(pixel_y) - origin_y) / max(1, height - 1), 0.0), 1.0),
        )

    def absolute_coordinates(
        self,
        normalized_x: float,
        normalized_y: float,
    ) -> tuple[int, int]:
        """按当前坐标系换算 SendInput 需要的 0—65535 绝对坐标。

        归一化基准是"尺寸-1"而不是尺寸，这样 1.0 恰好落在最后一个像素上，
        避免光标永远差一个像素。
        """

        width, height = self.size
        x = min(max(float(normalized_x), 0.0), 1.0)
        y = min(max(float(normalized_y), 0.0), 1.0)
        return (
            round(x * 65535),
            round(y * 65535),
        )

    def contains_pixel(self, pixel_x: float, pixel_y: float) -> bool:
        origin_x, origin_y = self.origin
        width, height = self.size
        return (
            origin_x <= pixel_x < origin_x + width
            and origin_y <= pixel_y < origin_y + height
        )

    def describe(self) -> str:
        origin_x, origin_y = self.origin
        width, height = self.size
        return (
            f"DPI={dpi_state()} 显示器={self.monitor_count} "
            f"注入坐标系=({origin_x},{origin_y}) {width}x{height} "
            f"主屏={self._metrics.primary_width}x{self._metrics.primary_height} "
            f"virtual_desk={self.use_virtual_desk}"
        )
