"""跨线程传递的数据契约。

所有时间戳均使用 ``time.monotonic_ns()`` 的同一时钟域，避免系统时间调整
造成状态机误判。注意二者虽同域但参考点不同：``HandObservation.timestamp_ns``
是**采集时刻**（由 ``captured_ns`` 量化到毫秒），状态机里的"现在"是**到达
墙钟**；跨用会把"取包延迟 + 帧间隔"算成经过时间。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True, slots=True)
class Landmark:
    """归一化手部关键点。"""

    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True, slots=True)
class FramePacket:
    """摄像头采集帧。"""

    frame: Any
    sequence: int
    captured_ns: int
    brightness: float


@dataclass(frozen=True, slots=True)
class HandObservation:
    """单只手的检测结果。

    MediaPipe Tasks 当前不会返回每个关键点各自的置信度。三个置信度字段
    因此为可选值，保留给未来模型或后端；当前实现使用配置中的阈值控制检测。
    """

    landmarks: tuple[Landmark, ...]
    handedness: str
    timestamp_ns: int
    inference_ms: float
    handedness_score: float = 0.0
    detection_confidence: float | None = None
    presence_confidence: float | None = None
    tracking_confidence: float | None = None
    partial_finger: str | None = None
    partial_position: Landmark | None = None


@dataclass(frozen=True, slots=True)
class DetectionPacket:
    """一次异步推理回调的数据。"""

    observations: tuple[HandObservation, ...]
    timestamp_ns: int
    inference_ms: float


class GestureType(str, Enum):
    """对界面公开的手势类型。"""

    NONE = "none"
    POINTER = "pointer"
    PINCH_ARMED = "pinch_armed"
    LEFT_PINCH = "left_pinch"
    RIGHT_PINCH = "right_pinch"
    DRAG = "drag"
    SCROLL = "scroll"
    PAGE_NEXT = "page_next"
    PAGE_PREVIOUS = "page_previous"
    PARTIAL_INDEX = "partial_index"
    PARTIAL_MIDDLE = "partial_middle"
    FIST = "fist"
    PAUSED = "paused"
    LOST = "lost"
    # 进入门：手在画面里但尚未做出进入姿势，处于"闸门未开"态。一切下游
    # 手势（捏合/翻页/拖拽）在此态一律不执行，用来挡住幻影手势。
    GATE_IDLE = "gate_idle"


class GesturePhase(str, Enum):
    """手势生命周期。"""

    STARTED = "started"
    UPDATED = "updated"
    ENDED = "ended"
    TRIGGERED = "triggered"


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """供界面展示和诊断使用的手势事件。"""

    gesture: GestureType
    phase: GesturePhase
    timestamp_ns: int
    duration_ms: float = 0.0


class MouseCommandType(str, Enum):
    """标准化鼠标指令。"""

    MOVE = "move"
    LEFT_CLICK = "left_click"
    LEFT_DOUBLE_CLICK = "left_double_click"
    LEFT_DOWN = "left_down"
    LEFT_UP = "left_up"
    RIGHT_CLICK = "right_click"
    WHEEL = "wheel"
    PAGE_NEXT = "page_next"
    PAGE_PREVIOUS = "page_previous"
    PAUSE = "pause"


@dataclass(frozen=True, slots=True)
class MouseCommand:
    """由状态机生成、由系统后端执行的鼠标指令。"""

    kind: MouseCommandType
    timestamp_ns: int
    x: float | None = None
    y: float | None = None
    delta: int = 0
    paused: bool | None = None


@dataclass(slots=True)
class RuntimeMetrics:
    """运行时性能快照。"""

    capture_fps: float = 0.0
    inference_fps: float = 0.0
    dropped_frames: int = 0
    latency_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    captured_frames: int = 0
    inferred_frames: int = 0
    hand_detection_rate: float = 0.0
    # 全程累计（自 start() 起）：基线判定用。hand_detection_rate 只是 3s 滚动窗，
    # 单独看它会被"最后几秒刚好没手"骗过。
    packets_total: int = 0
    hand_packets_total: int = 0
    handedness_score: float = 0.0
    palm_width_percent: float = 0.0


@dataclass(slots=True)
class PipelineSnapshot:
    """供GUI轮询的线程安全状态快照。"""

    running: bool = False
    paused: bool = False
    gesture: GestureType = GestureType.NONE
    message: str = "尚未启动"
    low_light: bool = False
    camera_signal_missing: bool = False
    camera_error: str | None = None
    snap_target_name: str | None = None
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    # HUD 调试行（VERBOSE_GATE 开启时由 pipeline 组装）。
    debug_lines: list[str] = field(default_factory=list)
