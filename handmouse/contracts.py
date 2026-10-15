"""跨线程传递的数据契约。

所有时间戳均使用 ``time.monotonic_ns()`` 的同一时钟域，避免系统时间调整
造成状态机误判。
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
    # 第二阶段：轨迹手势（DTW）
    MOTION_ARMED = "motion_armed"
    MOTION_DRAWING = "motion_drawing"
    MOTION_CANDIDATE = "motion_candidate"


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


class ActionType(str, Enum):
    """轨迹手势确认后执行的"非鼠标"动作。"""

    LAUNCH = "launch"
    HOTKEY = "hotkey"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class ActionCommand:
    """轨迹手势/LAUNCH 产生的动作指令。

    与 ``MouseCommand`` 分开：这类动作需要白名单校验和键盘注入，语义与
    "移动光标"完全不同，混进鼠标指令会让后端越权。
    """

    kind: ActionType
    timestamp_ns: int
    name: str = ""
    target: str = ""


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
    # 第二阶段：DTW 候选与待确认动作
    motion_candidates: tuple[str, ...] = ()
    motion_pending: str = ""
    last_action: str = ""
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
