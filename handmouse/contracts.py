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

    🔴 ``landmarks`` 与 ``world_landmarks`` 是**两套不同的坐标**，不能混用：

    * ``landmarks``：归一化图像坐标。x 除以画面**宽**、y 除以画面**高**，
      两个方向的比例不同——640×480 的画面里水平方向被拉伸 4/3，所以
      "手指横着指"和"竖着指"量出来的长度比不一样。z 是相对深度（腕点为
      原点），尺度与 x 大致同量级，是网络猜的，不适合做几何判据。
    * ``world_landmarks``：**米制 3D 坐标**，原点在手部几何中心。没有画面
      宽高比问题、也没有透视压缩问题，指节夹角与长度比在真实三维空间里
      度量。**任何"手指伸没伸直"的判据都应该用它。**

    ``world_landmarks`` 为空表示该后端不提供 3D（例如回放、测试、局部跟踪
    降级），调用方必须能退回 2D 判据。

    ``image_aspect`` = 画面宽 ÷ 画面高，默认 4:3（本项目全部阈值都在这个
    宽高比下标定，所以默认值等于"旧行为"，回放与测试不受影响）。

    🔴 **为什么必须带上来**：所有"两点距离 ÷ 掌宽"的比值判据都对宽高比敏感。
    掌宽（5→17）是**纯横向**的，捏合比（4→8）是**斜向**的，两者受横向
    压缩的影响不一样。换一台 16:9 的摄像头，归一化坐标里的横向长度缩水
    25%，捏合比整体漂移 → 用户的捏合阈值全部失准（要么点不出来，要么
    一动就误触发）。手势引擎用 ``aspect ÷ (4/3)`` 把横向长度还原到标定
    尺度，换摄像头就不用重新标定。
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
    world_landmarks: tuple[Landmark, ...] = ()
    #: 画面宽 ÷ 高。4/3 表示"按标定尺度"，无需任何补偿。
    image_aspect: float = 4.0 / 3.0


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
    #: 五指张开待机（不移动光标、不发滚轮，只保留左右挥动翻页）。
    #: 滚轮模式已于 2026-10-19 按用户要求整体移除。
    OPEN_PALM = "open_palm"
    #: 双指滑动（食指+中指伸出、其余握起）：上下划 = 滚动，左右划 = 翻页。
    SWIPE = "swipe"
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
    #: 拇指张开度 ×100（dist(拇指尖,食指根)/掌宽）。用于校准「拇指张开=停住」。
    thumb_spread_percent: float = 0.0
    #: 食指-拇指间距 ×100（dist(食指尖,拇指尖)/掌宽）。「拇指张开」的直观测法。
    thumb_index_gap_percent: float = 0.0


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
    # 吸附目标的屏幕矩形 (left, top, right, bottom)，光环层用它画圈。
    snap_rect: tuple[int, int, int, int] | None = None
    metrics: RuntimeMetrics = field(default_factory=RuntimeMetrics)
    # 个人手势样本库状态说明：空串表示未启用（HUD 不显示这一项）。
    model_note: str = ""
    # 本帧样本库的原始判定（如 "open@0.42" / "point@0.88 拒识"）。
    model_debug: str = ""
    # 是否处于「拇指张开 = 停住」状态（HUD 用来给用户确认手势触发）。
    thumb_stop: bool = False
    # 「停住→捏合」锚定窗口内（HUD 用 ⚓ 标记）。
    pinch_anchor: bool = False
    #: 原地停留已满阈值（磁吸窗口打开的那个"临门一脚"）。
    #: 2026-10-23 加：它之前**一次都没亮过**（阈值不可达），加这个字段
    #: 就是为了让他能一眼看到"停住一秒到底有没有被认出来"。
    dwell_active: bool = False
    #: 当前已经停稳了多少毫秒（HUD 显示成 0.6/1.0s 的进度）。
    dwell_ms: float = 0.0
    #: 瞄准死区生效中（手指只动一点点 → 光标被钉住）。HUD 用 ⌖ 标记，
    #: 让用户能确认"光标不动"是故意锁住了，而不是卡住了。
    micro_lock: bool = False
    #: 悬浮球诊断：当前谁有球、透明度、球心坐标。看不见球时靠这行定位。
    ball_note: str = ""
    # HUD 调试行（VERBOSE_GATE 开启时由 pipeline 组装）。
    debug_lines: list[str] = field(default_factory=list)
