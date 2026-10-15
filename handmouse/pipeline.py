"""采集 → 推理 → 手势 → 输入的实时流水线。

线程模型（与 Vimoose 一致，实测够用）：
    camera-capture   → LatestSlot[FramePacket]  （容量1，覆盖即丢，避免积压延迟）
    hand-landmarker  → LatestSlot[DetectionPacket]
    gesture-control  → 消费最新检测，驱动状态机并注入输入

``psutil`` 只用于 CPU/内存读数，属于可选依赖，缺失时自动降级为不采样资源。
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from math import hypot
from pathlib import Path

from handmouse.config import AppConfig
from handmouse.contracts import (
    ActionCommand,
    DetectionPacket,
    FramePacket,
    GestureType,
    MouseCommand,
    MouseCommandType,
    PipelineSnapshot,
    RuntimeMetrics,
)
from handmouse.filters import CursorMotionInterpolator
from handmouse.gestures import GestureEngine
from handmouse.latest_slot import LatestSlot
from handmouse.launch import LaunchDispatcher
from handmouse.motion_gestures import MotionGestureEngine, MotionOutcome
from handmouse.mouse import MouseController
from handmouse.vision import CameraWorker, HandLandmarkerWorker, RollingRate, default_model_path

try:  # 可选：没有 psutil 时只是不显示 CPU/内存
    import psutil
except ImportError:  # pragma: no cover - 取决于环境
    psutil = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class ControlWorker(threading.Thread):
    """消费最新检测结果并驱动状态机。"""

    def __init__(
        self,
        detections: LatestSlot[DetectionPacket],
        engine: GestureEngine,
        mouse: MouseController,
        stop_event: threading.Event,
        on_state: Callable[[GestureType, bool], None],
        interpolation_ms: int,
        motion: MotionGestureEngine | None = None,
        dispatcher: LaunchDispatcher | None = None,
        on_motion: Callable[[MotionOutcome], None] | None = None,
    ) -> None:
        super().__init__(name="gesture-control", daemon=True)
        self.detections = detections
        self.engine = engine
        self.mouse = mouse
        self.stop_event = stop_event
        self.on_state = on_state
        self.motion_engine = motion
        self.dispatcher = dispatcher
        self.on_motion = on_motion
        self.motion_state = motion.state if motion is not None else None
        self.motion_outcome: MotionOutcome | None = None
        self.last_action: str = ""
        # 注意：``self.motion`` 是光标插值器（Vimoose 的命名），与上面的轨迹
        # 手势引擎 ``self.motion_engine`` 是两件事，不要混。
        self.motion = CursorMotionInterpolator(
            min_duration_ms=12,
            max_duration_ms=interpolation_ms,
        )
        self._motion_mode = False

    def _execute_commands(self, commands: list[MouseCommand]) -> None:
        now = time.monotonic_ns()
        immediate: list[MouseCommand] = []
        for command in commands:
            if (
                command.kind == MouseCommandType.MOVE
                and command.x is not None
                and command.y is not None
            ):
                first = self.motion.set_target(command.x, command.y, now)
                if first is not None:
                    immediate.append(
                        MouseCommand(
                            MouseCommandType.MOVE,
                            now,
                            x=first[0],
                            y=first[1],
                        ),
                    )
            else:
                immediate.append(command)
                if command.kind == MouseCommandType.PAUSE:
                    self.motion.clear()
        # 实机验证要靠日志复盘点/拖/滚有没有丢键，所以把非移动类指令记下来。
        # WHEEL 是连续量、每帧都可能出，记了会淹没有用信息，故跳过。
        for command in immediate:
            if command.kind not in (MouseCommandType.MOVE, MouseCommandType.WHEEL):
                LOGGER.info("[注入] %s", command.kind.value)
        self.mouse.execute_many(immediate)

    def _advance_motion(self, timestamp_ns: int) -> None:
        if self._motion_mode:
            # 轨迹手势期间光标必须完全冻结，连插值尾巴也不能漏出去。
            self.motion.clear()
            return
        if self.engine.paused or self.engine.current_gesture == GestureType.LOST:
            self.motion.clear()
            return
        if self.engine.current_gesture not in {
            GestureType.POINTER,
            GestureType.DRAG,
            GestureType.PARTIAL_INDEX,
        }:
            self.motion.cancel()
            return
        position = self.motion.sample(timestamp_ns)
        if position is None:
            return
        self.mouse.execute(
            MouseCommand(
                MouseCommandType.MOVE,
                timestamp_ns,
                x=position[0],
                y=position[1],
            ),
        )

    # ---------------- 轨迹手势 ----------------
    def _enter_motion_mode(self) -> None:
        self._motion_mode = True
        self.motion.clear()

    def _leave_motion_mode(self) -> None:
        if not self._motion_mode:
            return
        self._motion_mode = False
        # 绘制期间滤波器没被喂数据，直接复用会导致恢复瞬间光标跳一下。
        self.engine.filter.reset()
        self.motion.clear()

    def _dispatch(self, action: ActionCommand) -> None:
        if self.dispatcher is None:
            return
        result = self.dispatcher.dispatch(action)
        self.last_action = result.message
        if not result.ok:
            LOGGER.warning("动作未执行：%s", result.message)

    def _handle_observation(self, observation) -> None:
        outcome: MotionOutcome | None = None
        motion = self.motion_engine
        if motion is not None and motion.config.motion_gestures_enabled:
            outcome = motion.process(observation, observation.timestamp_ns)
            self.motion_state = outcome.state
            self.motion_outcome = outcome
            if self.on_motion is not None:
                self.on_motion(outcome)
            if outcome.action is not None:
                self._dispatch(outcome.action)

        if outcome is not None and outcome.active:
            self._enter_motion_mode()
            # 只释放可能按住的左键，不把观测交给鼠标状态机。
            self._execute_commands(self.engine.hold(observation.timestamp_ns))
            self.on_state(outcome.gesture, self.engine.paused)
            return

        self._leave_motion_mode()
        commands = self.engine.process(observation)
        if self.engine.current_gesture in {
            GestureType.PINCH_ARMED,
            GestureType.LEFT_PINCH,
            GestureType.RIGHT_PINCH,
        }:
            # 在执行点击前立即丢弃上一帧尚未完成的插值轨迹。
            self.motion.cancel()
        self._execute_commands(commands)

    def _handle_tick(self, timestamp_ns: int) -> None:
        motion = self.motion_engine
        if motion is not None:
            outcome = motion.tick(timestamp_ns)
            self.motion_state = outcome.state
            self.motion_outcome = outcome
            if self.on_motion is not None:
                self.on_motion(outcome)
            if outcome.action is not None:
                self._dispatch(outcome.action)
            if outcome.active:
                self._execute_commands(self.engine.hold(timestamp_ns))
                self.on_state(outcome.gesture, self.engine.paused)
                return
        self._leave_motion_mode()
        self._execute_commands(self.engine.tick(timestamp_ns))
        # tick 是最容易发生状态迁移的路径（手离画 → lost），但这里过去没有回调
        # on_state：迁移既不落 [状态] 日志，也不更新快照手势，表现为"手早就不在
        # 画面里、状态却冻在上一个手势"。on_state 内部按 (gesture, paused) 去重，
        # 8ms 一次的高频调用不会刷屏。
        self.on_state(self.engine.current_gesture, self.engine.paused)

    def run(self) -> None:
        revision = 0
        try:
            while not self.stop_event.is_set():
                packet, next_revision = self.detections.wait_next(revision, timeout=0.008)
                now = time.monotonic_ns()
                if packet is not None:
                    revision = next_revision
                    if packet.observations:
                        self._handle_observation(packet.observations[0])
                        if self.motion_outcome is None or not self.motion_outcome.active:
                            self.on_state(self.engine.current_gesture, self.engine.paused)
                    else:
                        self._handle_tick(packet.timestamp_ns)
                self._handle_tick(now)
                self._advance_motion(time.monotonic_ns())
        except Exception:
            LOGGER.exception("控制线程异常")
        finally:
            self.mouse.release_all()


class GesturePipeline:
    """管理所有工作线程，并向调试界面提供快照。"""

    def __init__(
        self,
        config: AppConfig,
        mouse: MouseController,
        model_path: Path | None = None,
        dispatcher: LaunchDispatcher | None = None,
    ) -> None:
        self.config = config
        self.mouse = mouse
        self.model_path = model_path or default_model_path()
        self.engine = GestureEngine(config)
        self.motion_engine = MotionGestureEngine(config)
        self.dispatcher = dispatcher

        self.frames: LatestSlot[FramePacket] = LatestSlot()
        self.detections: LatestSlot[DetectionPacket] = LatestSlot()
        self.preview: LatestSlot[FramePacket] = LatestSlot()
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.RLock()
        self._snapshot = PipelineSnapshot()
        self._control: ControlWorker | None = None
        self._capture_rate = RollingRate()
        self._inference_rate = RollingRate()
        self._process = psutil.Process() if psutil is not None else None
        if self._process is not None:
            self._process.cpu_percent(interval=None)
        self._captured_frames = 0
        self._inferred_frames = 0
        self._last_resource_sample_ns = 0
        self._latencies: deque[float] = deque(maxlen=600)
        self._detection_results: deque[tuple[int, bool]] = deque(maxlen=180)
        self._packets_total = 0
        self._hand_packets_total = 0
        # 轨迹手势状态日志去重用的上一轮签名，避免每帧刷屏。
        self._motion_signature: tuple[str, tuple[str, ...], str] | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._snapshot.running

    def _set_error(self, message: str) -> None:
        LOGGER.error(message)
        self.stop_event.set()
        self.mouse.release_all()
        with self._lock:
            self._snapshot.camera_error = message
            self._snapshot.message = message
            self._snapshot.running = False

    def _on_frame(self, packet: FramePacket) -> None:
        self.preview.put(packet)
        self._capture_rate.mark(packet.captured_ns)
        with self._lock:
            self._captured_frames += 1
            self._snapshot.low_light = packet.brightness < 55.0
            self._snapshot.camera_signal_missing = packet.brightness < 12.0

    def _on_detection(self, packet: DetectionPacket) -> None:
        self._inference_rate.mark()
        now = time.monotonic_ns()
        with self._lock:
            self._inferred_frames += 1
            latency_ms = max(
                packet.inference_ms,
                0.0,
                (now - packet.timestamp_ns) / 1_000_000,
            )
            self._snapshot.metrics.latency_ms = latency_ms
            self._latencies.append(latency_ms)
            has_hand = bool(packet.observations)
            self._detection_results.append((now, has_hand))
            self._packets_total += 1
            if has_hand:
                self._hand_packets_total += 1
            self._snapshot.metrics.handedness_score = (
                packet.observations[0].handedness_score if has_hand else 0.0
            )
            if has_hand and len(packet.observations[0].landmarks) == 21:
                points = packet.observations[0].landmarks
                self._snapshot.metrics.palm_width_percent = 100.0 * hypot(
                    points[5].x - points[17].x,
                    points[5].y - points[17].y,
                )
            else:
                self._snapshot.metrics.palm_width_percent = 0.0

    def _on_motion(self, outcome: MotionOutcome) -> None:
        candidates = tuple(candidate.name for candidate in outcome.candidates)
        with self._lock:
            self._snapshot.motion_candidates = candidates
            self._snapshot.motion_pending = outcome.pending
            if outcome.active:
                self._snapshot.message = outcome.message
        # 只在轨迹状态/候选真正变化时记一行，用于实机复盘（悬停误触发、画完是否只出候选）。
        signature = (outcome.state.value, candidates, outcome.pending)
        if signature != self._motion_signature:
            self._motion_signature = signature
            LOGGER.info(
                "[轨迹] state=%s candidates=%s pending=%s | %s",
                outcome.state.value,
                ",".join(candidates) or "-",
                outcome.pending or "-",
                outcome.message or "-",
            )

    def _on_state(self, gesture: GestureType, paused: bool) -> None:
        with self._lock:
            previous = (self._snapshot.gesture, self._snapshot.paused)
            self._snapshot.gesture = gesture
            self._snapshot.paused = paused
            # 摄像头出错时保留错误消息，但仍然要记录状态迁移（恢复过程靠日志复盘）。
            if not self._snapshot.camera_error:
                if self._snapshot.camera_signal_missing:
                    self._snapshot.message = "摄像头已打开，但没有有效视频画面"
                elif paused:
                    self._snapshot.message = "控制已暂停"
                elif gesture == GestureType.LOST:
                    self._snapshot.message = "未检测到有效手部"
                elif gesture == GestureType.GATE_IDLE:
                    self._snapshot.message = "未进入控制态（掌心朝镜 + 五指张开）"
                elif gesture == GestureType.DRAG:
                    self._snapshot.message = "正在拖拽"
                elif gesture == GestureType.SCROLL:
                    self._snapshot.message = "滚轮模式"
                elif gesture == GestureType.PAGE_NEXT:
                    self._snapshot.message = "已翻到下一页"
                elif gesture == GestureType.PAGE_PREVIOUS:
                    self._snapshot.message = "已翻到上一页"
                elif gesture == GestureType.PINCH_ARMED:
                    self._snapshot.message = "点击位置已锁定"
                elif gesture == GestureType.LEFT_PINCH:
                    self._snapshot.message = "左键捏合"
                elif gesture == GestureType.RIGHT_PINCH:
                    self._snapshot.message = "右键捏合"
                elif gesture == GestureType.POINTER:
                    self._snapshot.message = "正在控制光标"
                elif gesture == GestureType.PARTIAL_INDEX:
                    self._snapshot.message = "食指局部续跟踪"
                elif gesture == GestureType.PARTIAL_MIDDLE:
                    self._snapshot.message = "中指局部续跟踪"
                elif gesture in {
                    GestureType.MOTION_ARMED,
                    GestureType.MOTION_DRAWING,
                    GestureType.MOTION_CANDIDATE,
                }:
                    pass  # 轨迹手势的消息由 _on_motion 提供，避免互相覆盖
                else:
                    self._snapshot.message = "等待手势"
        if (gesture, paused) != previous:
            LOGGER.info(
                "[状态] %s(paused=%s) → %s(paused=%s)",
                previous[0].value,
                previous[1],
                gesture.value,
                paused,
            )

    def start(self) -> bool:
        with self._lock:
            if self._snapshot.running:
                return True
            if not self.model_path.exists():
                self._set_error(
                    f"缺少模型文件：{self.model_path}。请先下载 hand_landmarker.task。",
                )
                return False
            self._snapshot = PipelineSnapshot(running=True, message="正在启动摄像头")
            self._captured_frames = 0
            self._inferred_frames = 0
            self._latencies.clear()
            self._detection_results.clear()
            self.stop_event = threading.Event()
            self.frames = LatestSlot()
            self.detections = LatestSlot()
            self.preview = LatestSlot()
            self.engine = GestureEngine(self.config)
            self.motion_engine = MotionGestureEngine(self.config)

        camera = CameraWorker(
            self.config,
            self.frames,
            self.stop_event,
            self._set_error,
            self._on_frame,
        )
        landmarker = HandLandmarkerWorker(
            self.config,
            self.model_path,
            self.frames,
            self.detections,
            self.stop_event,
            self._set_error,
            self._on_detection,
            lambda: self.engine.paused,
        )
        control = ControlWorker(
            self.detections,
            self.engine,
            self.mouse,
            self.stop_event,
            self._on_state,
            self.config.cursor_interpolation_ms,
            motion=self.motion_engine,
            dispatcher=self.dispatcher,
            on_motion=self._on_motion,
        )
        self._control = control
        self._threads = [camera, landmarker, control]
        for thread in self._threads:
            thread.start()
        LOGGER.info("实时流水线已启动")
        return True

    def stop(self) -> None:
        with self._lock:
            had_work = self._snapshot.running or bool(self._threads)
        if not had_work:
            self.mouse.release_all()
            return
        self.stop_event.set()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._threads.clear()
        self.mouse.release_all()
        with self._lock:
            self._snapshot.running = False
            self._snapshot.paused = False
            self._snapshot.gesture = GestureType.NONE
            if not self._snapshot.camera_error:
                self._snapshot.message = "控制已停止"
        LOGGER.info("实时流水线已停止")

    def set_paused(self, paused: bool) -> None:
        commands = self.engine.set_paused(paused)
        self.mouse.execute_many(commands)
        self._on_state(self.engine.current_gesture, paused)

    def toggle_paused(self) -> None:
        self.set_paused(not self.engine.paused)

    def emergency_stop(self) -> None:
        commands = self.engine.emergency_stop()
        self.mouse.execute_many(commands)
        self.mouse.release_all()
        self._on_state(GestureType.PAUSED, True)

    def update_config(self, config: AppConfig) -> None:
        was_running = self.running
        if was_running:
            self.stop()
        self.config = config
        self.engine.update_config(config)
        self.motion_engine.update_config(config)
        self.mouse.update_config(config)
        if self.dispatcher is not None:
            self.dispatcher.update_config(config)
        if was_running:
            self.start()

    def get_preview(self) -> tuple[FramePacket | None, DetectionPacket | None]:
        frame, _ = self.preview.get_latest()
        detection, _ = self.detections.get_latest()
        return frame, detection

    @property
    def motion_outcome(self) -> MotionOutcome | None:
        control = self._control
        return control.motion_outcome if control is not None else None

    def snapshot(self) -> PipelineSnapshot:
        now = time.monotonic_ns()
        with self._lock:
            metrics = self._snapshot.metrics
            metrics.capture_fps = self._capture_rate.value()
            metrics.inference_fps = self._inference_rate.value()
            metrics.dropped_frames = self.frames.overwrite_count
            metrics.captured_frames = self._captured_frames
            metrics.inferred_frames = self._inferred_frames
            detection_cutoff = now - 3_000_000_000
            while (
                self._detection_results
                and self._detection_results[0][0] < detection_cutoff
            ):
                self._detection_results.popleft()
            if self._detection_results:
                metrics.hand_detection_rate = 100.0 * sum(
                    1 for _timestamp, has_hand in self._detection_results if has_hand
                ) / len(self._detection_results)
            metrics.packets_total = self._packets_total
            metrics.hand_packets_total = self._hand_packets_total
            if self._latencies:
                ordered = sorted(self._latencies)
                metrics.latency_p50_ms = ordered[round((len(ordered) - 1) * 0.50)]
                metrics.latency_p95_ms = ordered[round((len(ordered) - 1) * 0.95)]

            control = self._control
            if control is not None:
                self._snapshot.last_action = control.last_action
            self._sample_resources(now)

            if self.mouse.last_error:
                self._snapshot.message = self.mouse.last_error
            snapshot = copy.deepcopy(self._snapshot)
            snapshot.snap_target_name = self.mouse.snap_target_name
            if (
                snapshot.snap_target_name
                and snapshot.gesture
                in {
                    GestureType.POINTER,
                    GestureType.PINCH_ARMED,
                    GestureType.LEFT_PINCH,
                    GestureType.RIGHT_PINCH,
                }
            ):
                snapshot.message = f"已吸附：{snapshot.snap_target_name}"
            return snapshot

    def _sample_resources(self, now: int) -> None:
        """CPU/内存按 1Hz 采样；没有 psutil 时静默跳过。"""

        if self._process is None:
            return
        if now - self._last_resource_sample_ns < 1_000_000_000:
            return
        # psutil 的进程 CPU 百分比可超过 100%（按逻辑核数计），这里折算成
        # 更接近任务管理器的"整机占比"口径。
        raw_cpu = self._process.cpu_percent(interval=None)
        cpu_count = (psutil.cpu_count() if psutil is not None else None) or 1
        self._snapshot.metrics.cpu_percent = raw_cpu / max(1, cpu_count)
        self._snapshot.metrics.memory_mb = self._process.memory_info().rss / (1024 * 1024)
        self._last_resource_sample_ns = now
