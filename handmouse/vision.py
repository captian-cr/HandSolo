"""摄像头采集和MediaPipe异步手部检测。"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import types
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _install_frozen_matplotlib_stub() -> None:
    """为冻结版屏蔽MediaPipe未使用的绘图依赖。

    MediaPipe顶层包会导入 ``drawing_utils``，后者无条件导入matplotlib；
    Vimoose自行用OpenCV绘制关键点，不会调用这些绘图函数。
    """

    if not getattr(sys, "frozen", False) or "matplotlib" in sys.modules:
        return
    matplotlib_stub = types.ModuleType("matplotlib")
    matplotlib_stub.__path__ = []  # type: ignore[attr-defined]
    pyplot_stub = types.ModuleType("matplotlib.pyplot")
    matplotlib_stub.pyplot = pyplot_stub  # type: ignore[attr-defined]
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub


_install_frozen_matplotlib_stub()

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from handmouse.config import AppConfig
from handmouse.contracts import (
    DetectionPacket,
    FramePacket,
    HandObservation,
    Landmark,
)
from handmouse.latest_slot import LatestSlot
from handmouse.partial_tracking import PartialFingerTracker


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RecognitionTransform:
    """模型输入裁剪区域到原始画面的归一化坐标变换。"""

    offset_x: float = 0.0
    offset_y: float = 0.0
    scale_x: float = 1.0
    scale_y: float = 1.0

    def restore(self, point: Landmark) -> Landmark:
        return Landmark(
            x=self.offset_x + point.x * self.scale_x,
            y=self.offset_y + point.y * self.scale_y,
            z=point.z * self.scale_x,
        )


def prepare_recognition_frame(
    frame: np.ndarray,
    zoom: float,
) -> tuple[np.ndarray, RecognitionTransform]:
    """温和放大中央识别区，并返回可逆坐标变换。

    这不是跟踪ROI：裁剪区固定居中、不会追随手部移动。默认1.15倍只裁掉
    四周约6.5%，能提高远手和小手在模型输入中的有效像素，同时保留大部分
    画面。设置为1.0时零拷贝返回原帧。
    """

    zoom = max(1.0, min(1.35, float(zoom)))
    if zoom <= 1.001:
        return frame, RecognitionTransform()
    height, width = frame.shape[:2]
    crop_width = max(2, round(width / zoom))
    crop_height = max(2, round(height / zoom))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    crop = frame[top : top + crop_height, left : left + crop_width]
    resized = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)
    return (
        resized,
        RecognitionTransform(
            offset_x=left / width,
            offset_y=top / height,
            scale_x=crop_width / width,
            scale_y=crop_height / height,
        ),
    )


def default_model_path() -> Path:
    """定位随源码或打包程序分发的模型文件。"""

    return Path(__file__).resolve().parent / "models" / "hand_landmarker.task"


class RollingRate:
    """统计最近两秒事件速率。"""

    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_ns = int(window_seconds * 1_000_000_000)
        self._samples: deque[int] = deque()
        self._lock = threading.Lock()

    def mark(self, timestamp_ns: int | None = None) -> None:
        now = timestamp_ns or time.monotonic_ns()
        with self._lock:
            self._samples.append(now)
            cutoff = now - self.window_ns
            while self._samples and self._samples[0] < cutoff:
                self._samples.popleft()

    def value(self) -> float:
        now = time.monotonic_ns()
        with self._lock:
            cutoff = now - self.window_ns
            while self._samples and self._samples[0] < cutoff:
                self._samples.popleft()
            if len(self._samples) < 2:
                return 0.0
            duration = (self._samples[-1] - self._samples[0]) / 1_000_000_000
            return (len(self._samples) - 1) / max(duration, 1e-6)


class CameraWorker(threading.Thread):
    """持续采集并覆盖最新帧。掉线后自动重连，不终止整条流水线。"""

    # 连续读失败达到这么多次就判定掉线，释放句柄后按退避重连。
    RECONNECT_AFTER_FAILURES = 10

    def __init__(
        self,
        config: AppConfig,
        output: LatestSlot[FramePacket],
        stop_event: threading.Event,
        on_error: Callable[[str], None],
        on_frame: Callable[[FramePacket], None],
        on_warning: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(name="camera-capture", daemon=True)
        self.config = config
        self.output = output
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_frame = on_frame
        # 非致命告警通道。摄像头拔插属于外设事件，不能走 on_error——
        # GesturePipeline._set_error 会直接停掉流水线并释放按键，插回去也回不来。
        self.on_warning = on_warning or (lambda message: LOGGER.warning(message))

    def _open_camera(self) -> cv2.VideoCapture:
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.config.camera_index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(self.config.camera_index)
        if os.name == "nt":
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)
        capture.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    @staticmethod
    def _brightness(frame: np.ndarray) -> float:
        small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    @staticmethod
    def _enhance_low_light(frame: np.ndarray, brightness: float) -> np.ndarray:
        """按亮度自适应提升暗部，同时限制增强强度以保护模型输入分布。"""

        if brightness >= 55.0:
            return frame
        gamma = max(0.62, min(0.90, brightness / 60.0))
        lookup = np.array(
            [((value / 255.0) ** gamma) * 255 for value in range(256)],
            dtype=np.uint8,
        )
        enhanced = cv2.LUT(frame, lookup)
        if brightness < 30.0:
            # 极暗画面只对亮度通道做温和局部对比度增强，避免扭曲颜色。
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            lightness, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            lightness = clahe.apply(lightness)
            enhanced = cv2.cvtColor(
                cv2.merge((lightness, a_channel, b_channel)),
                cv2.COLOR_LAB2BGR,
            )
        return enhanced

    def _reopen_camera(self) -> cv2.VideoCapture | None:
        """掉线后按退避反复重开摄像头，直到成功或收到停止信号。

        USB 重插需要几秒枚举时间，所以退避从 0.5s 起步、上限 3s；返回 None
        表示收到了停止信号，调用方应结束线程。
        """

        attempt = 0
        while not self.stop_event.is_set():
            capture = self._open_camera()
            if capture.isOpened():
                self.on_warning("摄像头已恢复。")
                return capture
            capture.release()
            attempt += 1
            self.on_warning(f"摄像头无信号，正在重连（第 {attempt} 次）…")
            if self.stop_event.wait(min(0.5 + 0.25 * attempt, 3.0)):
                return None
        return None

    def run(self) -> None:
        capture = self._open_camera()
        if not capture.isOpened():
            self.on_error(
                f"无法打开摄像头 {self.config.camera_index}，请检查占用和隐私权限。",
            )
            return

        sequence = 0
        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                captured_ns = time.monotonic_ns()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= self.RECONNECT_AFTER_FAILURES:
                        self.on_warning("摄像头已断开，正在尝试重连…")
                        capture.release()
                        reopened = self._reopen_camera()
                        if reopened is None:
                            return
                        capture = reopened
                        consecutive_failures = 0
                        continue
                    time.sleep(0.03)
                    continue

                consecutive_failures = 0
                if self.config.mirror:
                    frame = cv2.flip(frame, 1)
                brightness = self._brightness(frame)
                frame = self._enhance_low_light(frame, brightness)
                sequence += 1
                packet = FramePacket(
                    frame=frame,
                    sequence=sequence,
                    captured_ns=captured_ns,
                    brightness=brightness,
                )
                self.output.put(packet)
                self.on_frame(packet)
        except Exception:
            LOGGER.exception("摄像头采集线程异常")
            self.on_error("摄像头采集发生异常，控制已安全停止。")
        finally:
            capture.release()


class HandLandmarkerWorker(threading.Thread):
    """向MediaPipe实时模式提交最新帧并接收异步结果。"""

    def __init__(
        self,
        config: AppConfig,
        model_path: Path,
        input_slot: LatestSlot[FramePacket],
        output_slot: LatestSlot[DetectionPacket],
        stop_event: threading.Event,
        on_error: Callable[[str], None],
        on_detection: Callable[[DetectionPacket], None],
        paused_provider: Callable[[], bool],
    ) -> None:
        super().__init__(name="hand-landmarker", daemon=True)
        self.config = config
        self.model_path = model_path
        self.input_slot = input_slot
        self.output_slot = output_slot
        self.stop_event = stop_event
        self.on_error = on_error
        self.on_detection = on_detection
        self.paused_provider = paused_provider
        self._submitted_at: dict[int, tuple[int, RecognitionTransform]] = {}
        self._submitted_lock = threading.Lock()
        self._last_timestamp_ms = 0
        self._partial_tracker = PartialFingerTracker(
            max_frames=self.config.partial_tracking_frames,
        )
        self._last_handedness = ""
        self._last_handedness_score = 0.0

    def _callback(
        self,
        result: vision.HandLandmarkerResult,
        _output_image: mp.Image,
        timestamp_ms: int,
    ) -> None:
        callback_ns = time.monotonic_ns()
        with self._submitted_lock:
            submitted_ns, transform = self._submitted_at.pop(
                timestamp_ms,
                (callback_ns, RecognitionTransform()),
            )
            if len(self._submitted_at) > 60:
                oldest = sorted(self._submitted_at)[:-30]
                for key in oldest:
                    self._submitted_at.pop(key, None)
        inference_ms = max(0.0, (callback_ns - submitted_ns) / 1_000_000)
        timestamp_ns = timestamp_ms * 1_000_000
        observations: list[HandObservation] = []
        output_frame = np.ascontiguousarray(_output_image.numpy_view())

        for index, hand_points in enumerate(result.hand_landmarks):
            categories = result.handedness[index] if index < len(result.handedness) else []
            category = categories[0] if categories else None
            handedness = category.category_name if category else ""
            handedness_score = float(category.score) if category else 0.0
            if index == 0:
                self._last_handedness = handedness
                self._last_handedness_score = handedness_score
                if self.config.partial_finger_tracking and len(hand_points) == 21:
                    self._partial_tracker.seed(
                        output_frame,
                        {
                            "Index": (
                                float(hand_points[8].x),
                                float(hand_points[8].y),
                            ),
                            "Middle": (
                                float(hand_points[12].x),
                                float(hand_points[12].y),
                            ),
                        },
                    )
            points = tuple(
                transform.restore(
                    Landmark(float(point.x), float(point.y), float(point.z)),
                )
                for point in hand_points
            )
            observations.append(
                HandObservation(
                    landmarks=points,
                    handedness=handedness,
                    timestamp_ns=timestamp_ns,
                    inference_ms=inference_ms,
                    handedness_score=handedness_score,
                ),
            )

        if not observations and self.config.partial_finger_tracking:
            tracked = self._partial_tracker.track(output_frame)
            if tracked:
                selected = next(
                    (finger for finger in tracked if finger.name == "Index"),
                    tracked[0],
                )
                restored = transform.restore(
                    Landmark(selected.x, selected.y, 0.0),
                )
                observations.append(
                    HandObservation(
                        landmarks=(),
                        handedness=self._last_handedness,
                        timestamp_ns=timestamp_ns,
                        inference_ms=inference_ms,
                        handedness_score=self._last_handedness_score,
                        partial_finger=selected.name,
                        partial_position=restored,
                    ),
                )
        elif not self.config.partial_finger_tracking:
            self._partial_tracker.reset()

        packet = DetectionPacket(
            observations=tuple(observations),
            timestamp_ns=timestamp_ns,
            inference_ms=inference_ms,
        )
        self.output_slot.put(packet)
        self.on_detection(packet)

    def run(self) -> None:
        if not self.model_path.exists():
            self.on_error(
                f"缺少手部模型：{self.model_path}。请先运行 scripts\\download_model.ps1。",
            )
            return

        # MediaPipe的原生文件打开逻辑在部分Windows环境中不能处理中文路径。
        # 由Python读取字节后传入模型缓冲，可同时兼容中文安装目录和打包目录。
        model_buffer = self.model_path.read_bytes()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_buffer),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=self.config.min_detection_confidence,
            min_hand_presence_confidence=self.config.min_presence_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
            result_callback=self._callback,
        )

        try:
            with vision.HandLandmarker.create_from_options(options) as landmarker:
                # 原生任务已完成模型复制后释放Python侧字节缓冲，降低常驻内存。
                del options
                del model_buffer
                revision = 0
                last_paused_submit_ns = 0
                while not self.stop_event.is_set():
                    packet, revision = self.input_slot.wait_next(revision, timeout=0.1)
                    if packet is None:
                        continue
                    if self.paused_provider():
                        if packet.captured_ns - last_paused_submit_ns < 100_000_000:
                            continue
                        last_paused_submit_ns = packet.captured_ns

                    timestamp_ms = packet.captured_ns // 1_000_000
                    timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
                    self._last_timestamp_ms = timestamp_ms
                    recognition_frame, transform = prepare_recognition_frame(
                        packet.frame,
                        self.config.recognition_zoom,
                    )
                    rgb = cv2.cvtColor(recognition_frame, cv2.COLOR_BGR2RGB)
                    rgb = np.ascontiguousarray(rgb)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    with self._submitted_lock:
                        self._submitted_at[timestamp_ms] = (
                            time.monotonic_ns(),
                            transform,
                        )
                    landmarker.detect_async(image, timestamp_ms)
        except Exception:
            LOGGER.exception("MediaPipe推理线程异常")
            self.on_error("手部模型初始化或推理失败，请查看日志。")
