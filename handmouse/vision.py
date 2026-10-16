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


def _hand_bbox(points) -> tuple[float, float, float, float]:
    """从 21 个手部关键点计算归一化 bbox (x_min, y_min, x_max, y_max)。"""

    xs = [p.x for p in points[:21]]
    ys = [p.y for p in points[:21]]
    return (min(xs), min(ys), max(xs), max(ys))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """两个归一化 bbox 的 IoU。"""

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


LOGGER = logging.getLogger(__name__)

# 进入门诊断开关：环境变量控制，跨模块生效。
VERBOSE_GATE = os.environ.get("HANDMOUSE_VERBOSE_GATE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_DIAG_LOGGER = logging.getLogger("handmouse.gate-diag")


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
        # 采集层诊断：上一帧内容采样与采集时刻，用于判"冻结 vs 慢活"和帧间隔。
        # 仅在 VERBOSE_GATE 下写入，生产路径零变化。
        prev_sample: tuple[float, float] | None = None
        prev_captured_ns: int | None = None
        try:
            while not self.stop_event.is_set():
                read_start_ns = time.monotonic_ns()
                ok, frame = capture.read()
                captured_ns = time.monotonic_ns()
                read_ms = (captured_ns - read_start_ns) / 1_000_000
                if not ok or frame is None:
                    consecutive_failures += 1
                    if VERBOSE_GATE:
                        # 无内容可采样，单独一行：ok/耗时/间隔打在同一条，用于区分
                        # "真失败（ok=False）"与"旧帧回放（ok=True 但内容不变）"。
                        gap_ms = (
                            (captured_ns - prev_captured_ns) / 1_000_000
                            if prev_captured_ns is not None
                            else -1.0
                        )
                        _DIAG_LOGGER.info(
                            "cap seq=%d read_ms=%.1f ok=False frame=None gap_ms=%.1f "
                            "captured_ns=%d failures=%d",
                            sequence, read_ms, gap_ms, captured_ns,
                            consecutive_failures,
                        )
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
                if VERBOSE_GATE:
                    # 采样交付帧（镜像+低光增强之后）的全局均值/标准差：
                    # 冻结帧逐字节相同 → 采样值完全相等、差值恒为 0；慢活帧即使
                    # 只有 1Hz 也会因传感器噪声抖动。用差值是否为 0 直接分开
                    # "旧帧回放"和"低帧率活采"，不需要全帧哈希。
                    small = cv2.resize(frame, (80, 60), interpolation=cv2.INTER_AREA)
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    mean_v, std_v = cv2.meanStdDev(gray)
                    sample = (float(mean_v[0][0]), float(std_v[0][0]))
                    prev = prev_sample
                    gap_ms = (
                        (captured_ns - prev_captured_ns) / 1_000_000
                        if prev_captured_ns is not None
                        else -1.0
                    )
                    _DIAG_LOGGER.info(
                        "cap seq=%d read_ms=%.1f ok=True frame=%dx%d "
                        "mean=%.4f std=%.4f dmean=%s dstd=%s dup=%s "
                        "brightness=%.2f gap_ms=%.1f captured_ns=%d",
                        sequence, read_ms, frame.shape[1], frame.shape[0],
                        sample[0], sample[1],
                        f"{sample[0] - prev[0]:+.5f}" if prev is not None else "n/a",
                        f"{sample[1] - prev[1]:+.5f}" if prev is not None else "n/a",
                        prev is not None and sample == prev,
                        brightness, gap_ms, captured_ns,
                    )
                    prev_sample = sample
                    prev_captured_ns = captured_ns
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
            max_forward_backward_error=2.5,
            max_lk_error=80.0,
            max_step_ratio=0.28,
        )
        self._last_handedness = ""
        self._last_handedness_score = 0.0
        # OpenCV Haar Cascade 脸检测器：用于挡掉 MediaPipe 把脸误判为手的情况。
        # 轻量、不需要额外模型文件，与 zoom 后的识别帧同坐标系。
        # 同时检测正面脸和侧脸，union 结果覆盖各种转头角度。
        cascade_dir = cv2.data.haarcascades
        self._face_cascades = [
            cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_frontalface_alt2.xml")),
            cv2.CascadeClassifier(os.path.join(cascade_dir, "haarcascade_profileface.xml")),
        ]
        self._face_cache: tuple[list[tuple[float, float, float, float]], int] = ([], 0)
        self._FACE_CACHE_NS = 500_000_000  # 500ms 内复用上次脸检测结果
        # 没检出脸时必须照样缓存：实测在室内光线下 Haar 几乎检不出这张脸
        # （r2 的 314 张取证帧只有 4 张命中），若空结果不入缓存，就会退化成
        # 每帧全量跑两个级联分类器，capture_fps 从 29.7 掉到 10。空结果有效期
        # 短一些，避免脸重新入画后长时间漏过滤。
        self._FACE_MISS_CACHE_NS = 200_000_000

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

        # 脸检测：如果有手检测结果才跑，带 500ms 缓存（脸不会跑得那么快）。
        # 坐标系与 HandLandmarker 一致（zoom 后的归一化坐标）。
        face_bboxes_norm: list[tuple[float, float, float, float]] = []
        if result.hand_landmarks:
            cached_faces, cached_ns = self._face_cache
            cache_ttl = self._FACE_CACHE_NS if cached_faces else self._FACE_MISS_CACHE_NS
            if callback_ns - cached_ns < cache_ttl:
                face_bboxes_norm = cached_faces
            else:
                gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
                h, w = output_frame.shape[:2]
                all_faces: list[tuple[int, int, int, int]] = []
                for cascade in self._face_cascades:
                    try:
                        faces = cascade.detectMultiScale(
                            gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30),
                        )
                    except Exception:
                        faces = ()
                    all_faces.extend(faces)
                face_bboxes_norm = [
                    (fx / w, fy / h, (fx + fw) / w, (fy + fh) / h)
                    for (fx, fy, fw, fh) in all_faces
                ]
                self._face_cache = (face_bboxes_norm, callback_ns)

        for index, hand_points in enumerate(result.hand_landmarks):
            categories = result.handedness[index] if index < len(result.handedness) else []
            category = categories[0] if categories else None
            handedness = category.category_name if category else ""
            handedness_score = float(category.score) if category else 0.0

            # 脸 bbox overlap 过滤：先用 HandLandmarker 原始坐标（zoom 后）
            # 算 bbox，和 Haar Cascade 的脸 bbox 比 IoU。重叠 > 50% 才丢弃。
            if len(hand_points) == 21 and face_bboxes_norm:
                hand_bbox = _hand_bbox(hand_points)
                max_iou = max((_bbox_iou(hand_bbox, fb) for fb in face_bboxes_norm), default=0.0)
                if max_iou > 0.50:
                    LOGGER.info(
                        "[脸过滤] 丢弃疑似误检 IoU=%.2f handedness=%s score=%.2f",
                        max_iou, handedness, handedness_score,
                    )
                    continue

            points = tuple(
                transform.restore(
                    Landmark(float(point.x), float(point.y), float(point.z)),
                )
                for point in hand_points
            )
            # 通过脸过滤后才更新 handedness 和 seed 续跟踪器，避免假阳性污染状态。
            if len(observations) == 0:
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

        # -------- 进入门诊断：MediaPipe 原始输出层 --------
        # 限流：每帧打摘要；hand_count 或 handedness 值变化时追加完整明细。
        if VERBOSE_GATE:
            raw_hand_count = len(result.hand_landmarks)
            # 显式展开 handedness_raw：每手取 top classification 的 (label, score)
            raw_handedness: list[tuple[str, float]] = []
            for hc in result.handedness[:raw_hand_count]:
                if hc:
                    top = hc[0]
                    raw_handedness.append((top.category_name, float(top.score)))
                else:
                    raw_handedness.append(("", 0.0))
            raw_sig = (raw_hand_count, tuple(raw_handedness))
            ts_ns = callback_ns

            # 每帧摘要（轻量，无变化检测）
            _DIAG_LOGGER.info(
                "raw ts=%d hand_count=%d handedness_raw=%s observations_out=%d face_filtered=%s",
                ts_ns, raw_hand_count,
                [(label, f"{score:.2f}") for label, score in raw_handedness],
                len(observations),
                len(observations) < raw_hand_count,
            )

            # 值变化时追加完整明细（去重用 signature）
            if raw_sig != getattr(self, "_gate_diag_last_raw_sig", None):
                self._gate_diag_last_raw_sig = raw_sig
                for i, obs in enumerate(observations):
                    landmarks_len = (
                        len(obs.landmarks)
                        if obs.landmarks
                        else ("partial" if obs.partial_finger else 0)
                    )
                    _DIAG_LOGGER.info(
                        "raw-detail ts=%d hand[%d] landmarks_len=%s handedness=%s score=%.2f "
                        "partial_finger=%s",
                        ts_ns, i, landmarks_len,
                        obs.handedness or '""', obs.handedness_score,
                        obs.partial_finger or "None",
                    )

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
