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

from handmouse.config import PROJECT_ROOT, AppConfig
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
# 镜像语义取证开关：成对落盘"模型实际输入帧 / 等价 flip 前帧"+ handedness，
# 三值用同一 timestamp_ms 对齐，用于判定叉积 expected_sign 约定，默认关闭。
FLIP_DIAG = os.environ.get("HANDMOUSE_FLIP_DIAG", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_DIAG_LOGGER = logging.getLogger("handmouse.gate-diag")


def _thumb_axis_cross(points) -> tuple[float, float, float]:
    """掌轴 5→17 与拇指矢量 5→4 的二维叉积，及两轴长度（诊断用）。

    叉积符号是 ``gestures._palm_presented`` 判掌/背的唯一方向量；单独抽出
    供取证快照与单测共用，保证诊断算的就是判定层算的。
    """

    ax = points[17].x - points[5].x
    ay = points[17].y - points[5].y
    bx = points[4].x - points[5].x
    by = points[4].y - points[5].y
    return ax * by - ay * bx, float(np.hypot(ax, ay)), float(np.hypot(bx, by))


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
    max_width: int = 0,
) -> tuple[np.ndarray, RecognitionTransform]:
    """温和放大中央识别区，可选再把推理分辨率压下来，并返回可逆坐标变换。

    这不是跟踪ROI：裁剪区固定居中、不会追随手部移动。默认1.15倍只裁掉
    四周约6.5%，能提高远手和小手在模型输入中的有效像素，同时保留大部分
    画面。设置为1.0时零拷贝返回原帧。

    ``max_width`` 用于给 MediaPipe 的输入限宽（0 表示不限制）。实测在 CPU 上
    640×480 只能跑 7.4fps，降到 320×240 可翻倍以上。**归一化坐标与输出分辨率
    无关**，只要保持宽高比，``RecognitionTransform.restore()`` 仍然正确。
    """

    zoom = max(1.0, min(1.35, float(zoom)))
    height, width = frame.shape[:2]

    if zoom <= 1.001:
        left = top = 0
        crop_width, crop_height = width, height
        crop = frame
    else:
        crop_width = max(2, round(width / zoom))
        crop_height = max(2, round(height / zoom))
        left = max(0, (width - crop_width) // 2)
        top = max(0, (height - crop_height) // 2)
        crop = frame[top : top + crop_height, left : left + crop_width]

    # 默认放大回原尺寸（这是 zoom 的原语义）；只有显式限制宽度时才缩小。
    out_width, out_height = width, height
    limit = int(max_width or 0)
    if limit and crop_width > limit:
        ratio = limit / crop_width
        out_width = limit
        out_height = max(2, round(crop_height * ratio))

    if out_width == crop_width and out_height == crop_height:
        resized = crop
    else:
        resized = cv2.resize(
            crop, (out_width, out_height), interpolation=cv2.INTER_LINEAR,
        )

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

    #: 配置的分辨率拿不到时依次尝试的档位（通常只在驱动回报垃圾尺寸时才用得上）。
    #: 🔴 顺序是 **4:3 优先**：本项目的几何阈值全部在 4:3 下标定，4:3 能拿到就
    #: 不要退到 16:9——虽然手势引擎有宽高比补偿，但少一次补偿就少一处变量。
    #: 从大到小排，够清晰又不至于把 CPU 吃满。
    _FALLBACK_RESOLUTIONS: tuple[tuple[int, int], ...] = (
        (640, 480),
        (1280, 720),
        (960, 540),
        (800, 600),
        (640, 360),
        (320, 240),
        (160, 120),
    )

    def __init__(
        self,
        config: AppConfig,
        output: LatestSlot[FramePacket],
        stop_event: threading.Event,
        on_error: Callable[[str], None],
        on_frame: Callable[[FramePacket], None],
        on_warning: Callable[[str], None] | None = None,
        on_diag: Callable[[str], None] | None = None,
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
        # 诊断事件回调（如 suspicious_empty），供 HUD debug 层读取。
        self.on_diag = on_diag or (lambda _kind: None)
        #: 摄像头**实际**输出的尺寸与帧率（协商后读回，见 `_negotiate_format`）。
        #: 配置值只是"请求"，驱动可以拒绝或静默降级，所以真实值必须记下来。
        self.actual_size: tuple[int, int] = (
            int(config.camera_width),
            int(config.camera_height),
        )
        self.actual_fps: float = float(config.camera_fps)

    def _resolve_backend(self) -> int:
        """把配置里的后端名换成 OpenCV 的常量。

        🔴 不做自动探测：同一条命令两次启动必须走同一条路径，否则"这次好使
        下次不好使"根本没法复现。后端选错时的表现是**帧率不对**（不是报错），
        所以由用户显式指定，并在日志里把实际用的后端打出来。
        """

        if os.name != "nt":
            return cv2.CAP_ANY
        name = str(getattr(self.config, "camera_backend", "dshow") or "").strip().lower()
        return {
            "msmf": cv2.CAP_MSMF,
            "any": cv2.CAP_ANY,
        }.get(name, cv2.CAP_DSHOW)

    def _open_camera(self) -> cv2.VideoCapture:
        backend = self._resolve_backend()
        capture = cv2.VideoCapture(self.config.camera_index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            # 配置的后端打不开时退回 OpenCV 默认后端：能开总比开不了强。
            capture.release()
            capture = cv2.VideoCapture(self.config.camera_index)
        if os.name == "nt":
            capture.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*"MJPG"),
            )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._negotiate_format(capture)
        return capture

    @staticmethod
    def _sane_size(width: float, height: float) -> bool:
        """驱动回报的尺寸能不能用。

        有些驱动对不支持的档位返回 0×0 或 1×1（而不是拒绝 set），
        那种值必须当成"没拿到"，否则画面会被裁成一条线。
        """

        return width >= 160 and height >= 120

    def _resolution_candidates(self) -> list[tuple[int, int]]:
        wanted = (int(self.config.camera_width), int(self.config.camera_height))
        if not bool(getattr(self.config, "camera_allow_fallback", True)):
            return [wanted]
        candidates = [wanted]
        for size in self._FALLBACK_RESOLUTIONS:
            if size not in candidates:
                candidates.append(size)
        return candidates

    def _negotiate_format(self, capture: cv2.VideoCapture) -> None:
        """设置分辨率/帧率，并**读回实际值**——摄像头要是不支持，如实说出来。

        🔴 为什么必须读回（2026-10-23 实测）：这只内置摄像头请求 1080p 会被
        **静默降级**回 720p，`set()` 照样返回 True。不读回就永远以为自己在
        640×480 上跑，而实际画面是 16:9 —— 所有"距离 ÷ 掌宽"的比值阈值会
        整体漂 19%~32%（见 `_aspect_scale`）。

        策略：
        1. 先按配置值设置，读回能用就**直接用**（哪怕摄像头给的是别的档位——
           手势引擎有宽高比补偿，任何尺寸都能跑）；
        2. 只有读回是 0×0 这种垃圾值时，才按候选表逐个重试；
        3. 所有情况都记日志，与配置不一致时给用户一句可执行的提示。
        """

        requested = (int(self.config.camera_width), int(self.config.camera_height))
        best: tuple[int, int] | None = None
        for width, height in self._resolution_candidates():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            got_w = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            got_h = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if self._sane_size(got_w, got_h):
                best = (int(round(got_w)), int(round(got_h)))
                break
        if best is None:
            best = requested

        self.actual_size = best
        capture.set(cv2.CAP_PROP_FPS, self.config.camera_fps)
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.actual_fps = actual_fps if actual_fps > 0.0 else 0.0

        aspect = best[0] / best[1] if best[1] else 0.0
        LOGGER.info(
            "摄像头 %s 实际输出 %d×%d @ %.1ffps（配置 %d×%d@%dfps，宽高比 %.3f，后端 %s）",
            self.config.camera_index,
            best[0], best[1], self.actual_fps,
            requested[0], requested[1], self.config.camera_fps,
            aspect,
            getattr(self.config, "camera_backend", "dshow"),
        )
        if best != requested:
            self.on_warning(
                f"摄像头不支持 {requested[0]}×{requested[1]}，已自动改用 "
                f"{best[0]}×{best[1]}。"
                f"识别与手势判据会自动适配（宽高比 {aspect:.2f}），无需手动改配置。",
            )
        if self.actual_fps and self.actual_fps < self.config.camera_fps * 0.6:
            self.on_warning(
                f"摄像头实际帧率 {self.actual_fps:.0f}fps 明显低于配置的 "
                f"{self.config.camera_fps}fps，跟随手感会发涩。"
                f"可在 config.json 里把 camera_fps 调成实际值，或换 USB3 接口/更好的摄像头。",
            )

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
                if VERBOSE_GATE:
                    # ok=True 但内容近全黑/全灰（驱动交空缓冲）：只记日志不改判定，
                    # 不累加 consecutive_failures——是否计入是另一笔的事。
                    # 采样原始帧（flip/低光增强之前），观测的是驱动交付物本身。
                    # raw_std < 1.0 是初值：纯黑/纯灰 std≈0，带噪驱动交付物通常 std > 2。
                    # 观测后若从未触发，说明该故障模式在本机不存在，可删此段；
                    # 若触发但伴随明显可见画面，说明阈值偏松，需下调。
                    raw_std = float(frame[::16, ::16].std())
                    if raw_std < 1.0:
                        _DIAG_LOGGER.info(
                            "[read] suspicious empty frame seq=%d std=%.3f read_ms=%.1f",
                            sequence, raw_std, read_ms,
                        )
                        self.on_diag("suspicious_empty")
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
        # 镜像语义取证（HANDMOUSE_FLIP_DIAG=1）：默认目录为 None，生产零开销。
        self._flip_diag_dir: Path | None = None
        self._flip_diag_last_ns = 0
        self._flip_diag_count = 0
        if FLIP_DIAG:
            diag_dir = PROJECT_ROOT / "logs" / (
                f"flip-diag-{time.strftime('%Y%m%d-%H%M%S')}"
            )
            diag_dir.mkdir(parents=True, exist_ok=True)
            self._flip_diag_dir = diag_dir
            print(f"[flip-diag] 镜像语义取证已开启：{diag_dir}")
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

    # flip 语义取证节流：1.0s 一对、最多 40 对，一次运行覆盖三段姿势。
    _FLIP_DIAG_INTERVAL_NS = 1_000_000_000
    _FLIP_DIAG_MAX = 40

    def _flip_diag_dump(
        self,
        model_rgb: np.ndarray,
        raw_points,
        restored_points,
        handedness: str,
        score: float,
        timestamp_ms: int,
    ) -> None:
        """成对落盘模型实际输入帧与等价 flip 前帧，三值按 ts 对齐。

        flip 自逆（``flip(flip(x)) == x``），zoom/裁剪只平移缩放不改左右，
        故把模型输入帧再 flip 一次即等价于采集线程 flip 前的像素，无需跨
        线程回传采集帧。post 帧上画掌轴 5→17（红）与拇指矢量 5→4（青）。
        """

        if self._flip_diag_dir is None:
            return
        now_ns = time.monotonic_ns()
        if self._flip_diag_count >= self._FLIP_DIAG_MAX:
            return
        if now_ns - self._flip_diag_last_ns < self._FLIP_DIAG_INTERVAL_NS:
            return
        self._flip_diag_last_ns = now_ns
        self._flip_diag_count += 1

        post_bgr = cv2.cvtColor(model_rgb, cv2.COLOR_RGB2BGR)
        pre_bgr = cv2.flip(post_bgr, 1)
        height, width = post_bgr.shape[:2]
        label_line = (
            f"ts={timestamp_ms} mirror={self.config.mirror} "
            f"MP_label={handedness}({score:.2f})"
        )
        cv2.putText(
            post_bgr, "MODEL INPUT (post-flip)", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.putText(
            pre_bgr, "EQUIV PRE-FLIP (sensor view)", (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        for img in (post_bgr, pre_bgr):
            cv2.putText(
                img, label_line, (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )
            cv2.putText(
                img, "<- frame left", (8, height - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 120), 2,
            )

        def _px(point) -> tuple[int, int]:
            return int(point.x * width), int(point.y * height)

        p4 = _px(raw_points[4])
        p5 = _px(raw_points[5])
        p17 = _px(raw_points[17])
        cv2.arrowedLine(post_bgr, p5, p17, (0, 0, 255), 2)
        cv2.arrowedLine(post_bgr, p5, p4, (255, 128, 0), 2)
        for point, text, color in (
            (p5, "5", (0, 255, 255)),
            (p17, "17", (0, 0, 255)),
            (p4, "4", (255, 128, 0)),
        ):
            cv2.putText(
                post_bgr, text, point,
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
            )

        cross_raw, len_across, len_thumb = _thumb_axis_cross(raw_points)
        cross_restored, _, _ = _thumb_axis_cross(restored_points)
        sin_theta = (
            abs(cross_raw) / (len_across * len_thumb)
            if len_across > 1e-6 and len_thumb > 1e-6
            else 0.0
        )
        tag = f"{self._flip_diag_count:02d}-{timestamp_ms}"
        # cv2.imwrite 在 Windows 中文路径下静默失败，改 imencode+tofile。
        for name, img in (f"post-{tag}.png", post_bgr), (f"pre-{tag}.png", pre_bgr):
            ok, buffer = cv2.imencode(".png", img)
            if ok:
                buffer.tofile(str(self._flip_diag_dir / name))
        print(
            f"[flip-diag] #{self._flip_diag_count:02d} ts={timestamp_ms} "
            f"mirror={self.config.mirror} label={handedness} score={score:.2f} "
            f"cross_raw={cross_raw:+.6f} cross_restored={cross_restored:+.6f} "
            f"sin={sin_theta:.3f}"
        )

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
        # 画面宽高比：手势引擎用它把"距离 ÷ 掌宽"这类比值还原到 4:3 标定尺度。
        # 必须取**识别输入图**的尺寸（zoom 裁剪与 inference_width 限宽都保持
        # 宽高比，所以它等于摄像头画面的宽高比）。
        frame_h, frame_w = output_frame.shape[:2]
        frame_aspect = (float(frame_w) / float(frame_h)) if frame_h and frame_w else 4.0 / 3.0

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
            # 米制 3D 坐标（原点在手部几何中心）。**不做 restore**：它不在
            # 归一化图像坐标系里，裁剪/缩放变换对它没有意义。
            # 这是「手指本体」的真实三维骨架，指节夹角和长度比都不受画面
            # 宽高比与透视压缩影响，是判断"手指伸没伸直"唯一正确的输入。
            world_points: tuple[Landmark, ...] = ()
            # ``getattr`` 不是防御性代码洁癖：旧版 mediapipe 的 Result 类没有
            # 这个字段，测试里的假 Result 也没有。缺了就退化成空元组，
            # 由 2D 判据兜底，不要因此让整条流水线炸掉。
            world_all = getattr(result, "hand_world_landmarks", None) or ()
            if index < len(world_all) and world_all[index]:
                world_points = tuple(
                    Landmark(float(point.x), float(point.y), float(point.z))
                    for point in world_all[index]
                )
            # 通过脸过滤后才更新 handedness 和 seed 续跟踪器，避免假阳性污染状态。
            if len(observations) == 0:
                self._last_handedness = handedness
                self._last_handedness_score = handedness_score
                if len(hand_points) == 21:
                    self._flip_diag_dump(
                        output_frame,
                        hand_points,
                        points,
                        handedness,
                        handedness_score,
                        timestamp_ms,
                    )
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
                    world_landmarks=world_points,
                    image_aspect=frame_aspect,
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
                        image_aspect=frame_aspect,
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
                        self.config.inference_width,
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
