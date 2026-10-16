import threading
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from handmouse.config import AppConfig
from handmouse.contracts import Landmark
from handmouse.latest_slot import LatestSlot
from handmouse.vision import (
    CameraWorker,
    HandLandmarkerWorker,
    RecognitionTransform,
    _thumb_axis_cross,
    prepare_recognition_frame,
)


def test_low_light_enhancement_increases_brightness_without_clipping_everything() -> None:
    frame = np.full((60, 80, 3), 35, dtype=np.uint8)

    enhanced = CameraWorker._enhance_low_light(frame, 35.0)

    assert enhanced.mean() > frame.mean() + 15
    assert enhanced.max() < 200


def test_normal_light_frame_is_not_modified() -> None:
    frame = np.full((10, 10, 3), 90, dtype=np.uint8)

    assert CameraWorker._enhance_low_light(frame, 90.0) is frame


def test_recognition_zoom_is_reversible_for_landmark_coordinates() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    prepared, transform = prepare_recognition_frame(frame, 1.20)
    restored = transform.restore(Landmark(0.25, 0.75, -0.10))

    assert prepared.shape == frame.shape
    assert 0.25 < restored.x < 0.30
    assert 0.70 < restored.y < 0.75
    assert -0.10 < restored.z < -0.07


def test_recognition_zoom_one_returns_identity_without_copy() -> None:
    frame = np.zeros((10, 20, 3), dtype=np.uint8)

    prepared, transform = prepare_recognition_frame(frame, 1.0)

    assert prepared is frame
    assert transform == RecognitionTransform()


class _FakeImage:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def numpy_view(self) -> np.ndarray:
        return self.frame


def _tracking_frame() -> np.ndarray:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    for x, color in ((30, (255, 255, 255)), (90, (180, 255, 120))):
        cv2.rectangle(frame, (x - 7, 33), (x + 7, 47), color, 2)
        cv2.line(frame, (x - 6, 34), (x + 6, 46), color, 2)
        cv2.line(frame, (x + 6, 34), (x - 6, 46), color, 2)
    return frame


def _shift(frame: np.ndarray, dx: int) -> np.ndarray:
    transform = np.float32(((1, 0, dx), (0, 1, 0)))
    return cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))


def test_landmarker_callback_keeps_middle_identity_well_beyond_eight_frames() -> None:
    detections = []
    worker = HandLandmarkerWorker(
        AppConfig(partial_finger_tracking=True, partial_tracking_frames=0),
        Path("unused.task"),
        LatestSlot(),
        LatestSlot(),
        threading.Event(),
        lambda _message: None,
        detections.append,
        lambda: False,
    )
    # 构造一个几何上合理的手骨架（所有 21 个点不能堆在同一处，否则
    # 会被 _is_plausible_hand 过滤掉）。
    # 手腕在下方，指关节向上扇形展开。
    wrist = SimpleNamespace(x=0.50, y=0.80, z=0.0)
    index_mcp = SimpleNamespace(x=0.40, y=0.62, z=0.0)
    index_pip = SimpleNamespace(x=0.38, y=0.48, z=0.0)
    index_dip = SimpleNamespace(x=0.36, y=0.38, z=0.0)
    index_tip = SimpleNamespace(x=30 / 120, y=40 / 80, z=0.0)  # 保留测试原意
    middle_mcp = SimpleNamespace(x=0.50, y=0.60, z=0.0)
    middle_pip = SimpleNamespace(x=0.50, y=0.45, z=0.0)
    middle_dip = SimpleNamespace(x=0.50, y=0.35, z=0.0)
    middle_tip = SimpleNamespace(x=90 / 120, y=40 / 80, z=0.0)  # 保留测试原意
    ring_mcp = SimpleNamespace(x=0.60, y=0.62, z=0.0)
    ring_pip = SimpleNamespace(x=0.62, y=0.50, z=0.0)
    ring_dip = SimpleNamespace(x=0.63, y=0.40, z=0.0)
    ring_tip = SimpleNamespace(x=0.63, y=0.32, z=0.0)
    pinky_mcp = SimpleNamespace(x=0.70, y=0.66, z=0.0)
    pinky_pip = SimpleNamespace(x=0.73, y=0.56, z=0.0)
    pinky_dip = SimpleNamespace(x=0.75, y=0.48, z=0.0)
    pinky_tip = SimpleNamespace(x=0.76, y=0.42, z=0.0)
    thumb_cmc = SimpleNamespace(x=0.30, y=0.68, z=0.0)
    thumb_mcp = SimpleNamespace(x=0.25, y=0.58, z=0.0)
    thumb_ip = SimpleNamespace(x=0.22, y=0.50, z=0.0)
    thumb_tip = SimpleNamespace(x=0.20, y=0.44, z=0.0)
    points = [
        wrist,           # 0
        thumb_cmc,       # 1
        thumb_mcp,       # 2
        thumb_ip,        # 3
        thumb_tip,       # 4
        index_mcp,       # 5
        index_pip,       # 6
        index_dip,       # 7
        index_tip,       # 8
        middle_mcp,      # 9
        middle_pip,      # 10
        middle_dip,      # 11
        middle_tip,      # 12
        ring_mcp,        # 13
        ring_pip,        # 14
        ring_dip,        # 15
        ring_tip,        # 16
        pinky_mcp,       # 17
        pinky_pip,       # 18
        pinky_dip,       # 19
        pinky_tip,       # 20
    ]
    category = SimpleNamespace(category_name="Right", score=0.98)
    full_result = SimpleNamespace(
        hand_landmarks=[points],
        handedness=[[category]],
    )
    frame = _tracking_frame()
    worker._submitted_at[1] = (time.monotonic_ns(), RecognitionTransform())
    worker._callback(full_result, _FakeImage(frame), 1)

    empty_result = SimpleNamespace(hand_landmarks=[], handedness=[])
    for timestamp, shift in enumerate(range(-2, -62, -2), start=2):
        worker._submitted_at[timestamp] = (
            time.monotonic_ns(),
            RecognitionTransform(),
        )
        worker._callback(
            empty_result,
            _FakeImage(_shift(frame, shift)),
            timestamp,
        )

    observation = detections[-1].observations[0]
    assert observation.partial_finger == "Middle"
    assert observation.partial_position is not None


# ---------------------------------------------------------------------------
# 摄像头掉线自动重连（拔插属于外设事件，不能终止整条流水线）
# ---------------------------------------------------------------------------


class _ScriptedCapture:
    """按脚本吐出 read 结果，用于模拟掉线与重连。"""

    def __init__(self, opened: bool, reads: list[bool]) -> None:
        self._opened = opened
        self._reads = list(reads)
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def set(self, *_args: object) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok = self._reads.pop(0) if self._reads else False
        if not ok:
            return False, None
        return True, np.zeros((8, 8, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def test_camera_worker_reconnects_after_disconnect() -> None:
    # 第一次打开成功但连续读失败 10 次触发重连 → 重开失败一次 → 再开成功。
    first = _ScriptedCapture(True, [False] * 10 + [True])
    failed_reopen = _ScriptedCapture(False, [])
    recovered = _ScriptedCapture(True, [True])
    captures = iter([first, failed_reopen, recovered])

    frames = []
    warnings: list[str] = []
    errors: list[str] = []
    stop_event = threading.Event()

    def on_frame(packet: object) -> None:
        frames.append(packet)
        stop_event.set()

    worker = CameraWorker(
        AppConfig(),
        LatestSlot(),
        stop_event,
        errors.append,
        on_frame,
        warnings.append,
    )
    worker._open_camera = lambda: next(captures)  # type: ignore[method-assign]
    worker.run()

    # 掉线绝不能走 on_error —— 那会停掉整条流水线，插回去也回不来。
    assert errors == []
    assert frames
    assert first.released
    assert failed_reopen.released
    assert any("摄像头已断开" in message for message in warnings)
    assert any("摄像头已恢复。" == message for message in warnings)


def _hand_with_axes(k4, k5, k17) -> tuple[Landmark, ...]:
    points = [Landmark(0.0, 0.0, 0.0)] * 21
    points[4] = Landmark(*k4, 0.0)
    points[5] = Landmark(*k5, 0.0)
    points[17] = Landmark(*k17, 0.0)
    return tuple(points)


def test_thumb_axis_cross_sign_for_known_configuration() -> None:
    # 掌轴 5→17 指向左下、拇指 5→4 指向右上：叉积为正。
    points = _hand_with_axes(k4=(0.6, 0.3), k5=(0.5, 0.5), k17=(0.3, 0.6))

    cross, len_across, len_thumb = _thumb_axis_cross(points)

    assert np.isclose(cross, 0.03)
    assert len_across > 0 and len_thumb > 0


def test_thumb_axis_cross_flips_sign_under_horizontal_mirror() -> None:
    # flip 语义取证的数学前提：像素水平镜像（x→1-x）只反相叉积符号，
    # 不改幅度。若模型看到的是 flip 后像素，叉积方向量必然整体反号。
    points = _hand_with_axes(k4=(0.6, 0.3), k5=(0.5, 0.5), k17=(0.3, 0.6))
    mirrored = tuple(Landmark(1.0 - p.x, p.y, p.z) for p in points)

    cross, _, _ = _thumb_axis_cross(points)
    cross_m, _, _ = _thumb_axis_cross(mirrored)

    assert np.isclose(cross_m, -cross)
