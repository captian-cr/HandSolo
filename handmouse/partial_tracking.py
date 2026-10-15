"""完整手丢失后的食指/中指身份续跟踪。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class TrackedFinger:
    """带有继承身份的归一化单指位置。"""

    name: str
    x: float
    y: float


class PartialFingerTracker:
    """用前后向Lucas–Kanade光流继承指尖身份。

    身份必须由最近一次完整21点检测提供。该类不会尝试从一截孤立手指
    的外观猜测解剖身份，从而避免把不确定结果伪装成可靠识别。默认只
    由光流质量决定跟踪寿命； ``max_frames`` 大于0时才启用帧数上限。
    """

    def __init__(
        self,
        max_frames: int = 0,
        max_forward_backward_error: float = 2.0,
        max_lk_error: float = 60.0,
        max_step_ratio: float = 0.18,
    ) -> None:
        self.max_frames = max(0, int(max_frames))
        self.max_forward_backward_error = max_forward_backward_error
        self.max_lk_error = max_lk_error
        self.max_step_ratio = max(0.02, float(max_step_ratio))
        self._previous_gray: np.ndarray | None = None
        self._points: np.ndarray | None = None
        self._names: list[str] = []
        self._tracked_frames = 0

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return np.ascontiguousarray(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def reset(self) -> None:
        self._previous_gray = None
        self._points = None
        self._names = []
        self._tracked_frames = 0

    def seed(
        self,
        frame: np.ndarray,
        fingers: dict[str, tuple[float, float]],
    ) -> None:
        """用完整手结果设置食指和中指的已知身份位置。"""

        height, width = frame.shape[:2]
        names: list[str] = []
        points: list[tuple[float, float]] = []
        for name in ("Index", "Middle"):
            position = fingers.get(name)
            if position is None:
                continue
            x = float(position[0]) * width
            y = float(position[1]) * height
            if 1.0 <= x < width - 1.0 and 1.0 <= y < height - 1.0:
                names.append(name)
                points.append((x, y))
        if not points:
            self.reset()
            return
        self._previous_gray = self._gray(frame).copy()
        self._points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        self._names = names
        self._tracked_frames = 0

    def track(self, frame: np.ndarray) -> tuple[TrackedFinger, ...]:
        if (
            self._previous_gray is None
            or self._points is None
            or not self._names
            or (self.max_frames > 0 and self._tracked_frames >= self.max_frames)
        ):
            self.reset()
            return ()

        current_gray = self._gray(frame)
        if current_gray.shape != self._previous_gray.shape:
            self.reset()
            return ()

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            0.01,
        )
        next_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
            self._previous_gray,
            current_gray,
            self._points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if next_points is None or forward_status is None:
            self.reset()
            return ()
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            current_gray,
            self._previous_gray,
            next_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=criteria,
        )
        if back_points is None or backward_status is None:
            self.reset()
            return ()

        height, width = current_gray.shape[:2]
        valid_names: list[str] = []
        valid_points: list[tuple[float, float]] = []
        output: list[TrackedFinger] = []
        for index, name in enumerate(self._names):
            x, y = (float(value) for value in next_points[index, 0])
            previous_x, previous_y = (
                float(value) for value in self._points[index, 0]
            )
            forward_ok = bool(forward_status[index, 0])
            backward_ok = bool(backward_status[index, 0])
            lk_error = (
                float(forward_error[index, 0])
                if forward_error is not None
                else 0.0
            )
            fb_error = float(
                np.linalg.norm(
                    back_points[index, 0] - self._points[index, 0],
                ),
            )
            inside = 2.0 <= x < width - 2.0 and 2.0 <= y < height - 2.0
            step_ratio = float(
                np.hypot(x - previous_x, y - previous_y)
                / np.hypot(width, height),
            )
            if not (
                forward_ok
                and backward_ok
                and inside
                and lk_error <= self.max_lk_error
                and fb_error <= self.max_forward_backward_error
                and step_ratio <= self.max_step_ratio
            ):
                continue
            valid_names.append(name)
            valid_points.append((x, y))
            output.append(TrackedFinger(name, x / width, y / height))

        if not valid_points:
            self.reset()
            return ()

        self._previous_gray = current_gray.copy()
        self._points = np.asarray(valid_points, dtype=np.float32).reshape(-1, 1, 2)
        self._names = valid_names
        self._tracked_frames += 1
        return tuple(output)
