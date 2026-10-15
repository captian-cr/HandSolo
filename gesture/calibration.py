"""引导式校准：把食指尖依次停到画面中的三个角点上，静止约 0.4 秒自动采样。

为什么在画面里而不是桌面上做校准：
桌面标记要用透明覆盖层才能画，第一阶段还没做覆盖层；
把标记画在摄像头预览里，用户看着预览把自己的手移到红点上，效果一样，
而且顺带把"手的舒适活动范围"一并框定了。
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .config import Config
from .hud import DEFAULT_GRAY, DEFAULT_GREEN, DEFAULT_RED, DEFAULT_WHITE, DEFAULT_YELLOW
from .hud import draw_panel, draw_texts, estimate_width

TARGET_NAMES = ("左上", "右上", "左下")


class CalibrationSession:
    FRAMES_NEEDED = 12  # 30fps 下约 0.4 秒
    CAPTURE_RADIUS = 24.0  # 指尖必须落在目标点多近
    STABILITY_RADIUS = 14.0  # 窗口内所有采样点的离散程度上限
    UNLOCK_RADIUS = 40.0  # 记录后必须先离开这么远，才允许采下一个点

    def __init__(self, cfg: Config, frame_size) -> None:
        self.cfg = cfg
        self.frame_w, self.frame_h = int(frame_size[0]), int(frame_size[1])

        margin_x = self.frame_w * cfg.calib_margin_x
        margin_y = self.frame_h * cfg.calib_margin_y
        x0, y0 = margin_x, margin_y
        x1, y1 = self.frame_w - margin_x, self.frame_h - margin_y

        self.rect = (float(x0), float(y0), float(x1), float(y1))
        self.corners = np.array([[x0, y0], [x1, y0], [x0, y1]], dtype=np.float32)
        self.samples: list[np.ndarray] = []
        self.step = 0
        self.progress = 0.0

        self._window: deque = deque(maxlen=self.FRAMES_NEEDED)
        self._locked_point: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self.step >= len(TARGET_NAMES)

    def reset(self) -> None:
        self.samples.clear()
        self.step = 0
        self._window.clear()
        self._locked_point = None
        self.progress = 0.0

    def back(self) -> None:
        if self.step == 0:
            return
        self.step -= 1
        if self.samples:
            self.samples.pop()
        self._window.clear()
        self._locked_point = None
        self.progress = 0.0

    def update(self, point) -> bool:
        """point 是滤波后的相机像素坐标（None 表示当前没检测到手）。返回 True 表示刚记录一个点。"""
        if self.done:
            return False

        if point is None:
            self._window.clear()
            self.progress = 0.0
            return False

        p = np.asarray(point, dtype=float)

        if self._locked_point is not None and float(np.linalg.norm(p - self._locked_point)) < self.UNLOCK_RADIUS:
            self._window.clear()
            self.progress = 0.0
            return False

        target = self.corners[self.step]
        self._window.append(p)

        distances = np.linalg.norm(np.asarray(self._window) - target, axis=1)
        self.progress = float(np.count_nonzero(distances <= self.CAPTURE_RADIUS)) / self.FRAMES_NEEDED

        if len(self._window) < self.FRAMES_NEEDED:
            return False
        if float(distances.max()) > self.CAPTURE_RADIUS:
            return False

        window_points = np.asarray(self._window)
        spread = float(np.max(np.linalg.norm(window_points - window_points.mean(axis=0), axis=1)))
        if spread > self.STABILITY_RADIUS:
            return False

        self.samples.append(p.copy())
        self._locked_point = p.copy()
        self.step += 1
        self._window.clear()
        self.progress = 0.0
        return True

    def draw(self, frame_bgr: np.ndarray, point, fps: float | None = None) -> None:
        x0, y0, x1, y1 = (int(round(v)) for v in self.rect)
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (110, 110, 110), 1, cv2.LINE_AA)

        for index, corner in enumerate(self.corners):
            center = tuple(np.round(corner).astype(int))
            if index < self.step:
                cv2.circle(frame_bgr, center, 7, (0, 220, 0), -1, cv2.LINE_AA)
            elif index == self.step:
                cv2.circle(frame_bgr, center, int(self.CAPTURE_RADIUS), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.line(frame_bgr, (center[0] - 13, center[1]), (center[0] + 13, center[1]), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.line(frame_bgr, (center[0], center[1] - 13), (center[0], center[1] + 13), (0, 0, 255), 1, cv2.LINE_AA)
                if self.progress > 0:
                    cv2.ellipse(
                        frame_bgr,
                        center,
                        (34, 34),
                        0,
                        -90,
                        -90 + int(360 * self.progress),
                        (0, 215, 255),
                        3,
                        cv2.LINE_AA,
                    )
            else:
                cv2.circle(frame_bgr, center, 5, (110, 110, 110), 1, cv2.LINE_AA)

        if point is not None:
            tip = tuple(np.round(np.asarray(point)).astype(int))
            cv2.circle(frame_bgr, tip, 8, (255, 0, 255), 2, cv2.LINE_AA)

        if self.done:
            return

        step_index = min(self.step, len(TARGET_NAMES) - 1)
        lines = [
            (f"校准 {self.step + 1}/3：把食指尖停到 {TARGET_NAMES[step_index]} 的红点上", DEFAULT_WHITE),
            ("保持不动约 0.4 秒，进度圈满即自动记录", DEFAULT_YELLOW),
            ("B 上一步    R 重新校准    ESC 退出", DEFAULT_GRAY),
        ]
        if fps is not None:
            lines.append((f"FPS {fps:.1f}", DEFAULT_GREEN))

        width = max(estimate_width(text, 18) for text, _ in lines) + 20
        draw_panel(frame_bgr, 12, 10, width, 24 * len(lines) + 12)
        draw_texts(
            frame_bgr,
            [(text, (22, 16 + i * 24), 18, color) for i, (text, color) in enumerate(lines)],
        )
