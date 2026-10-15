"""相机像素坐标 → 屏幕像素坐标。

用三点仿射（左上 / 右上 / 左下）求解，等价于"把校准矩形拉成整块屏幕"。
仿射是全局线性的，所以指尖超出矩形范围时只是线性外推，
不会像透视变换那样在四角出现诡异的弯曲——这正是四角可达性的关键。
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .config import PROJECT_ROOT

CALIB_PATH = os.path.join(PROJECT_ROOT, "calibration.json")


class ScreenMapper:
    def __init__(self) -> None:
        self._matrix: np.ndarray | None = None  # 2x3 仿射矩阵
        self.screen_w: int = 0
        self.screen_h: int = 0
        self.cam_rect: tuple | None = None  # 校准时使用的相机像素矩形 (x0, y0, x1, y1)

    @property
    def ready(self) -> bool:
        return self._matrix is not None

    def reset(self) -> None:
        self._matrix = None
        self.cam_rect = None

    def calibrate(self, src_points, screen_size, cam_rect=None) -> None:
        """src_points: [左上, 右上, 左下]，相机像素坐标。"""
        src = np.asarray(src_points, dtype=np.float32)
        if src.shape != (3, 2):
            raise ValueError("校准需要恰好 3 个点：左上、右上、左下")

        screen_w, screen_h = int(screen_size[0]), int(screen_size[1])
        dst = np.array([[0, 0], [screen_w - 1, 0], [0, screen_h - 1]], dtype=np.float32)

        area = abs(
            (src[1][0] - src[0][0]) * (src[2][1] - src[0][1])
            - (src[2][0] - src[0][0]) * (src[1][1] - src[0][1])
        )
        if area < 500:
            raise ValueError("三个校准点几乎共线，请重新采集（左右要拉开距离）")

        matrix = cv2.getAffineTransform(src, dst)
        if not np.isfinite(matrix).all():
            raise ValueError("校准矩阵求解失败，请重新采集")

        self._matrix = matrix.astype(np.float64)
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.cam_rect = tuple(cam_rect) if cam_rect else None

    def to_screen(self, point) -> tuple[float, float]:
        p = np.array([float(point[0]), float(point[1]), 1.0], dtype=np.float64)
        q = self._matrix @ p
        return float(q[0]), float(q[1])

    def to_camera(self, screen_point) -> tuple[float, float]:
        inverse = cv2.invertAffineTransform(self._matrix)
        p = np.array([float(screen_point[0]), float(screen_point[1]), 1.0], dtype=np.float64)
        q = inverse @ p
        return float(q[0]), float(q[1])

    def save(self, path: str = CALIB_PATH) -> None:
        if not self.ready:
            raise RuntimeError("尚未校准，无法保存")
        payload = {
            "matrix": self._matrix.tolist(),
            "screen": [self.screen_w, self.screen_h],
            "cam_rect": list(self.cam_rect) if self.cam_rect else None,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def load(self, path: str = CALIB_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            matrix = np.asarray(payload["matrix"], dtype=np.float64)
            if matrix.shape != (2, 3):
                raise ValueError("矩阵维度不是 2x3")
            self._matrix = matrix
            self.screen_w = int(payload["screen"][0])
            self.screen_h = int(payload["screen"][1])
            rect = payload.get("cam_rect")
            self.cam_rect = tuple(rect) if rect else None
        except Exception as exc:
            print(f"[mapping] 读取校准文件失败：{exc}")
            self.reset()
            return False
        return True
