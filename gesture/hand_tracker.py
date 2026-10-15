"""MediaPipe Hands 封装：把 BGR 帧变成稳定的 21 关键点 + 手指状态。

关键点索引是固定的（与"哪根手指看起来像食指"无关）：
   4 = 拇指尖, 8 = 食指尖, 12 = 中指尖, 16 = 无名指尖, 20 = 小指尖
所以"识别不出食指"时不需要去猜手指身份，直接按索引取值即可。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

WRIST = 0
THUMB_TIP = 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20

# 四指：(PIP 关节, 指尖)。拇指不参与握拳判定——它捏合时要动，容易误判。
_FINGERS = {
    "index": (INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_PIP, RING_TIP),
    "pinky": (PINKY_PIP, PINKY_TIP),
}


@dataclass
class HandObservation:
    points: np.ndarray  # (21, 2) 相机像素坐标
    handedness: str  # "Left" / "Right"
    palm_scale: float  # 手腕→中指根距离，用于尺度归一（抗远近变化）
    extended: dict  # 四指是否伸直
    pinch_ratio: float  # 拇指尖到食指尖距离 / palm_scale
    pointer: np.ndarray  # 指向点
    pointer_source: str  # "index" | "middle-fallback"
    is_fist: bool


class HandTracker:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        index_extend_ratio: float = 1.06,
        model_complexity: int = 1,
    ) -> None:
        self.index_extend_ratio = float(index_extend_ratio)
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def close(self) -> None:
        self._hands.close()

    def process(self, frame_bgr: np.ndarray) -> Optional[HandObservation]:
        """frame_bgr 必须是最终显示用的那张（镜像已经在外面做完）。"""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        height, width = frame_bgr.shape[:2]
        landmarks = result.multi_hand_landmarks[0].landmark
        points = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=float)

        handedness = ""
        if result.multi_handedness:
            handedness = result.multi_handedness[0].classification[0].label

        return self._build_observation(points, handedness)

    def _build_observation(self, points: np.ndarray, handedness: str) -> HandObservation:
        palm_scale = float(np.linalg.norm(points[MIDDLE_MCP] - points[WRIST]))
        palm_scale = max(palm_scale, 1e-6)

        dist_to_wrist = np.linalg.norm(points - points[WRIST], axis=1)
        extended = {}
        for name, (pip, tip) in _FINGERS.items():
            ratio = float(dist_to_wrist[tip]) / max(float(dist_to_wrist[pip]), 1e-6)
            extended[name] = ratio > self.index_extend_ratio

        # 食指伸直就用食指尖；否则退回"第二根手指的根部"（中指 MCP）。
        # 中指根在掌心附近，手指蜷缩时几乎不动，比硬取食指尖稳定得多。
        if extended["index"]:
            pointer = points[INDEX_TIP]
            pointer_source = "index"
        else:
            pointer = points[MIDDLE_MCP]
            pointer_source = "middle-fallback"

        pinch_ratio = float(np.linalg.norm(points[THUMB_TIP] - points[INDEX_TIP])) / palm_scale
        is_fist = not any(extended.values())

        return HandObservation(
            points=points,
            handedness=handedness,
            palm_scale=palm_scale,
            extended=extended,
            pinch_ratio=pinch_ratio,
            pointer=pointer,
            pointer_source=pointer_source,
            is_fist=is_fist,
        )

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        points: np.ndarray,
        highlight_index: int = INDEX_TIP,
    ) -> None:
        """把骨架画到画面上（原地修改）。"""
        for a, b in self._connections:
            pa = tuple(np.round(points[a]).astype(int))
            pb = tuple(np.round(points[b]).astype(int))
            cv2.line(frame_bgr, pa, pb, (0, 200, 0), 2, cv2.LINE_AA)
        for i, p in enumerate(points):
            center = tuple(np.round(p).astype(int))
            if i == highlight_index:
                cv2.circle(frame_bgr, center, 7, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, center, 11, (255, 0, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, center, 3, (255, 255, 255), -1, cv2.LINE_AA)
