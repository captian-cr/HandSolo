"""集中配置。所有现场可调参数都在这里，改完 config.json 重启即可生效。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


@dataclass
class Config:
    # ---- 摄像头 ----
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    mirror: bool = True  # 镜像显示，用户抬手往左，画面里也往左，符合直觉

    # ---- MediaPipe ----
    max_num_hands: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6

    # ---- One Euro 滤波 ----
    # min_cutoff 越小抖动越小但越迟钝；beta 越大快速移动时越跟手
    filter_min_cutoff: float = 1.2
    filter_beta: float = 0.08

    # ---- 指向 ----
    index_extend_ratio: float = 1.06  # 指尖到手腕 / PIP 到手腕 的比值阈值，判定食指是否伸直
    pointer_dead_zone_px: float = 1.0  # 屏幕像素死区，抑制静止时的微抖

    # ---- 捏合 ----
    pinch_on_ratio: float = 0.32  # 低于该值判定为捏合（需按实测数据调整）
    pinch_off_ratio: float = 0.45  # 高于该值判定为松开（迟滞，防抖）
    pinch_stable_frames: int = 3  # 连续帧数确认，抗单帧误判

    # ---- 握拳暂停 ----
    fist_hold_seconds: float = 0.6  # 握拳保持多久后冻结光标

    # ---- 校准 ----
    calib_margin_x: float = 0.12  # 校准矩形相对画面的左右内缩比例
    calib_margin_y: float = 0.12  # 上下内缩比例

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "Config":
        cfg = cls()
        if not os.path.exists(path):
            return cfg
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # 配置损坏时退回默认值，不阻塞启动
            print(f"[config] 读取 {path} 失败，使用默认值：{exc}")
            return cfg
        known = {f.name for f in fields(cls)}
        for key, value in data.items():
            if key in known:
                setattr(cfg, key, value)
        return cfg

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
