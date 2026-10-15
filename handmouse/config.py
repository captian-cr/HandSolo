"""配置：仓库根目录的 ``config.json`` 为唯一真源，CLI flags 可临时覆盖。

设计取舍：MVP 阶段不做设置面板，所以没有 GUI 配置界面。所有现场可调参数
集中在 ``AppConfig``，改完 ``config.json`` 重启即可生效；命令行只提供少量
覆盖项（``--camera`` / ``--margin`` / ``--safe-mode`` 等）。

读取策略：文件缺失或损坏时退回默认值并且**不阻塞启动**，只在控制台提示。
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


def _base_dir() -> Path:
    """配置文件所在目录。

    打包成单 exe 后 ``__file__`` 指向解压出来的一次性临时目录，写进去的配置
    下次启动就没了，所以冻结模式下改用 exe 自身所在目录。
    """

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _base_dir()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

# 默认动作绑定：画完字母/形状后捏合确认即执行。
# 想改成自己的习惯，直接编辑 config.json 的 motion_bindings 即可。
DEFAULT_MOTION_BINDINGS: dict[str, dict[str, str]] = {
    "W": {"kind": "launch", "target": "notepad"},
    "C": {"kind": "launch", "target": "calc"},
    "E": {"kind": "launch", "target": "explorer"},
    "X": {"kind": "hotkey", "target": "ctrl+w"},
    "CIRCLE": {"kind": "hotkey", "target": "win+d"},
    "LINE": {"kind": "hotkey", "target": "alt+tab"},
}

# LAUNCH 白名单：别名 → 可执行文件（写全路径或 PATH 内名字均可）。
# 不在白名单里的目标一律拒绝执行。
DEFAULT_LAUNCH_WHITELIST: dict[str, str] = {
    "notepad": "notepad.exe",
    "calc": "calc.exe",
    "explorer": "explorer.exe",
    "settings": "ms-settings:",
}


@dataclass(slots=True)
class AppConfig:
    """应用可持久化配置。"""

    # ---- 摄像头 ----
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    mirror: bool = True
    handedness: str = "Any"
    recognition_zoom: float = 1.15

    # ---- 识别阈值 ----
    min_detection_confidence: float = 0.40
    min_presence_confidence: float = 0.35
    min_tracking_confidence: float = 0.40

    # ---- 光标滤波与映射 ----
    pointer_stability: float = 0.58
    pointer_response: float = 0.82
    cursor_interpolation_ms: int = 42
    pointer_margin_x: float = 0.20
    pointer_margin_y: float = 0.16
    pointer_edge_override: bool = True

    # ---- 捏合 / 手势 ----
    pinch_engage: float = 0.42
    pinch_release: float = 0.55
    pinch_arm: float = 0.72
    right_pinch_approach_delta: float = 0.22
    drag_hold_ms: int = 5000
    fist_hold_ms: int = 500
    hand_lost_ms: int = 200
    scroll_threshold: float = 0.12
    scroll_sensitivity: float = 1.0
    rotation_gestures_enabled: bool = True
    palm_rotation_threshold: float = 0.18
    page_gestures_enabled: bool = True
    page_swipe_threshold: float = 0.70

    # ---- 进入门（测量域闸门）----
    # 全局闸门：未进入控制态时，下游手势（捏合/翻页/拖拽）一律不执行，从根上
    # 挡住"脸被误判成手"产生的幻影动作。默认关，与 virtual_desk_experimental
    # 一样先在一轮引导基线里验证，再决定是否转正。
    #
    # 已知 gap（勿误读为"门是全局的"）：本闸门当前只覆盖 GestureEngine，
    # DTW 轨迹引擎（motion_gestures）未纳入——它在 pipeline 里先于 GestureEngine
    # 运行，输入不受闸门约束。待 r3 数据确认 DTW 在"无意图段"是否出现非 idle
    # （r2 为 0 次）后再决定是否补纳入；r3 复盘时 DTW 的 [轨迹] 单独统计，
    # 它出非 idle 记"DTW 未纳入门"，不记"门漏了"。
    entry_gate_enabled: bool = False
    # 进入姿势（掌心朝镜 + 五指张开）需连续保持的时间；与双指悬停同量级，
    # 不引入新的时间常数。退出复用既有常数：握拳退出沿用 fist_hold_ms，
    # 手出画退出沿用 hand_lost_ms。
    entry_hold_ms: int = 300

    # ---- 局部续跟踪 ----
    partial_finger_tracking: bool = True
    # 0 表示只按光流质量决定续跟踪寿命；正数用于诊断或保守限时。
    partial_tracking_frames: int = 0

    # ---- 控件磁吸（需要 handmouse[uia]）----
    cursor_snap_enabled: bool = True
    cursor_snap_radius_px: int = 30

    # ---- 屏幕 / 多屏（ScreenMap）----
    multi_monitor: bool = False
    # 实验开关：multi_monitor=False 时也走 MOUSEEVENTF_VIRTUALDESK，
    # 用于在多屏机器上先验证官方推荐路径，再决定是否把 multi_monitor 打开。
    virtual_desk_experimental: bool = False

    # ---- 轨迹手势（DTW，第二阶段新增）----
    # 流程：双指(食指+中指)悬停 motion_arm_hold_ms → 绘制 → 收指出候选 → 捏合确认执行。
    motion_gestures_enabled: bool = True
    motion_arm_hold_ms: int = 250
    # 悬停期间允许的指尖漂移（归一化图像坐标）；超过就认为是"在移动光标"
    # 而不是"准备画画"，重新计时。这条是防误触发的关键。
    motion_arm_max_drift: float = 0.05
    motion_min_duration_ms: int = 250
    motion_min_path_px: float = 60.0
    motion_template_points: int = 32
    motion_max_candidates: int = 3
    motion_accept_distance: float = 0.18
    motion_confirm_timeout_ms: int = 2500
    motion_confirm_pinch: float = 0.42
    motion_hand_lost_ms: int = 200

    # ---- 动作绑定与 LAUNCH 白名单 ----
    motion_bindings: dict[str, dict[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_MOTION_BINDINGS),
    )
    launch_enabled: bool = True
    launch_whitelist: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_LAUNCH_WHITELIST),
    )

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "AppConfig":
        """忽略未知字段，避免旧配置文件带崩新版本。"""

        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in raw.items() if key in allowed}
        return cls(**values)  # type: ignore[arg-type]

    def validate(self) -> None:
        """保存或使用前的防御性校验，全部就地钳到安全区间。"""

        self.camera_index = max(0, int(self.camera_index))
        self.camera_width = max(320, int(self.camera_width))
        self.camera_height = max(240, int(self.camera_height))
        self.camera_fps = max(5, min(60, int(self.camera_fps)))
        self.recognition_zoom = max(1.0, min(1.35, float(self.recognition_zoom)))
        self.pointer_stability = max(0.0, min(1.0, float(self.pointer_stability)))
        self.pointer_response = max(0.0, min(1.0, float(self.pointer_response)))
        self.cursor_interpolation_ms = max(12, min(60, int(self.cursor_interpolation_ms)))
        self.pointer_margin_x = max(0.0, min(0.4, float(self.pointer_margin_x)))
        self.pointer_margin_y = max(0.0, min(0.4, float(self.pointer_margin_y)))
        self.pinch_engage = max(0.1, min(1.0, float(self.pinch_engage)))
        self.pinch_release = max(self.pinch_engage + 0.02, min(1.2, float(self.pinch_release)))
        self.pinch_arm = max(self.pinch_release + 0.05, min(1.4, float(self.pinch_arm)))
        self.right_pinch_approach_delta = max(
            0.10,
            min(0.60, float(self.right_pinch_approach_delta)),
        )
        self.drag_hold_ms = max(300, min(10_000, int(self.drag_hold_ms)))
        self.fist_hold_ms = max(300, min(1500, int(self.fist_hold_ms)))
        self.hand_lost_ms = max(100, min(1000, int(self.hand_lost_ms)))
        self.scroll_threshold = max(0.03, min(0.6, float(self.scroll_threshold)))
        self.scroll_sensitivity = max(0.2, min(4.0, float(self.scroll_sensitivity)))
        self.palm_rotation_threshold = max(0.08, min(0.60, float(self.palm_rotation_threshold)))
        self.page_swipe_threshold = max(0.35, min(1.5, float(self.page_swipe_threshold)))
        self.entry_hold_ms = max(80, min(1500, int(self.entry_hold_ms)))
        self.partial_tracking_frames = max(0, min(600, int(self.partial_tracking_frames)))
        self.min_detection_confidence = max(0.25, min(0.90, float(self.min_detection_confidence)))
        self.min_presence_confidence = max(0.25, min(0.90, float(self.min_presence_confidence)))
        self.min_tracking_confidence = max(0.25, min(0.90, float(self.min_tracking_confidence)))
        self.cursor_snap_radius_px = max(12, min(80, int(self.cursor_snap_radius_px)))
        if self.handedness not in {"Left", "Right", "Any"}:
            self.handedness = "Right"

        self.motion_arm_hold_ms = max(80, min(1500, int(self.motion_arm_hold_ms)))
        self.motion_arm_max_drift = max(0.005, min(0.3, float(self.motion_arm_max_drift)))
        self.motion_min_duration_ms = max(80, min(3000, int(self.motion_min_duration_ms)))
        self.motion_min_path_px = max(10.0, min(1000.0, float(self.motion_min_path_px)))
        self.motion_template_points = max(8, min(128, int(self.motion_template_points)))
        self.motion_max_candidates = max(1, min(6, int(self.motion_max_candidates)))
        self.motion_accept_distance = max(0.02, min(0.6, float(self.motion_accept_distance)))
        self.motion_confirm_timeout_ms = max(
            500,
            min(10_000, int(self.motion_confirm_timeout_ms)),
        )
        self.motion_confirm_pinch = max(0.10, min(0.95, float(self.motion_confirm_pinch)))
        self.motion_hand_lost_ms = max(100, min(1000, int(self.motion_hand_lost_ms)))

        self.motion_bindings = self._sanitize_bindings(self.motion_bindings)
        self.launch_whitelist = self._sanitize_whitelist(self.launch_whitelist)

    @staticmethod
    def _sanitize_bindings(raw: object) -> dict[str, dict[str, str]]:
        """只保留 ``{kind, target}`` 且 kind 合法的绑定。"""

        if not isinstance(raw, dict):
            return dict(DEFAULT_MOTION_BINDINGS)
        cleaned: dict[str, dict[str, str]] = {}
        for name, spec in raw.items():
            if not isinstance(spec, dict):
                continue
            kind = str(spec.get("kind", "")).strip().lower()
            target = str(spec.get("target", "")).strip()
            if kind not in {"launch", "hotkey"} or not target:
                continue
            cleaned[str(name).strip().upper()] = {"kind": kind, "target": target}
        return cleaned

    @staticmethod
    def _sanitize_whitelist(raw: object) -> dict[str, str]:
        if not isinstance(raw, dict):
            return dict(DEFAULT_LAUNCH_WHITELIST)
        return {
            str(alias).strip().lower(): str(command).strip()
            for alias, command in raw.items()
            if str(command).strip()
        }


class ConfigStore:
    """以原子替换方式管理 JSON 配置。"""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_CONFIG_PATH

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            config = AppConfig.from_mapping(raw)
            config.validate()
            return config
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"[config] 读取 {self.path} 失败，使用默认值：{exc}")
            return AppConfig()

    def save(self, config: AppConfig) -> None:
        config.validate()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(asdict(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def reset(self) -> AppConfig:
        config = AppConfig()
        self.save(config)
        return config
