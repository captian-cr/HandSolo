"""基于关键点几何关系的手势状态机。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, hypot, sqrt
from time import monotonic_ns

from handmouse.config import AppConfig
from handmouse.contracts import (
    GestureType,
    HandObservation,
    Landmark,
    MouseCommand,
    MouseCommandType,
)
from handmouse.filters import OneEuroPointerFilter, map_activity_region


class ControlState(str, Enum):
    """控制状态机内部状态。"""

    IDLE = "idle"
    MOVING = "moving"
    PINCH_PENDING = "pinch_pending"
    DRAGGING = "dragging"
    SCROLLING = "scrolling"
    PAUSED = "paused"
    LOST = "lost"


def _distance(a: Landmark, b: Landmark) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _distance_2d(a: Landmark, b: Landmark) -> float:
    """手势阈值使用图像平面距离，避开单目深度坐标的帧间噪声。"""

    return hypot(a.x - b.x, a.y - b.y)


def _angle(a: Landmark, vertex: Landmark, c: Landmark) -> float:
    """返回以 ``vertex`` 为顶点的图像平面夹角。

    单目模型的相对深度在侧手和弱光下更容易波动，屈伸分类只依赖二维
    骨架即可，因此不让z噪声把伸直的手指误判为弯曲。
    """

    first = (a.x - vertex.x, a.y - vertex.y)
    second = (c.x - vertex.x, c.y - vertex.y)
    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return 0.0
    cosine = sum(a_value * b_value for a_value, b_value in zip(first, second))
    cosine /= first_norm * second_norm
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def _palm_width(points: tuple[Landmark, ...]) -> float:
    return max(_distance_2d(points[5], points[17]), 1e-4)


def _palm_center(points: tuple[Landmark, ...]) -> Landmark:
    indices = (0, 5, 9, 13, 17)
    return Landmark(
        x=sum(points[index].x for index in indices) / len(indices),
        y=sum(points[index].y for index in indices) / len(indices),
        z=sum(points[index].z for index in indices) / len(indices),
    )


def _palm_depth_tilt(points: tuple[Landmark, ...]) -> float:
    """返回腕部到中指掌指关节的归一化深度差。

    掌心前后翻转时，掌心在画面中的中心可以几乎不动，但9号点相对
    手腕的深度会稳定改变。以掌宽归一化后可兼容不同距离和手型。
    """

    return (points[9].z - points[0].z) / _palm_width(points)


def _finger_extensions(points: tuple[Landmark, ...]) -> tuple[bool, bool, bool, bool]:
    fingers = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    return tuple(_angle(points[mcp], points[pip], points[tip]) >= 135.0 for mcp, pip, tip in fingers)  # type: ignore[return-value]


def _is_pointer(points: tuple[Landmark, ...]) -> bool:
    # 张掌和捏合已在状态机中优先处理。此处只要求食指伸展，允许其余
    # 手指自然半弯，避免用户必须维持僵硬的“标准手势”。
    index, _middle, _ring, _pinky = _finger_extensions(points)
    return index


def _is_open_palm(points: tuple[Landmark, ...]) -> bool:
    index, middle, ring, pinky = _finger_extensions(points)
    return index and middle and sum((index, middle, ring, pinky)) >= 3


def _is_fist(points: tuple[Landmark, ...]) -> bool:
    if any(_finger_extensions(points)):
        return False
    center = _palm_center(points)
    width = _palm_width(points)
    tips = (8, 12, 16, 20)
    mean_distance = sum(_distance(points[index], center) for index in tips) / len(tips)
    return mean_distance / width < 1.25


#: 进入门阈值：拇指外展程度 = 拇指尖到小指掌指关节的距离 / 掌宽。
_ENTRY_THUMB_SPREAD_MIN = 0.9
#: 进入门阈值：掌面正对镜头程度 = 掌宽 / 腕到中指指尖的距离，用于排除侧掌。
_ENTRY_PALM_FRONTAL_MIN = 0.25


def _is_five_open(points: tuple[Landmark, ...]) -> bool:
    """五指张开：四指伸直且拇指明显外展。

    ``_is_open_palm`` 只要 3/4 指伸直（照顾日常滚轮姿势），进不了进入门。
    进入门是全局闸门，判据必须更严，否则"半握"也能把闸门顶开。
    """

    if not all(_finger_extensions(points)):
        return False
    return (
        _distance_2d(points[4], points[17]) / _palm_width(points)
        >= _ENTRY_THUMB_SPREAD_MIN
    )


def _palm_presented(points: tuple[Landmark, ...], handedness: str) -> bool:
    """掌心朝镜且掌面基本正对镜头（排除侧掌与手背朝镜）。

    单目模型给不出掌心法向，但"掌心/掌背"在图像里是镜像关系：掌心朝镜时
    拇指落在掌轴(5→17)的一侧，掌背朝镜时落在另一侧。取掌轴与"指向拇指"
    矢量(5→4)的二维叉积，符号对图像内旋转不变；再用 MediaPipe 的
    handedness（自拍镜像输入下即解剖学左右手）选择期望符号。

    符号推导（镜像自拍坐标，x 向右、y 向下）：
      * 右手掌心朝镜：拇指外侧朝人体右侧→画面左→镜像后画面右，故掌轴
        (5→17) 指向 +x，指向拇指的矢量 (5→4) 指向 -y，叉积为正；
      * 手背朝镜则拇指翻到另一侧，叉积为负；
      * 左手关于画面纵轴镜像，符号相反。

    handedness 缺失时返回 False（保守关闭闸门）：宁可进不去控制态，也不
    能让手背/侧掌把手误开闸门。
    """

    span = _distance_2d(points[0], points[12])
    if span < 1e-6 or _palm_width(points) / span < _ENTRY_PALM_FRONTAL_MIN:
        return False
    if handedness not in ("Left", "Right"):
        return False
    across_x = points[17].x - points[5].x
    across_y = points[17].y - points[5].y
    thumb_x = points[4].x - points[5].x
    thumb_y = points[4].y - points[5].y
    cross = across_x * thumb_y - across_y * thumb_x
    if abs(cross) < 1e-9:
        return False
    return cross > 0.0 if handedness == "Right" else cross < 0.0


# 进入门诊断开关：环境变量控制，跨模块生效。
VERBOSE_GATE = os.environ.get("HANDMOUSE_VERBOSE_GATE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_DIAG_LOGGER = logging.getLogger("handmouse.gate-diag")


@dataclass(slots=True)
class GestureDebug:
    """最近一次识别的几何量，便于后续校准。"""

    left_pinch_ratio: float = 99.0
    right_pinch_ratio: float = 99.0
    palm_width: float = 0.0


class GestureEngine:
    """处理时序冲突并生成标准鼠标指令。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.filter = self._build_pointer_filter(config)
        self.state = ControlState.IDLE
        self.current_gesture = GestureType.NONE
        self.paused = False
        # 最近一帧有效观测"到达控制线程"的墙钟时刻。tick() 的失联判定用的也是
        # 墙钟，两者必须同源；若记采集时刻戳，elapsed 会混入取包延迟而假性失联。
        self.last_hand_ns: int | None = None
        self.debug = GestureDebug()
        # 进入门：entry_gate_enabled 关闭时恒为 True，与旧行为完全等价。
        self._gate_active = not config.entry_gate_enabled
        self._entry_started_ns: int | None = None
        # 诊断用：进入门几何判定层的帧计数和去重缓存
        self._gate_diag_frame = 0
        self._gate_diag_last_sig: tuple | None = None
        # 诊断用：gate_eligible 的逐帧翻转计数（False→True）。用于把"lost 路径
        # 清零"与"姿势判定在门槛附近翻动导致清零"分开归因。
        self._gate_diag_prev_eligible = False
        self._gate_diag_eligible_flips = 0
        # 诊断用：最近一次观测被 process() 消费的墙钟时刻（到达时刻戳）。
        # 笔1 之后 last_hand_ns 已改用同一到达时刻打点，本字段因此恒等于
        # last_hand_ns（差值为 0），tick() 的 elapsed 不再含"取包延迟"段；
        # 取包延迟改用 pipeline 的 [采样] latency_ms 观察。仅在 VERBOSE_GATE 下写入。
        self._last_hand_arrival_ns: int | None = None

        self._left_active = False
        self._left_started_ns: int | None = None
        self._right_active = False
        self._right_started_ns: int | None = None
        self._right_ready = False
        self._right_open_ratio = 0.0
        self._right_requires_middle_extension = False
        self._fist_started_ns: int | None = None
        self._fist_latched = False
        self._open_anchor_x: float | None = None
        self._scroll_anchor_y: float | None = None
        self._rotation_tilt_anchor: float | None = None
        self._rotation_scroll_active = False
        self._rotation_last_open_ns: int | None = None
        self._pointer_tilt_anchor: float | None = None
        self._pointer_rotation_active = False
        self._page_swipe_latched = False
        self._lost_reported = False
        self._left_button_down = False

    def update_config(self, config: AppConfig) -> None:
        self.config = config
        self.filter = self._build_pointer_filter(config)
        if not config.entry_gate_enabled:
            # 关掉闸门时立刻回到"始终可控"，不再要求重新做进入姿势。
            self._gate_active = True
            self._entry_started_ns = None

    def _exit_gate(self) -> None:
        """握拳退出控制态：闸门回到未开。

        退出后手必须重新做进入姿势（掌心朝镜+五指张开 300ms）才能再控制，
        这样一次幻影握拳最多只是把闸门关上，不会连锁触发下游动作。
        """

        self._gate_active = False
        self._entry_started_ns = None
        self._reset_transient()
        self.filter.reset()
        self.state = ControlState.IDLE
        self.current_gesture = GestureType.GATE_IDLE

    @staticmethod
    def _build_pointer_filter(config: AppConfig) -> OneEuroPointerFilter:
        min_cutoff = 3.0 - 2.2 * config.pointer_stability
        beta = 0.35 + 1.55 * config.pointer_response
        return OneEuroPointerFilter(min_cutoff=min_cutoff, beta=beta)

    def _command(
        self,
        kind: MouseCommandType,
        timestamp_ns: int,
        **kwargs: float | int | bool | None,
    ) -> MouseCommand:
        return MouseCommand(kind=kind, timestamp_ns=timestamp_ns, **kwargs)

    def _cursor(self, point: Landmark, timestamp_ns: int) -> tuple[float, float]:
        mapped = map_activity_region(
            point.x,
            point.y,
            self.config.pointer_margin_x,
            self.config.pointer_margin_y,
        )
        filtered_x, filtered_y = self.filter.update(*mapped, timestamp_ns)
        if self.config.pointer_edge_override:
            # 关键点到达安全活动区边界时直接完成最后一小段屏幕行程。
            # 这样不需要把手移出摄像头等待滤波器慢慢收敛到0或1。
            if mapped[0] <= 0.0 or mapped[0] >= 1.0:
                filtered_x = mapped[0]
            if mapped[1] <= 0.0 or mapped[1] >= 1.0:
                filtered_y = mapped[1]
        return filtered_x, filtered_y

    def _release_left(self, timestamp_ns: int) -> list[MouseCommand]:
        if not self._left_button_down:
            return []
        self._left_button_down = False
        return [self._command(MouseCommandType.LEFT_UP, timestamp_ns)]

    def _reset_transient(self) -> None:
        self._left_active = False
        self._left_started_ns = None
        self._right_active = False
        self._right_started_ns = None
        self._right_ready = False
        self._right_open_ratio = 0.0
        self._right_requires_middle_extension = False
        self._reset_orientation_tracking()

    def _reset_orientation_tracking(self) -> None:
        self._reset_scroll_tracking()
        self._reset_pointer_rotation()

    def _reset_scroll_tracking(self) -> None:
        self._open_anchor_x = None
        self._scroll_anchor_y = None
        self._rotation_tilt_anchor = None
        self._rotation_scroll_active = False
        self._rotation_last_open_ns = None
        self._page_swipe_latched = False

    def _reset_pointer_rotation(self) -> None:
        self._pointer_tilt_anchor = None
        self._pointer_rotation_active = False

    def set_paused(
        self,
        paused: bool,
        timestamp_ns: int | None = None,
    ) -> list[MouseCommand]:
        """由托盘或快捷键显式设置暂停状态。"""

        now = timestamp_ns or monotonic_ns()
        commands = self._release_left(now)
        self.paused = paused
        self.state = ControlState.PAUSED if paused else ControlState.IDLE
        self.current_gesture = GestureType.PAUSED if paused else GestureType.NONE
        self._reset_transient()
        self.filter.reset()
        commands.append(
            self._command(MouseCommandType.PAUSE, now, paused=paused),
        )
        return commands

    def emergency_stop(self, timestamp_ns: int | None = None) -> list[MouseCommand]:
        """立即停止注入并进入暂停状态。"""

        return self.set_paused(True, timestamp_ns)

    def hold(self, timestamp_ns: int) -> list[MouseCommand]:
        """轨迹手势进行中：冻结光标、释放已按下的左键、但不触发失联保护。

        绘制期间观测帧不交给本状态机（否则捏合会被当成点击），但手其实一直
        在线，所以必须自己续上 ``last_hand_ns``，不然 200ms 后 ``tick`` 会误报
        手部丢失并把状态刷成 LOST。
        """

        self.last_hand_ns = monotonic_ns()
        self._lost_reported = False
        commands = self._release_left(timestamp_ns)
        self._reset_transient()
        return commands

    def process(self, observation: HandObservation) -> list[MouseCommand]:
        """处理一帧有效手部观测。"""

        # 到达时刻戳：本帧交付到控制线程的墙钟时刻。tick() 的失联判定用的是
        # 墙钟，last_hand_ns 必须记在同一时钟上；若沿用 observation.timestamp_ns
        # （采集时刻戳），elapsed = 墙钟 now - 采集时刻 会把"取包延迟 + 帧间隔"
        # 算成经过时间，10Hz 下即可逼近甚至越过 hand_lost_ms 而假性失联。
        arrival_ns = monotonic_ns()
        points = observation.landmarks
        if self.config.handedness != "Any" and observation.handedness:
            if observation.handedness != self.config.handedness:
                # handedness 不符时本帧按"无有效观测"处理，同样交给墙钟。
                return self.tick(arrival_ns)

        now = observation.timestamp_ns

        # -------- 进入门诊断：几何判定层 --------
        # 限流：每 5 帧（~6Hz）+ gate_eligible/palm_presented/five_open 任一变 True 时立即打。
        # 相同内容去重用 signature。所有新计算都包在 if VERBOSE_GATE 内。
        if VERBOSE_GATE:
            self._gate_diag_frame += 1
            # 到达时刻戳：与本帧 last_hand_ns 同源（同一个 arrival_ns），
            # 因此 elapsed 拆分里的 dwell 段在笔1后恒为 0。
            self._last_hand_arrival_ns = arrival_ns
            should_log = False

            if len(points) == 21:
                # ---- 完整几何量 ----
                finger_exts = _finger_extensions(points)
                all_ext = all(finger_exts)
                palm_w = _palm_width(points)
                span = _distance_2d(points[0], points[12])
                palm_frontal_ratio = palm_w / span if span >= 1e-6 else 0.0
                five_open = all_ext and (
                    _distance_2d(points[4], points[17]) / palm_w >= _ENTRY_THUMB_SPREAD_MIN
                )

                hn = observation.handedness
                hn_valid = hn in ("Left", "Right")

                palm_presented = False
                if hn_valid and palm_frontal_ratio >= _ENTRY_PALM_FRONTAL_MIN and span >= 1e-6:
                    across_x = points[17].x - points[5].x
                    across_y = points[17].y - points[5].y
                    thumb_x = points[4].x - points[5].x
                    thumb_y = points[4].y - points[5].y
                    cross = across_x * thumb_y - across_y * thumb_x
                    expected_sign = 1.0 if hn == "Right" else -1.0
                    cross_match = (cross > 0.0) if hn == "Right" else (cross < 0.0)
                    palm_presented = cross_match and abs(cross) >= 1e-9
                else:
                    cross = 0.0
                    expected_sign = 0.0
                    cross_match = False

                gate_eligible = (not self._gate_active) and five_open and palm_presented
                held_ms = (now - self._entry_started_ns) / 1_000_000 if self._entry_started_ns else 0.0
                # 翻转计数：本帧由"不满足"变"满足"记一次。lost 清零前若 flips 无增量，
                # 说明清零来自 lost 路径；若 flips 增量大，说明是姿势在门槛附近翻动。
                if gate_eligible and not self._gate_diag_prev_eligible:
                    self._gate_diag_eligible_flips += 1
                self._gate_diag_prev_eligible = gate_eligible

                sig = (
                    self._gate_active, gate_eligible, palm_presented,
                    five_open, all_ext, hn_valid,
                )

                # 立即打条件：任一关键布尔量变 True
                if sig != self._gate_diag_last_sig:
                    should_log = True
                elif self._gate_diag_frame % 5 == 0:
                    should_log = True

                if should_log:
                    self._gate_diag_last_sig = sig
                    arrival_ns = self._last_hand_arrival_ns
                    _DIAG_LOGGER.info(
                        "geom ts=%d arrival=%d arrive_ms=%.0f gate_active=%s "
                        "entry_started=%s held_ms=%.0f flips=%d state=%s gesture=%s",
                        now,
                        arrival_ns if arrival_ns is not None else -1,
                        (arrival_ns - now) / 1_000_000 if arrival_ns is not None else -1.0,
                        self._gate_active,
                        self._entry_started_ns is not None, held_ms,
                        self._gate_diag_eligible_flips,
                        self.state.value, self.current_gesture.value,
                    )
                    _DIAG_LOGGER.info(
                        "  palm_width=%.4f span=%.4f frontal_ratio=%.4f(min=%.2f) "
                        "→ palm_presented=%s hn=%s valid=%s",
                        palm_w, span, palm_frontal_ratio, _ENTRY_PALM_FRONTAL_MIN,
                        palm_presented, hn or '""', hn_valid,
                    )
                    _DIAG_LOGGER.info(
                        "  k0=(%.3f,%.3f,%.3f) k4=(%.3f,%.3f,%.3f) k5=(%.3f,%.3f,%.3f)",
                        points[0].x, points[0].y, points[0].z,
                        points[4].x, points[4].y, points[4].z,
                        points[5].x, points[5].y, points[5].z,
                    )
                    _DIAG_LOGGER.info(
                        "  k8=(%.3f,%.3f,%.3f) k9=(%.3f,%.3f,%.3f) k12=(%.3f,%.3f,%.3f) k17=(%.3f,%.3f,%.3f)",
                        points[8].x, points[8].y, points[8].z,
                        points[9].x, points[9].y, points[9].z,
                        points[12].x, points[12].y, points[12].z,
                        points[17].x, points[17].y, points[17].z,
                    )
                    _DIAG_LOGGER.info(
                        "  finger_exts=(%s,%s,%s,%s) all_ext=%s thumb_spread=%.4f(min=%.2f) "
                        "→ five_open=%s",
                        *finger_exts, all_ext,
                        _distance_2d(points[4], points[17]) / palm_w,
                        _ENTRY_THUMB_SPREAD_MIN, five_open,
                    )
                    if hn_valid and palm_frontal_ratio >= _ENTRY_PALM_FRONTAL_MIN:
                        _DIAG_LOGGER.info(
                            "  cross=%.6f expected_sign=%s cross_match=%s → gate_eligible=%s",
                            cross, "+1" if expected_sign > 0 else "-1",
                            cross_match, gate_eligible,
                        )
                    else:
                        _DIAG_LOGGER.info(
                            "  cross=(skip) expected_sign=N/A → gate_eligible=%s",
                            gate_eligible,
                        )
            else:
                # landmarks 不完整（partial 或真 0 点）：判定链必然不满足，
                # 同步翻转状态，避免下一帧被误记成 False→True。
                self._gate_diag_prev_eligible = False
                if self._gate_diag_frame % 5 == 0:
                    _DIAG_LOGGER.info(
                        "geom ts=%d landmarks_len=%d partial_finger=%s "
                        "gate_active=%s held_ms=%.0f",
                        now, len(points),
                        observation.partial_finger or "None",
                        self._gate_active,
                        (now - (self._entry_started_ns or now)) / 1_000_000,
                    )

        # ---- 进入门（测量域闸门）----
        # 未进入控制态时只评估"进入姿势"，捏合/翻页/拖拽等下游判定一概不跑。
        # 这样脸被误判成手时最多停在 gate_idle，幻影手势连控制态都进不去。
        if not self._gate_active:
            self.last_hand_ns = arrival_ns
            self._lost_reported = False
            if (
                len(points) == 21
                and _is_five_open(points)
                and _palm_presented(points, observation.handedness)
            ):
                if self._entry_started_ns is None:
                    self._entry_started_ns = now
                held_ms = (now - self._entry_started_ns) / 1_000_000
                if held_ms < self.config.entry_hold_ms:
                    self.state = ControlState.IDLE
                    self.current_gesture = GestureType.GATE_IDLE
                    return []
                # 姿势已满时长：本帧开闸，继续走正常手势判定。
                self._gate_active = True
            else:
                self._entry_started_ns = None
                self.state = ControlState.IDLE
                self.current_gesture = GestureType.GATE_IDLE
                return []
            self._entry_started_ns = None

        if observation.partial_finger and observation.partial_position is not None:
            commands = self._release_left(now)
            self._reset_transient()
            self.last_hand_ns = arrival_ns
            self._lost_reported = False
            if self.paused:
                self.state = ControlState.PAUSED
                self.current_gesture = GestureType.PAUSED
                return commands
            if observation.partial_finger == "Index":
                self.current_gesture = GestureType.PARTIAL_INDEX
                self.state = ControlState.MOVING
                x, y = self._cursor(observation.partial_position, now)
                commands.append(self._command(MouseCommandType.MOVE, now, x=x, y=y))
            else:
                self.current_gesture = GestureType.PARTIAL_MIDDLE
                self.state = ControlState.IDLE
            return commands
        if len(points) != 21:
            return []

        self.last_hand_ns = arrival_ns
        self._lost_reported = False
        width = _palm_width(points)
        left_ratio = _distance_2d(points[4], points[8]) / width
        right_ratio = _distance_2d(points[4], points[12]) / width
        self.debug = GestureDebug(left_ratio, right_ratio, width)

        open_palm = _is_open_palm(points)
        pointer_shape = _is_pointer(points)
        palm_tilt = _palm_depth_tilt(points)
        middle_pinch_extended = (
            _angle(points[9], points[10], points[12]) >= 105.0
        )
        if not self.paused and not self._right_active:
            if (
                (open_palm or pointer_shape)
                and right_ratio >= self.config.pinch_arm
            ):
                self._right_ready = True
                self._right_open_ratio = max(self._right_open_ratio, right_ratio)
                # 张掌可直接开始右键动作，但后续接触帧必须仍保留中指
                # 主动伸出；这能区分真正的中指捏合与收起中指时偶然贴近。
                self._right_requires_middle_extension = open_palm
        right_shape_valid = (
            not self._right_requires_middle_extension
            or middle_pinch_extended
        )
        right_approach = self._right_open_ratio - right_ratio
        right_near_candidate = (
            self._right_active
            or (
                self._right_ready
                and right_shape_valid
                and right_ratio <= self.config.pinch_arm
                and right_approach
                >= self.config.right_pinch_approach_delta * 0.5
            )
        )
        pinch_proximity = (
            left_ratio <= self.config.pinch_arm
            or right_near_candidate
        )

        # 录屏中的“食指指向后向下翻腕”会让二维骨架短暂接近握拳。
        # 若前一帧明确处于光标模式，则以掌面深度变化确认这是翻腕，
        # 继续使用食指尖更新光标，而不是启动握拳暂停计时。
        if (
            not self.paused
            and self.config.rotation_gestures_enabled
            and self._pointer_tilt_anchor is not None
            and not open_palm
            and not pinch_proximity
        ):
            pointer_tilt_delta = palm_tilt - self._pointer_tilt_anchor
            if (
                self._pointer_rotation_active
                or abs(pointer_tilt_delta) >= self.config.palm_rotation_threshold
            ):
                self._pointer_rotation_active = True
                self._pointer_tilt_anchor = palm_tilt
                self.current_gesture = GestureType.POINTER
                self.state = ControlState.MOVING
                x, y = self._cursor(points[8], now)
                return [self._command(MouseCommandType.MOVE, now, x=x, y=y)]

        # 张掌后允许通过前后翻腕控制滚轮。该判定必须先于握拳：翻腕时
        # 四指的二维投影会缩短，旧逻辑会把平展的手误判成握拳并暂停。
        if (
            not self.paused
            and self.config.rotation_gestures_enabled
            and self._rotation_tilt_anchor is not None
            and not open_palm
            and not pinch_proximity
        ):
            tilt_delta = palm_tilt - self._rotation_tilt_anchor
            if abs(tilt_delta) >= self.config.palm_rotation_threshold:
                notches = round(
                    -tilt_delta * self.config.scroll_sensitivity * 6,
                )
                if notches == 0:
                    notches = -1 if tilt_delta > 0 else 1
                notches = max(-3, min(3, notches))
                self._rotation_tilt_anchor = palm_tilt
                self._rotation_scroll_active = True
                self.current_gesture = GestureType.SCROLL
                self.state = ControlState.SCROLLING
                return [
                    self._command(
                        MouseCommandType.WHEEL,
                        now,
                        delta=notches * 120,
                    ),
                ]
            if self._rotation_scroll_active and not open_palm:
                self.current_gesture = GestureType.SCROLL
                self.state = ControlState.SCROLLING
                return []

        fist = _is_fist(points)
        if fist and not pinch_proximity:
            # 从张掌进入翻腕的最初几帧可能尚未跨过滚轮阈值。给深度
            # 姿态一个短暂判定窗口，避免先启动500ms握拳暂停计时。
            if (
                not self.paused
                and self.config.rotation_gestures_enabled
                and self._rotation_last_open_ns is not None
                and (now - self._rotation_last_open_ns) / 1_000_000 <= 350
            ):
                self.current_gesture = GestureType.SCROLL
                self.state = ControlState.SCROLLING
                return []
            if self.config.entry_gate_enabled:
                # 进入门语义：握拳 = 退出控制态。
                # 退出判定必须豁免 CLICK/DRAG——捏合本身就会让手形偏离"张开"，
                # 若在点击/拖拽中把拳头当成退出信号，用户一捏合就掉出控制态，
                # 直接自锁。这些状态下拳头只冻结本次动作，不关闸门。
                self.current_gesture = GestureType.FIST
                if not (
                    self._left_active
                    or self._right_active
                    or self._left_button_down
                ):
                    if self._fist_started_ns is None:
                        self._fist_started_ns = now
                    held_ms = (now - self._fist_started_ns) / 1_000_000
                    if held_ms >= self.config.fist_hold_ms:
                        commands = self._release_left(now)
                        self._fist_started_ns = None
                        self._exit_gate()
                        return commands
                return []
            # 进入门未启用：保留"握拳切换暂停"的旧语义。
            commands = self._release_left(now)
            self._reset_transient()
            self.current_gesture = GestureType.FIST
            if self._fist_started_ns is None:
                self._fist_started_ns = now
            held_ms = (now - self._fist_started_ns) / 1_000_000
            if held_ms >= self.config.fist_hold_ms and not self._fist_latched:
                self._fist_latched = True
                commands.extend(self.set_paused(not self.paused, now))
            return commands

        self._fist_started_ns = None
        self._fist_latched = False
        if self.paused:
            self.state = ControlState.PAUSED
            self.current_gesture = GestureType.PAUSED
            return []

        commands: list[MouseCommand] = []
        left_now = left_ratio <= (
            self.config.pinch_release if self._left_active else self.config.pinch_engage
        )
        right_candidate = self._right_active or (
            self._right_ready
            and right_shape_valid
            and right_approach >= self.config.right_pinch_approach_delta
        )
        right_now = right_candidate and right_ratio <= (
            self.config.pinch_release
            if self._right_active
            else self.config.pinch_engage
        )

        # 已锁定的右键意图优先保持到松开，防止中指向拇指靠近时食指
        # 短暂经过拇指附近，造成一次右键动作被中途窃取成左键。
        if self._right_active:
            if right_now:
                self.current_gesture = GestureType.RIGHT_PINCH
                self._reset_orientation_tracking()
                return []
            self._right_active = False
            self._right_started_ns = None
            self._right_ready = right_ratio >= self.config.pinch_arm
            self._right_open_ratio = right_ratio if self._right_ready else 0.0
            self._right_requires_middle_extension = (
                self._right_ready and open_palm
            )
            self.current_gesture = GestureType.RIGHT_PINCH
            return [self._command(MouseCommandType.RIGHT_CLICK, now)]

        # 已开始的左键捏合保持原有点击/拖拽语义。
        if self._left_active:
            if not left_now:
                self._left_active = False
                self.current_gesture = GestureType.LEFT_PINCH
                if self._left_button_down:
                    commands.extend(self._release_left(now))
                elif self._left_started_ns is not None:
                    commands.append(self._command(MouseCommandType.LEFT_CLICK, now))
                self._left_started_ns = None
                self.state = ControlState.IDLE
                return commands
            self.current_gesture = (
                GestureType.DRAG if self._left_button_down else GestureType.LEFT_PINCH
            )
            self._reset_orientation_tracking()
            if self._left_started_ns is not None:
                held_ms = (now - self._left_started_ns) / 1_000_000
                if held_ms >= self.config.drag_hold_ms and not self._left_button_down:
                    commands.append(self._command(MouseCommandType.LEFT_DOWN, now))
                    self._left_button_down = True
                    self.state = ControlState.DRAGGING
            if self._left_button_down:
                x, y = self._cursor(points[8], now)
                commands.append(self._command(MouseCommandType.MOVE, now, x=x, y=y))
            return commands

        # 新右键只有在“先张开、再明显靠近”的时序成立时才允许进入。
        # 若同一帧两指都靠近，距离明显更近的目标指获胜；接近程度相同
        # 时仍保留左键优先，避免改变高频左键操作。
        right_wins = right_now and (
            not left_now or right_ratio + 0.08 < left_ratio
        )
        if right_wins:
            self.current_gesture = GestureType.RIGHT_PINCH
            self._reset_orientation_tracking()
            self._right_active = True
            self._right_started_ns = now
            self._right_ready = False
            self._right_requires_middle_extension = False
            return []

        if left_now:
            self.current_gesture = GestureType.LEFT_PINCH
            self._right_ready = False
            self._right_open_ratio = 0.0
            self._right_requires_middle_extension = False
            self._reset_orientation_tracking()
            self._left_active = True
            self._left_started_ns = now
            self.state = ControlState.PINCH_PENDING
            return []

        # 在真正接触阈值之前提前进入点击意图区，停止所有光标轨迹。
        # 这样用户将光标对准目标后，拇指靠近指尖的过程不会拖动点击位置。
        left_near = left_ratio <= self.config.pinch_arm
        right_near = (
            self._right_ready
            and right_shape_valid
            and right_ratio <= self.config.pinch_arm
            and right_approach >= self.config.right_pinch_approach_delta * 0.5
        )
        if left_near or right_near:
            self.current_gesture = GestureType.PINCH_ARMED
            self.state = ControlState.PINCH_PENDING
            self._reset_orientation_tracking()
            return []

        if open_palm:
            self._reset_pointer_rotation()
            self.current_gesture = GestureType.SCROLL
            self.state = ControlState.SCROLLING
            center = _palm_center(points)
            if self._open_anchor_x is None or self._scroll_anchor_y is None:
                self._open_anchor_x = center.x
                self._scroll_anchor_y = center.y
                self._rotation_tilt_anchor = palm_tilt
                self._rotation_last_open_ns = now
                return []
            self._rotation_last_open_ns = now
            movement_x = (center.x - self._open_anchor_x) / width
            movement_y = (center.y - self._scroll_anchor_y) / width
            horizontal = abs(movement_x)
            vertical = abs(movement_y)
            tilt_delta = palm_tilt - (
                self._rotation_tilt_anchor
                if self._rotation_tilt_anchor is not None
                else palm_tilt
            )
            if (
                self.config.page_gestures_enabled
                and not self._page_swipe_latched
                and horizontal >= self.config.page_swipe_threshold
                and horizontal > vertical * 1.35
            ):
                self._page_swipe_latched = True
                self.current_gesture = (
                    GestureType.PAGE_NEXT
                    if movement_x < 0
                    else GestureType.PAGE_PREVIOUS
                )
                kind = (
                    MouseCommandType.PAGE_NEXT
                    if movement_x < 0
                    else MouseCommandType.PAGE_PREVIOUS
                )
                self._rotation_tilt_anchor = palm_tilt
                return [self._command(kind, now)]
            if (
                self.config.rotation_gestures_enabled
                and not pinch_proximity
                and abs(tilt_delta) >= self.config.palm_rotation_threshold
            ):
                notches = round(
                    -tilt_delta * self.config.scroll_sensitivity * 6,
                )
                if notches == 0:
                    notches = -1 if tilt_delta > 0 else 1
                notches = max(-3, min(3, notches))
                self._rotation_tilt_anchor = palm_tilt
                self._rotation_scroll_active = True
                return [
                    self._command(
                        MouseCommandType.WHEEL,
                        now,
                        delta=notches * 120,
                    ),
                ]
            if (
                vertical >= self.config.scroll_threshold
                and vertical > horizontal * 1.15
            ):
                notches = max(
                    -3,
                    min(3, round(-movement_y * self.config.scroll_sensitivity * 4)),
                )
                if notches:
                    commands.append(
                        self._command(
                            MouseCommandType.WHEEL,
                            now,
                            delta=notches * 120,
                        ),
                    )
                self._open_anchor_x = center.x
                self._scroll_anchor_y = center.y
                self._rotation_tilt_anchor = palm_tilt
            return commands

        self._reset_scroll_tracking()
        # 走到这里说明MediaPipe已经给出完整的21点手部观测，并且该帧
        # 不是握拳、捏合或张掌滚动。移动模式不再额外要求食指维持
        # ≥135°的僵硬姿势；自然微弯、侧手或中指半伸时仍可控制光标。
        self.current_gesture = GestureType.POINTER
        self.state = ControlState.MOVING
        self._pointer_tilt_anchor = palm_tilt
        self._pointer_rotation_active = False
        x, y = self._cursor(points[8], now)
        commands.append(self._command(MouseCommandType.MOVE, now, x=x, y=y))
        return commands

    def tick(self, timestamp_ns: int | None = None) -> list[MouseCommand]:
        """检查手部数据超时并执行失联保护。"""

        now = timestamp_ns or monotonic_ns()
        if self.last_hand_ns is None:
            return []
        elapsed_ms = (now - self.last_hand_ns) / 1_000_000
        if elapsed_ms < self.config.hand_lost_ms or self._lost_reported:
            return []

        commands = self._release_left(now)
        self._reset_transient()
        self.filter.reset()
        self.state = ControlState.LOST
        self.current_gesture = GestureType.LOST
        self._lost_reported = True
        if VERBOSE_GATE:
            # 只在 lost 真正触发时打（_lost_reported 保证每 episode 一行）。
            # elapsed = dwell(取包延迟) + gap(消费后空转)，两段之和恒等于 elapsed_ms。
            arrival = self._last_hand_arrival_ns
            if arrival is None:
                _DIAG_LOGGER.info(
                    "lost-trigger now=%d last_hand_ns=%s elapsed_ms=%.0f "
                    "arrival=None（本 episode 未记录到达时刻，无法拆分）"
                    "flips=%d hand_lost_ms=%d",
                    now, self.last_hand_ns, elapsed_ms,
                    self._gate_diag_eligible_flips, self.config.hand_lost_ms,
                )
            else:
                _DIAG_LOGGER.info(
                    "lost-trigger now=%d last_hand_ns=%s elapsed_ms=%.0f "
                    "arrival=%d dwell_ms=%.0f gap_ms=%.0f flips=%d hand_lost_ms=%d",
                    now, self.last_hand_ns, elapsed_ms, arrival,
                    (arrival - self.last_hand_ns) / 1_000_000,
                    (now - arrival) / 1_000_000,
                    self._gate_diag_eligible_flips,
                    self.config.hand_lost_ms,
                )
        if self.config.entry_gate_enabled:
            # 手出画 ≥ hand_lost_ms 也是进入门的退出条件：重新入画后必须再
            # 做一次进入姿势，避免"手一晃出去又晃回来"直接恢复控制。
            self._gate_active = False
            self._entry_started_ns = None
        return commands
