"""轨迹手势状态机：IDLE → ARMED → DRAWING → CANDIDATE → 执行。

为什么需要独立状态机
--------------------
Vimoose 的手势状态机是"姿态即时触发"（捏合=点击），而画字母是"先进入一种
模式、画完再确认"。两者混在一个状态机里会互相污染：绘制过程中任何一次
捏合都会被解释成左键点击。所以这里独立成一个引擎，并遵守两条硬约束：

1. **ARMED / DRAWING / CANDIDATE 期间光标注入必须冻结**——由 ``pipeline``
   保证：处于这些状态时不把观测交给鼠标状态机。
2. **只有捏合才能确认执行**——候选算出来后不会自动执行，必须重新捏合，
   避免画错一笔就误触发。

进入方式
--------
食指+中指同时伸出、其余手指收拢、拇指不靠近食指（即"手枪/双指"姿势），
并且**指尖原地悬停** ``motion_arm_hold_ms``。要求原地悬停是因为光标控制
本身也允许中指自然半伸，只靠姿势会误触发；加上"没在移动"这个条件后，
正常移动光标的手不会进入绘制模式。

收笔由姿势消失触发，候选阶段等待 ``motion_confirm_timeout_ms`` 后自动作废。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from math import hypot
from time import monotonic_ns

from handmouse.config import AppConfig
from handmouse.contracts import ActionCommand, ActionType, GestureType, HandObservation
from handmouse.dtw import (
    MatchCandidate,
    TrajectoryMatcher,
    build_default_matcher,
    path_length,
)
from handmouse.gestures import _distance_2d, _finger_extensions, _palm_width

Point = tuple[float, float]

# 诊断开关：环境变量控制，与 gestures/vision 共用同一开关（默认关闭）。
VERBOSE_GATE = os.environ.get("HANDMOUSE_VERBOSE_GATE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_DIAG_LOGGER = logging.getLogger("handmouse.gate-diag")


class MotionState(str, Enum):
    """轨迹手势内部状态。"""

    IDLE = "idle"
    ARMED = "armed"
    DRAWING = "drawing"
    CANDIDATE = "candidate"


_STATE_GESTURE = {
    MotionState.IDLE: GestureType.NONE,
    MotionState.ARMED: GestureType.MOTION_ARMED,
    MotionState.DRAWING: GestureType.MOTION_DRAWING,
    MotionState.CANDIDATE: GestureType.MOTION_CANDIDATE,
}


@dataclass(slots=True)
class MotionOutcome:
    """一帧的处理结果。"""

    state: MotionState
    gesture: GestureType
    active: bool
    candidates: tuple[MatchCandidate, ...] = ()
    pending: str = ""
    action: ActionCommand | None = None
    message: str = ""
    drawing: tuple[Point, ...] = ()


def _idle_outcome(message: str = "") -> MotionOutcome:
    return MotionOutcome(state=MotionState.IDLE, gesture=GestureType.NONE, active=False, message=message)


class MotionGestureEngine:
    """把手指轨迹变成"候选 + 确认"的轨迹手势引擎。"""

    def __init__(
        self,
        config: AppConfig,
        matcher: TrajectoryMatcher | None = None,
    ) -> None:
        self.config = config
        self.matcher = matcher or build_default_matcher(config.motion_template_points)
        self.state = MotionState.IDLE
        self.drawing: list[Point] = []
        self.candidates: tuple[MatchCandidate, ...] = ()
        self.pending: str = ""

        self._arm_started_ns: int | None = None
        self._arm_anchor: Point | None = None
        self._draw_started_ns: int | None = None
        self._last_point_ns: int | None = None
        self._candidate_since_ns: int | None = None
        self._last_hand_ns: int | None = None
        # 诊断用：最近一次观测被 process() 消费的墙钟时刻（到达时刻戳）。
        # 笔2 之后 _last_hand_ns 也改用到达墙钟，因此本字段与 _last_hand_ns 恒等
        # （dwell 段恒为 0），tick() 的 elapsed 不再含"取包延迟"段；取包延迟改由
        # pipeline 的 [采样] latency_ms 观察。字段保留以便将来"到达 vs 采集"差
        # 诊断复用，仅在 VERBOSE_GATE 下写入。
        self._last_hand_arrival_ns: int | None = None
        # 本帧到达墙钟，process() 入口处打一次，供 _finish_drawing 设置
        # _candidate_since_ns 时复用，保证与 _last_hand_ns 同源。
        self._arrival_ns: int = 0

    def update_config(self, config: AppConfig) -> None:
        self.config = config
        self.matcher = build_default_matcher(config.motion_template_points)

    @property
    def active(self) -> bool:
        return self.state is not MotionState.IDLE

    def reset(self, message: str = "") -> MotionOutcome:
        self.state = MotionState.IDLE
        self.drawing = []
        self.candidates = ()
        self.pending = ""
        self._arm_started_ns = None
        self._arm_anchor = None
        self._draw_started_ns = None
        self._last_point_ns = None
        self._candidate_since_ns = None
        return _idle_outcome(message)

    # ---------------- 姿态判定 ----------------
    @staticmethod
    def _two_finger_pose(points, pinch_ratio: float, thumb_away: float) -> bool:
        """食指+中指伸出、无名指与小指收拢、拇指不贴近食指。

        ARMED 进入条件（H6 硬线判定依据）：
        - index=True, middle=True（食指+中指伸出，关节角 ≥ 135°）
        - ring=False, pinky=False（无名指+小指收拢）
        - pinch_ratio > thumb_away（拇指-食指距离 > pinch_arm，默认 0.72 掌宽）
          → 拇指必须与食指**分开**，不是捏合；捏合时 pinch_ratio < 0.72 不会进 ARMED
        - partial_finger=None（手部关键点完整，否则 process() 直接走 tick()）
        - drift ≤ motion_arm_max_drift（默认 0.05，食指尖原地悬停）
        - held_ms ≥ motion_arm_hold_ms（默认 250ms）

        ``thumb_away`` 复用 ``pinch_arm``：比它更近就说明用户正准备捏合，
        不该理解成"要开始画画"。
        """

        index, middle, ring, pinky = _finger_extensions(points)
        return index and middle and not ring and not pinky and pinch_ratio > thumb_away

    # ---------------- 主入口 ----------------
    def process(
        self,
        observation: HandObservation,
        timestamp_ns: int | None = None,
    ) -> MotionOutcome:
        # 到达墙钟：本帧交付到控制线程的时刻。tick() 的 hand_lost / confirm_timeout
        # 判定用的是墙钟，_last_hand_ns 与 _candidate_since_ns 必须记在同一时钟上；
        # 若沿用 now（采集时刻戳），elapsed = 墙钟 now - 采集时刻 会把"取包延迟 +
        # 帧间隔"算成经过时间，逼近 motion_hand_lost_ms 时假性作废轨迹。
        self._arrival_ns = monotonic_ns()
        now = timestamp_ns or observation.timestamp_ns
        if not self.config.motion_gestures_enabled:
            if self.active:
                return self.reset("轨迹手势已关闭")
            return _idle_outcome()

        points = observation.landmarks
        if len(points) != 21 or observation.partial_finger:
            # 关键点不完整时无法判定姿势。绘制中遇到这种情况直接作废轨迹：
            # 半截轨迹算出来的候选一定是错的，宁可让用户重画。
            if self.state in (MotionState.ARMED, MotionState.DRAWING):
                return self.reset("手部数据中断，已作废本次轨迹")
            return self.tick(now)

        self._last_hand_ns = self._arrival_ns
        if VERBOSE_GATE:
            self._last_hand_arrival_ns = self._arrival_ns
        width = _palm_width(points)
        pinch_ratio = _distance_2d(points[4], points[8]) / width
        tip = (float(points[8].x), float(points[8].y))
        two_finger = self._two_finger_pose(points, pinch_ratio, self.config.pinch_arm)
        if VERBOSE_GATE:
            index, middle, ring, pinky = _finger_extensions(points)
            _DIAG_LOGGER.info(
                "motion-arm two_finger=%s index=%s middle=%s ring=%s pinky=%s "
                "pinch_ratio=%.3f pinch_arm=%.3f partial_finger=%s state=%s",
                two_finger, index, middle, ring, pinky,
                pinch_ratio, self.config.pinch_arm,
                observation.partial_finger or "None", self.state.value,
            )

        if self.state is MotionState.IDLE:
            return self._step_idle(now, tip, two_finger)
        if self.state is MotionState.ARMED:
            return self._step_armed(now, tip, two_finger)
        if self.state is MotionState.DRAWING:
            return self._step_drawing(now, tip, two_finger)
        return self._step_candidate(now, pinch_ratio)

    def tick(self, timestamp_ns: int | None = None) -> MotionOutcome:
        """没有新观测时调用：处理确认超时与手部失联作废。"""

        now = timestamp_ns or monotonic_ns()
        if self.state is MotionState.CANDIDATE:
            if self._candidate_since_ns is not None and (
                now - self._candidate_since_ns
            ) / 1_000_000 >= self.config.motion_confirm_timeout_ms:
                if VERBOSE_GATE:
                    _DIAG_LOGGER.info(
                        "motion-trigger kind=confirm_timeout now=%d candidate_since_ns=%d "
                        "elapsed_ms=%.0f timeout_ms=%d",
                        now, self._candidate_since_ns,
                        (now - self._candidate_since_ns) / 1_000_000,
                        self.config.motion_confirm_timeout_ms,
                    )
                return self.reset("确认超时，已丢弃本次轨迹")
            return self._candidate_outcome("捏合确认或等待超时")

        if self.state in (MotionState.ARMED, MotionState.DRAWING):
            if self._last_hand_ns is not None and (
                now - self._last_hand_ns
            ) / 1_000_000 >= self.config.motion_hand_lost_ms:
                if VERBOSE_GATE:
                    # 与 gestures.tick() 同一类清零机制，独立取证。
                    arrival = self._last_hand_arrival_ns
                    if arrival is None:
                        _DIAG_LOGGER.info(
                            "motion-trigger kind=hand_lost now=%d last_hand_ns=%d "
                            "elapsed_ms=%.0f arrival=None hand_lost_ms=%d state=%s",
                            now, self._last_hand_ns,
                            (now - self._last_hand_ns) / 1_000_000,
                            self.config.motion_hand_lost_ms, self.state.value,
                        )
                    else:
                        _DIAG_LOGGER.info(
                            "motion-trigger kind=hand_lost now=%d last_hand_ns=%d "
                            "elapsed_ms=%.0f arrival=%d dwell_ms=%.0f gap_ms=%.0f "
                            "hand_lost_ms=%d state=%s",
                            now, self._last_hand_ns,
                            (now - self._last_hand_ns) / 1_000_000,
                            arrival,
                            (arrival - self._last_hand_ns) / 1_000_000,
                            (now - arrival) / 1_000_000,
                            self.config.motion_hand_lost_ms, self.state.value,
                        )
                return self.reset("手部丢失，已作废本次轨迹")
            return self._armed_outcome("保持双指姿势开始绘制")

        return _idle_outcome()

    # ---------------- 各状态 ----------------
    def _step_idle(self, now: int, tip: Point, two_finger: bool) -> MotionOutcome:
        if not two_finger:
            self._arm_started_ns = None
            self._arm_anchor = None
            return _idle_outcome()

        if self._arm_started_ns is None or self._arm_anchor is None:
            self._arm_started_ns = now
            self._arm_anchor = tip
            return _idle_outcome("双指悬停以进入绘制")

        drift = hypot(tip[0] - self._arm_anchor[0], tip[1] - self._arm_anchor[1])
        if drift > self.config.motion_arm_max_drift:
            # 手指在动说明用户还在控制光标，不是准备画画，重新计时。
            if VERBOSE_GATE:
                _DIAG_LOGGER.info(
                    "motion-arm drift=%.4f max_drift=%.4f → reset (finger moving)",
                    drift, self.config.motion_arm_max_drift,
                )
            self._arm_started_ns = now
            self._arm_anchor = tip
            return _idle_outcome()

        held_ms = (now - self._arm_started_ns) / 1_000_000
        if held_ms < self.config.motion_arm_hold_ms:
            if VERBOSE_GATE:
                _DIAG_LOGGER.info(
                    "motion-arm held_ms=%.0f hold_ms=%d → waiting",
                    held_ms, self.config.motion_arm_hold_ms,
                )
            return _idle_outcome()

        self.state = MotionState.ARMED
        self.drawing = []
        self.candidates = ()
        self.pending = ""
        return self._armed_outcome("已就绪：开始画字母或形状")

    def _step_armed(self, now: int, tip: Point, two_finger: bool) -> MotionOutcome:
        if not two_finger:
            # 只是点了一下姿势就放下，不算绘制。
            return self.reset("已取消")
        self.state = MotionState.DRAWING
        self.drawing = [tip]
        self._draw_started_ns = now
        self._last_point_ns = now
        return self._drawing_outcome()

    def _step_drawing(self, now: int, tip: Point, two_finger: bool) -> MotionOutcome:
        if two_finger:
            self.drawing.append(tip)
            self._last_point_ns = now
            return self._drawing_outcome()
        return self._finish_drawing(now)

    def _finish_drawing(self, now: int) -> MotionOutcome:
        duration_ms = (
            (now - self._draw_started_ns) / 1_000_000
            if self._draw_started_ns is not None
            else 0.0
        )
        min_length = self.config.motion_min_path_px / max(1, self.config.camera_width)
        length = path_length(self.drawing)
        if duration_ms < self.config.motion_min_duration_ms or length < min_length:
            return self.reset("轨迹太短，已丢弃")

        candidates = self.matcher.match(self.drawing, self.config.motion_max_candidates)
        best = candidates[0] if candidates else None
        if best is None or best.distance > self.config.motion_accept_distance:
            return self.reset("没有匹配到的形状，已丢弃")

        self.candidates = tuple(candidates)
        self.pending = best.name
        self._candidate_since_ns = self._arrival_ns
        self.state = MotionState.CANDIDATE
        return self._candidate_outcome("捏合确认执行")

    def _step_candidate(self, now: int, pinch_ratio: float) -> MotionOutcome:
        if pinch_ratio > self.config.motion_confirm_pinch:
            return self.tick(now)

        binding = self.config.motion_bindings.get(self.pending)
        if binding is None:
            name = self.pending
            self.reset()
            return _idle_outcome(f"{name} 没有配置动作绑定")

        kind = ActionType.LAUNCH if binding["kind"] == "launch" else ActionType.HOTKEY
        action = ActionCommand(
            kind=kind,
            timestamp_ns=now,
            name=self.pending,
            target=binding["target"],
        )
        name = self.pending
        self.reset()
        return MotionOutcome(
            state=MotionState.IDLE,
            gesture=GestureType.NONE,
            active=False,
            action=action,
            message=f"已确认 {name}",
        )

    # ---------------- 结果封装 ----------------
    def _armed_outcome(self, message: str) -> MotionOutcome:
        return MotionOutcome(
            state=self.state,
            gesture=_STATE_GESTURE[self.state],
            active=True,
            message=message,
            drawing=tuple(self.drawing),
        )

    def _drawing_outcome(self) -> MotionOutcome:
        return MotionOutcome(
            state=MotionState.DRAWING,
            gesture=GestureType.MOTION_DRAWING,
            active=True,
            message=f"绘制中（{len(self.drawing)} 点）",
            drawing=tuple(self.drawing),
        )

    def _candidate_outcome(self, message: str) -> MotionOutcome:
        label = " / ".join(str(candidate) for candidate in self.candidates)
        return MotionOutcome(
            state=MotionState.CANDIDATE,
            gesture=GestureType.MOTION_CANDIDATE,
            active=True,
            candidates=self.candidates,
            pending=self.pending,
            message=f"{message}：{label}" if label else message,
            drawing=tuple(self.drawing),
        )
