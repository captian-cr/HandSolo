"""轨迹手势状态机测试。

覆盖：双指悬停进入 ARMED、绘制期间光标必须冻结、收笔出候选、捏合才确认执行，
以及超时/手部中断/功能关闭这些退化路径。
"""

from __future__ import annotations

import threading

from handmouse.config import AppConfig
from handmouse.contracts import GestureType, HandObservation, Landmark
from handmouse.gestures import GestureEngine
from handmouse.latest_slot import LatestSlot
from handmouse.motion_gestures import MotionGestureEngine, MotionState
from handmouse.mouse import MouseController, NullMouseBackend
from handmouse.pipeline import ControlWorker

DESIGN_W = [(0.12, 0.12), (0.31, 0.88), (0.50, 0.42), (0.69, 0.88), (0.88, 0.12)]


def _config(**overrides) -> AppConfig:
    values: dict[str, object] = {
        "motion_gestures_enabled": True,
        "motion_arm_hold_ms": 200,
        "motion_arm_max_drift": 0.05,
        "motion_min_duration_ms": 100,
        "motion_min_path_px": 20.0,
        "motion_confirm_timeout_ms": 1000,
        "motion_confirm_pinch": 0.42,
        "motion_accept_distance": 0.18,
    }
    values.update(overrides)
    config = AppConfig(**values)  # type: ignore[arg-type]
    config.validate()
    return config


# ---------------- 合成手部姿势 ----------------
def _two_finger_mapping(tip: tuple[float, float]) -> dict[int, tuple[float, float]]:
    """食指+中指伸出、无名指与小指收拢、拇指远离食指的"双指"姿势。

    食指由 ``(tx, ty)`` 往上排 8/7/6/5 四个点，保证三点共线（伸直）；
    中指同理，因此 ``_finger_extensions`` 会判为 (True, True, False, False)。
    """

    tx, ty = tip
    return {
        0: (tx + 0.08, ty + 0.50),
        1: (tx - 0.06, ty + 0.46), 2: (tx - 0.16, ty + 0.40),
        3: (tx - 0.20, ty + 0.34), 4: (tx - 0.24, ty + 0.30),
        5: (tx, ty + 0.30), 6: (tx, ty + 0.20), 7: (tx, ty + 0.10), 8: (tx, ty),
        9: (tx + 0.06, ty + 0.34), 10: (tx + 0.06, ty + 0.22),
        11: (tx + 0.06, ty + 0.12), 12: (tx + 0.06, ty + 0.02),
        13: (tx + 0.15, ty + 0.36), 14: (tx + 0.15, ty + 0.26),
        15: (tx + 0.16, ty + 0.30), 16: (tx + 0.17, ty + 0.36),
        17: (tx + 0.22, ty + 0.34), 18: (tx + 0.22, ty + 0.24),
        19: (tx + 0.23, ty + 0.28), 20: (tx + 0.24, ty + 0.34),
    }


def _to_landmarks(mapping: dict[int, tuple[float, float]]) -> tuple[Landmark, ...]:
    return tuple(Landmark(mapping[index][0], mapping[index][1], 0.0) for index in range(21))


def two_finger_observation(
    tip: tuple[float, float],
    timestamp_ns: int,
    *,
    pinch: bool = False,
) -> HandObservation:
    mapping = _two_finger_mapping(tip)
    if pinch:
        mapping[4] = mapping[8]
    return HandObservation(
        landmarks=_to_landmarks(mapping),
        handedness="Right",
        timestamp_ns=timestamp_ns,
        inference_ms=10.0,
        handedness_score=0.99,
    )


def pen_up_observation(timestamp_ns: int) -> HandObservation:
    """食指单伸、中指收拢——姿势消失即收笔。"""

    mapping = _two_finger_mapping((0.50, 0.40))
    mapping[12] = (mapping[10][0] + 0.01, mapping[10][1] + 0.08)
    return HandObservation(
        landmarks=_to_landmarks(mapping),
        handedness="Right",
        timestamp_ns=timestamp_ns,
        inference_ms=10.0,
        handedness_score=0.99,
    )


def partial_observation(timestamp_ns: int) -> HandObservation:
    """关键点不完整（局部续跟踪）的帧。"""

    return HandObservation(
        landmarks=_to_landmarks(_two_finger_mapping((0.50, 0.40))),
        handedness="Right",
        timestamp_ns=timestamp_ns,
        inference_ms=10.0,
        handedness_score=0.99,
        partial_finger="Index",
    )


def _densify(
    points: list[tuple[float, float]],
    steps: int = 12,
    scale: float = 0.30,
) -> list[tuple[float, float]]:
    dense: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        for step in range(steps):
            ratio = step / steps
            dense.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                ),
            )
    dense.append(points[-1])
    return [(0.5 + (x - 0.5) * scale, 0.45 + (y - 0.5) * scale) for x, y in dense]


# ---------------- IDLE → ARMED ----------------
def test_two_finger_hover_arms_after_hold_window() -> None:
    engine = MotionGestureEngine(_config(motion_arm_hold_ms=200))
    tip = (0.50, 0.45)
    start = 1_000_000_000

    assert engine.process(two_finger_observation(tip, start)).state is MotionState.IDLE
    assert engine.process(two_finger_observation(tip, start + 190_000_000)).state is MotionState.IDLE

    armed = engine.process(two_finger_observation(tip, start + 260_000_000))
    assert armed.state is MotionState.ARMED
    assert armed.gesture is GestureType.MOTION_ARMED
    assert armed.active


def test_moving_during_hover_restarts_the_hold_timer() -> None:
    """正在移动光标的手不能进入绘制模式。"""

    engine = MotionGestureEngine(_config(motion_arm_hold_ms=200))
    start = 1_000_000_000
    engine.process(two_finger_observation((0.50, 0.45), start))

    # 超过 arm_max_drift 的位移让计时归零。
    assert engine.process(two_finger_observation((0.60, 0.45), start + 220_000_000)).state is MotionState.IDLE
    assert engine.process(two_finger_observation((0.60, 0.45), start + 300_000_000)).state is MotionState.IDLE
    assert engine.process(two_finger_observation((0.60, 0.45), start + 480_000_000)).state is MotionState.ARMED


def test_dropping_the_pose_before_drawing_only_cancels() -> None:
    engine = MotionGestureEngine(_config())
    start = 1_000_000_000
    tip = (0.50, 0.45)
    engine.process(two_finger_observation(tip, start))
    engine.process(two_finger_observation(tip, start + 260_000_000))

    outcome = engine.process(pen_up_observation(start + 280_000_000))

    assert outcome.state is MotionState.IDLE
    assert not outcome.active


# ---------------- 完整链路 ----------------
def _run_until_candidate(engine: MotionGestureEngine, start: int = 1_000_000_000):
    """双指悬停 → 画 W → 收笔，返回候选阶段的 outcome 与收笔时间。"""

    tip = (0.50, 0.45)
    engine.process(two_finger_observation(tip, start))
    assert engine.process(two_finger_observation(tip, start + 220_000_000)).state is MotionState.ARMED

    now = start + 240_000_000
    for x, y in _densify(DESIGN_W):
        now += 16_000_000
        engine.process(two_finger_observation((x, y), now))

    now += 16_000_000
    return engine.process(pen_up_observation(now)), now


def test_drawn_w_becomes_candidate_and_requires_pinch_to_execute() -> None:
    engine = MotionGestureEngine(_config())

    candidate, now = _run_until_candidate(engine)

    assert candidate.state is MotionState.CANDIDATE
    assert candidate.pending == "W"
    assert [item.name for item in candidate.candidates][:1] == ["W"]
    assert candidate.candidates[0].distance < 0.18
    assert candidate.action is None, "候选阶段绝不能自动执行"

    confirmed = engine.process(two_finger_observation((0.50, 0.45), now + 100_000_000, pinch=True))

    assert confirmed.state is MotionState.IDLE
    assert confirmed.action is not None
    assert confirmed.action.kind.value == "launch"
    assert confirmed.action.target == "notepad"
    assert confirmed.action.name == "W"


def test_candidate_is_dropped_when_confirmation_times_out() -> None:
    engine = MotionGestureEngine(_config(motion_confirm_timeout_ms=1000))
    _, now = _run_until_candidate(engine)

    assert engine.tick(now + 500_000_000).state is MotionState.CANDIDATE

    timed_out = engine.tick(now + 1_100_000_000)

    assert timed_out.state is MotionState.IDLE
    assert "超时" in timed_out.message
    assert timed_out.action is None


def test_losing_hand_during_drawing_discards_the_trajectory() -> None:
    engine = MotionGestureEngine(_config(motion_hand_lost_ms=200))
    start = 1_000_000_000
    tip = (0.50, 0.45)
    engine.process(two_finger_observation(tip, start))
    engine.process(two_finger_observation(tip, start + 220_000_000))
    engine.process(two_finger_observation((0.60, 0.50), start + 240_000_000))

    assert engine.state is MotionState.DRAWING

    lost = engine.tick(start + 240_000_000 + 300_000_000)

    assert lost.state is MotionState.IDLE
    assert "作废" in lost.message


def test_partial_landmarks_abort_drawing_without_producing_candidates() -> None:
    engine = MotionGestureEngine(_config())
    start = 1_000_000_000
    tip = (0.50, 0.45)
    engine.process(two_finger_observation(tip, start))
    engine.process(two_finger_observation(tip, start + 220_000_000))
    engine.process(two_finger_observation((0.60, 0.50), start + 240_000_000))

    aborted = engine.process(partial_observation(start + 260_000_000))

    assert aborted.state is MotionState.IDLE
    assert aborted.candidates == ()


def test_too_short_trajectory_is_rejected_before_matching() -> None:
    engine = MotionGestureEngine(_config(motion_min_path_px=200.0))
    start = 1_000_000_000
    tip = (0.50, 0.45)
    engine.process(two_finger_observation(tip, start))
    engine.process(two_finger_observation(tip, start + 220_000_000))
    engine.process(two_finger_observation((0.52, 0.46), start + 240_000_000))

    rejected = engine.process(pen_up_observation(start + 400_000_000))

    assert rejected.state is MotionState.IDLE
    assert "太短" in rejected.message
    assert rejected.candidates == ()


def test_disabled_motion_gestures_stay_idle() -> None:
    engine = MotionGestureEngine(_config(motion_gestures_enabled=False))

    outcome = engine.process(two_finger_observation((0.50, 0.45), 1_000_000_000))

    assert outcome.state is MotionState.IDLE
    assert not outcome.active


# ---------------- 与流水线的协作 ----------------
def test_drawing_freezes_cursor_injection() -> None:
    """ARMED/DRAWING 期间不能再注入任何 MOVE，否则光标会跟着笔画拖影。"""

    config = _config()
    backend = NullMouseBackend()
    engine = GestureEngine(config)
    motion = MotionGestureEngine(config)
    worker = ControlWorker(
        LatestSlot(),
        engine,
        MouseController(backend),
        threading.Event(),
        lambda _gesture, _paused: None,
        interpolation_ms=25,
        motion=motion,
        on_motion=lambda _outcome: None,
    )

    tip = (0.50, 0.45)
    start = 1_000_000_000
    worker._handle_observation(two_finger_observation(tip, start))
    worker._advance_motion(start + 10_000_000)
    moves_before = len([item for item in backend.commands if item[0] == "move"])
    assert moves_before > 0, "进入绘制模式前，光标应当正常跟随"

    worker._handle_observation(two_finger_observation(tip, start + 260_000_000))
    assert worker.motion_state is MotionState.ARMED

    for offset, point in enumerate(_densify(DESIGN_W, steps=6), start=1):
        stamp = start + 260_000_000 + offset * 16_000_000
        worker._handle_observation(two_finger_observation(point, stamp))
        worker._advance_motion(stamp + 8_000_000)

    assert worker.motion_state is MotionState.DRAWING
    assert worker.motion_outcome is not None and worker.motion_outcome.active
    assert len([item for item in backend.commands if item[0] == "move"]) == moves_before


def test_leaving_motion_mode_resets_cursor_state() -> None:
    """绘制期间滤波器与插值器都没有输入，恢复控制时不能吐出旧轨迹。"""

    config = _config()
    worker = ControlWorker(
        LatestSlot(),
        GestureEngine(config),
        MouseController(NullMouseBackend()),
        threading.Event(),
        lambda _gesture, _paused: None,
        interpolation_ms=25,
    )
    now = 1_000_000_000
    worker._enter_motion_mode()
    assert worker._motion_mode

    pointer_filter = worker.engine.filter
    pointer_filter.update(0.10, 0.10, now)
    pointer_filter.update(0.90, 0.90, now + 33_000_000)
    lagged = pointer_filter.update(0.20, 0.20, now + 66_000_000)
    assert lagged[0] > 0.20, "有历史时 One Euro 输出应当被滞后"

    worker._leave_motion_mode()

    assert not worker._motion_mode
    # 重置后第一次更新必须原样透传，否则恢复控制的瞬间光标会跳。
    assert pointer_filter.update(0.30, 0.30, now + 99_000_000) == (0.30, 0.30)
    assert worker.motion.sample(now + 120_000_000) is None
