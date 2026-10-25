"""「捏合趋向 → 自动钉住光标」的行为约定。

背景（2026-10-21 用户要求）：拇指张开的手动"停住"因为侧手投影下判据不可分
已经关闭（见 `gestures._THUMB_SPREAD_STOP` 注释）。替代方案是——瞄准好之后
自然捏合，**在捏合把光标带偏之前就冻住**，这样落点不会漂。

阈值 0.70 由 521 帧实测标定（`logs/pinch-anchor-20261021-150447.csv`）：
向下移动全程仅 1 帧低于 0.70，且那帧光标已钳在屏幕底。
"""

from __future__ import annotations

import pytest

from handmouse.config import AppConfig
from handmouse.contracts import HandObservation, Landmark, MouseCommandType
from handmouse.gestures import GestureEngine

from conftest import pointer_points

BASE = 1_000_000_000


def _observation(points, timestamp_ns: int = BASE) -> HandObservation:
    return HandObservation(
        landmarks=tuple(points),
        handedness="Right",
        timestamp_ns=timestamp_ns,
        inference_ms=1.0,
    )


def _thumb_at_distance_from_tip(points, ratio: float):
    """把拇指尖放到离食指尖 ``ratio × 掌宽`` 的位置，从而精确设定捏合比。"""

    values = list(points)
    tip = values[8]
    palm = (
        (values[5].x - values[17].x) ** 2 + (values[5].y - values[17].y) ** 2
    ) ** 0.5
    values[4] = Landmark(tip.x + ratio * palm, tip.y)
    return values


@pytest.fixture
def engine_no_thumb_stop() -> GestureEngine:
    config = AppConfig()
    config.thumb_stop_enabled = False
    config.pinch_anchor_ratio = 0.70
    return GestureEngine(config)


def _confirm(engine, points, frames: int = 3) -> None:
    """喂够"连续帧确认"所需的帧数。

    2026-10-22 起锚定需要 ``_PINCH_ANCHOR_FRAMES``(**3**) 帧连续成立：
    单帧触发会让"指向前方"被随机冻 1 秒（他 701 帧里 move 的捏合比 p5 只有
    0.47，实测移动时有 30.6% 的时间被冻住）。
    """

    for i in range(frames):
        engine._allow_cursor_move(_observation(points, BASE + i * 33_000_000))


def test_single_frame_below_threshold_does_not_anchor(
    engine_no_thumb_stop: GestureEngine,
) -> None:
    """单帧掠过阈值不许锚定 —— 否则指向前方时会被随机冻住 1 秒。"""

    points = _thumb_at_distance_from_tip(pointer_points(), 0.58)
    engine = engine_no_thumb_stop

    engine._allow_cursor_move(_observation(points))

    assert engine.pinch_anchor_active is False


def test_pinch_approach_anchors_cursor_without_thumb_stop(
    engine_no_thumb_stop: GestureEngine,
) -> None:
    """捏合比连续 3 帧进入过渡带（<=0.70）就提前钉住，不依赖 thumb_stop。"""

    points = _thumb_at_distance_from_tip(pointer_points(), 0.58)
    engine = engine_no_thumb_stop
    _confirm(engine, points)

    allowed = engine._allow_cursor_move(_observation(points, BASE + 200_000_000))

    assert engine.pinch_anchor_active is True, "应已进入锚定窗口"
    assert allowed is False, "锚定期间必须冻结 MOVE"


def test_open_hand_is_not_anchored(engine_no_thumb_stop: GestureEngine) -> None:
    """正常指向（拇指远离食指）绝不能触发锚定，否则一放下手就冻。"""

    engine = engine_no_thumb_stop
    points = pointer_points()  # 捏合比约 1.35
    allowed = engine._allow_cursor_move(_observation(points))

    assert engine.pinch_anchor_active is False
    assert allowed is True


def test_anchor_suppressed_when_fingertip_leaves_frame(
    engine_no_thumb_stop: GestureEngine,
) -> None:
    """指尖被画面边缘切掉时几何不可信，不该锚定。

    ⚠️ 注意：判据是"指尖在不在画面内"，**不是"光标是否已到屏幕边缘"**。
    早期版本用的是后者，结果用户点屏幕下部的应用时锚定永远不生效
    （底部应用恰恰就在屏幕边缘），2026-10-21 实测后改掉。
    """

    points = _thumb_at_distance_from_tip(pointer_points(), 0.58)
    points[8] = Landmark(points[8].x, 1.20)  # 指尖已跑到画面外
    points[4] = Landmark(points[8].x + 0.10, 1.20)

    engine = engine_no_thumb_stop
    _confirm(engine, points)

    assert engine.pinch_anchor_active is False


def test_anchor_still_works_at_screen_bottom(
    engine_no_thumb_stop: GestureEngine,
) -> None:
    """光标已经贴在屏幕下沿时仍要锚定 —— 底部应用就在那里点。"""

    config = engine_no_thumb_stop.config
    left, top, right, bottom = config.effective_margins()
    # 选一个映射后 >= 1.0 的指尖位置（光标会被钳在屏幕底），但不越出画面。
    tip_y = min(0.98, 1.0 - bottom + 0.05)
    points = _thumb_at_distance_from_tip(pointer_points(), 0.58)
    points[8] = Landmark(0.5, tip_y)
    points[4] = Landmark(0.5 + 0.10, tip_y)

    engine = engine_no_thumb_stop
    _confirm(engine, points)

    assert engine.pinch_anchor_active is True, "屏幕下沿也必须能锚定"


def test_engaged_pinch_without_thumb_stop_still_freezes(
    engine_no_thumb_stop: GestureEngine,
) -> None:
    """捏到底（<= pinch_release）时即便 thumb_stop 关闭也不能让光标乱跑。"""

    points = _thumb_at_distance_from_tip(pointer_points(), 0.30)
    engine = engine_no_thumb_stop
    _confirm(engine, points)

    allowed = engine._allow_cursor_move(_observation(points, BASE + 200_000_000))

    assert allowed is False
    assert engine.cursor_frozen is True


def test_drag_is_never_frozen_by_pinch_anchor(
    engine_no_thumb_stop: GestureEngine,
) -> None:
    """按着左键拖拽时必须放行，否则拖不动。"""

    points = _thumb_at_distance_from_tip(pointer_points(), 0.30)
    engine = engine_no_thumb_stop
    engine._left_button_down = True

    assert engine._allow_cursor_move(_observation(points)) is True
    assert engine.cursor_frozen is False


def test_thumb_stop_enabled_keeps_previous_behaviour() -> None:
    """thumb_stop 打开时的旧路径不能被新逻辑改坏。"""

    config = AppConfig()
    config.thumb_stop_enabled = True
    engine = GestureEngine(config)

    allowed = engine._allow_cursor_move(_observation(pointer_points()))

    assert allowed is True


# ---------------------------------------------------------------------------
# 「朝下 + 食指弯 → 不算捏合」这条抑制（pinch_down_suppress_enabled）
# ---------------------------------------------------------------------------


def _downward_pinch_points():
    """食指尖明显低于食指 MCP，同时拇指贴上去 —— 用户真实捏合的一种姿态。"""

    points = list(pointer_points())
    points[5] = Landmark(0.42, 0.60)   # 食指 MCP
    points[8] = Landmark(0.42, 0.78)   # 食指尖朝下
    points[4] = Landmark(0.42, 0.78)   # 拇指贴到食指尖
    return tuple(points)


def _gesture_after_frames(flag: bool) -> str:
    config = AppConfig()
    config.entry_gate_enabled = False
    config.pinch_down_suppress_enabled = flag
    engine = GestureEngine(config)
    engine.set_paused(False)
    observation = _observation(_downward_pinch_points())
    for _ in range(4):
        engine.process(observation)
    return engine.current_gesture.value


def test_downward_pinch_is_suppressed_by_default() -> None:
    """默认保留抑制：朝下 + 食指弯不算捏合（避免指向下方时误点）。"""

    assert _gesture_after_frames(True) == "pointer"


def test_suppression_can_be_disabled_for_side_hands() -> None:
    """关掉抑制后，同一帧必须识别成捏合。

    2026-10-21 实测：该用户 HUD 显示捏合比 0.32（已低于 engage 0.42），
    状态却仍是「正在控制光标」—— 就是这条抑制把他的真捏合挡掉了
    （他的侧手投影让 `_finger_extensions` 把弯食指判成伸直、朝前判成朝下）。
    """

    assert _gesture_after_frames(False) == "left_pinch"


# ---------------------------------------------------------------------------
# 「停留满 1 秒 = 我要捏合」（2026-10-22 用户选定方案，与捏合比并联）
# ---------------------------------------------------------------------------


def _still(engine, clock, points, *, still_ms: int = 1100) -> None:
    """模拟"原地静止满 still_ms"。

    停留计时读的是 `handmouse.gestures.monotonic_ns`，而 conftest 里有个
    **autouse** 的 `ArrivalClock` 把它换成了冻结在 0 的可控时钟——所以这里
    必须用同一个时钟往前拨，不能读 `time.monotonic_ns()`（那会算出负的
    经过时间，正是我第一次写错时的现象）。
    """

    engine.process(_observation(points, BASE))  # 第一帧：停留计时从时钟当前值起算
    clock.advance_ms(still_ms)
    engine.process(_observation(points, BASE + 100_000_000))  # 第二帧：判定"已经停了 1.1 秒"


def test_dwell_alone_freezes_cursor_with_fingers_wide_open(arrival_clock) -> None:
    """手指完全没合拢，只要**静止满 1 秒**也要冻住光标。

    这是"无论有没有磁吸，停留约一秒就说明我要捏合"这条路的验收：
    它不依赖捏合比，所以不会被"指向前方时拇指天然贴着食指"骗到。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    config.thumb_stop_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    points = _thumb_at_distance_from_tip(pointer_points(), 1.30)  # 手指大方张开
    _still(engine, arrival_clock, points)

    assert engine.cursor_frozen is True, "静止满 1 秒却没冻住光标"


def test_dwell_freeze_lifts_as_soon_as_he_moves_again(arrival_clock) -> None:
    """一挪开就立刻解冻——停留只是"停住时别抖"，不是"停住就锁死"。"""

    from conftest import translate_points

    config = AppConfig()
    config.entry_gate_enabled = False
    config.thumb_stop_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    points = _thumb_at_distance_from_tip(pointer_points(), 1.30)
    _still(engine, arrival_clock, points)
    assert engine.cursor_frozen is True

    moved = _thumb_at_distance_from_tip(
        translate_points(pointer_points(), dx=0.05), 1.30,
    )
    engine.process(_observation(moved, BASE + 200_000_000))

    assert engine.cursor_frozen is False, "他明明动了，光标却没解冻"


def test_dwell_never_blocks_clicks(arrival_clock) -> None:
    """停留冻结只作用于移动，不能把捏合点击一并吞掉。"""

    config = AppConfig()
    config.entry_gate_enabled = False
    config.thumb_stop_enabled = False
    config.click_immediate = True
    engine = GestureEngine(config)
    engine.set_paused(False)

    points = _thumb_at_distance_from_tip(pointer_points(), 1.30)
    _still(engine, arrival_clock, points)
    assert engine.cursor_frozen is True

    tight = _thumb_at_distance_from_tip(pointer_points(), 0.30)
    engine.process(_observation(tight, BASE + 300_000_000))
    released = engine.process(_observation(points, BASE + 500_000_000))

    assert MouseCommandType.LEFT_CLICK in [c.kind for c in released], "停留冻结把点击吞了"


def test_dwell_freeze_can_be_switched_off_without_closing_snap_window(
    arrival_clock,
) -> None:
    """「停留冻结光标」与「停留开磁吸窗口」必须能分别关闭。

    2026-10-23：dwell 判据修好之前它**一次都没亮过**，所以"停住一秒就冻光标"
    这个 2026-10-22 加的行为，用户其实是第一次真的体验到，很可能觉得
    "我只是停了一下，光标怎么就冻了"。`pinch_pre_dwell_freeze=False` 只该
    关掉冻结——磁吸窗口必须照旧按停留打开，否则等于顺手把磁吸也废了。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    config.thumb_stop_enabled = False
    config.pinch_pre_dwell_freeze = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    points = _thumb_at_distance_from_tip(pointer_points(), 1.30)
    _still(engine, arrival_clock, points)

    assert engine.dwell_active is True, "停留本身仍然要亮（它决定磁吸窗口）"
    assert engine.snap_engaged is True, "关掉的只是冻结，磁吸窗口必须照开"
    assert engine.cursor_frozen is False, "冻结已关闭，光标必须能动"



