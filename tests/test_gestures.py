from handmouse.config import AppConfig
from handmouse.contracts import HandObservation, Landmark, MouseCommandType
from handmouse.gestures import GestureEngine

from conftest import (
    ambiguous_right_pinch_points,
    entry_back_points,
    entry_palm_points,
    fist_like_left_pinch_points,
    fist_points,
    left_pinch_points,
    middle_tucked_pointer_points,
    observation,
    open_palm_points,
    pointer_points,
    right_pinch_from_open_points,
    right_pinch_points,
    rotated_projected_closed_points,
    rotated_scroll_points,
    side_palm_points,
    translate_points,
)


BASE = 1_000_000_000


def kinds(commands):
    return [command.kind for command in commands]


def test_pointer_generates_move() -> None:
    engine = GestureEngine(AppConfig())
    commands = engine.process(observation(pointer_points(), BASE))
    assert kinds(commands) == [MouseCommandType.MOVE]
    assert 0 <= commands[0].x <= 1
    assert 0 <= commands[0].y <= 1


def test_pointer_reaches_screen_edge_before_landmark_leaves_camera() -> None:
    """默认安全区内的食指位置应已能覆盖屏幕最右侧。"""

    points = list(pointer_points())
    points[6] = Landmark(0.55, 0.45)
    points[7] = Landmark(0.68, 0.33)
    points[8] = Landmark(0.81, points[8].y)
    engine = GestureEngine(AppConfig())

    commands = engine.process(observation(tuple(points), BASE))

    assert kinds(commands) == [MouseCommandType.MOVE]
    assert commands[0].x == 1.0


def test_natural_pointer_tolerates_partially_extended_middle_finger() -> None:
    points = list(pointer_points())
    points[12] = Landmark(0.51, 0.38)
    engine = GestureEngine(AppConfig())

    commands = engine.process(observation(tuple(points), BASE))

    assert kinds(commands) == [MouseCommandType.MOVE]


def test_relaxed_curved_index_still_controls_pointer() -> None:
    """手已经被模型检测到时，不应再要求用户维持僵硬的标准食指姿势。"""

    points = list(pointer_points())
    points[8] = Landmark(0.70, 0.20)
    points[12] = Landmark(0.51, 0.32)
    engine = GestureEngine(AppConfig())

    commands = engine.process(observation(tuple(points), BASE))

    assert kinds(commands) == [MouseCommandType.MOVE]


def test_pinch_ignores_unstable_depth_coordinate() -> None:
    points = list(left_pinch_points())
    points[4] = Landmark(points[4].x, points[4].y, 0.25)
    engine = GestureEngine(AppConfig())

    assert engine.process(observation(tuple(points), BASE)) == []
    commands = engine.process(observation(pointer_points(), BASE + 120_000_000))

    assert kinds(commands) == [MouseCommandType.LEFT_CLICK]


def test_pointer_extension_ignores_unstable_depth_coordinate() -> None:
    points = list(pointer_points())
    points[8] = Landmark(points[8].x, points[8].y, 0.5)
    engine = GestureEngine(AppConfig())

    commands = engine.process(observation(tuple(points), BASE))

    assert kinds(commands) == [MouseCommandType.MOVE]


def test_short_left_pinch_clicks_only_after_release() -> None:
    engine = GestureEngine(AppConfig())
    assert engine.process(observation(left_pinch_points(), BASE)) == []

    commands = engine.process(
        observation(pointer_points(), BASE + 200_000_000),
    )

    assert kinds(commands) == [MouseCommandType.LEFT_CLICK]


def test_fist_like_left_pinch_has_priority_over_pause_confirmation() -> None:
    """录像复现：食指与拇指接触时不能先进入握拳确认。"""

    engine = GestureEngine(AppConfig(fist_hold_ms=500))
    started = engine.process(observation(fist_like_left_pinch_points(), BASE))
    held = engine.process(
        observation(fist_like_left_pinch_points(), BASE + 700_000_000),
    )
    released = engine.process(
        observation(pointer_points(), BASE + 760_000_000),
    )

    assert started == []
    assert held == []
    assert engine.paused is False
    assert kinds(released) == [MouseCommandType.LEFT_CLICK]


def test_default_drag_requires_five_seconds_of_continuous_pinch() -> None:
    """视频明确要求长按约5秒后才进入拖拽。"""

    engine = GestureEngine(AppConfig())
    engine.process(observation(left_pinch_points(), BASE))

    before_threshold = engine.process(
        observation(left_pinch_points(), BASE + 4_900_000_000),
    )
    at_threshold = engine.process(
        observation(left_pinch_points(), BASE + 5_010_000_000),
    )

    assert before_threshold == []
    assert kinds(at_threshold) == [MouseCommandType.LEFT_DOWN, MouseCommandType.MOVE]


def test_pre_pinch_intent_freezes_pointer_before_contact() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(pointer_points(), BASE))
    near_pinch = list(pointer_points())
    near_pinch[4] = Landmark(0.42, 0.36)

    commands = engine.process(
        observation(tuple(near_pinch), BASE + 45_000_000),
    )

    assert commands == []
    assert engine.current_gesture.value == "pinch_armed"


def test_long_left_pinch_becomes_drag_then_releases() -> None:
    engine = GestureEngine(AppConfig(drag_hold_ms=300))
    engine.process(observation(left_pinch_points(), BASE))

    held = engine.process(
        observation(left_pinch_points(), BASE + 310_000_000),
    )
    released = engine.process(
        observation(pointer_points(), BASE + 360_000_000),
    )

    assert kinds(held) == [MouseCommandType.LEFT_DOWN, MouseCommandType.MOVE]
    assert kinds(released) == [MouseCommandType.LEFT_UP]


def test_hand_loss_releases_active_drag_once() -> None:
    engine = GestureEngine(AppConfig(drag_hold_ms=300, hand_lost_ms=200))
    engine.process(observation(left_pinch_points(), BASE))
    engine.process(observation(left_pinch_points(), BASE + 310_000_000))

    first = engine.tick(BASE + 520_000_000)
    second = engine.tick(BASE + 800_000_000)

    assert kinds(first) == [MouseCommandType.LEFT_UP]
    assert second == []


def test_right_click_triggers_on_release() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(pointer_points(), BASE))
    assert engine.process(observation(right_pinch_points(), BASE)) == []
    commands = engine.process(observation(pointer_points(), BASE + 100_000_000))
    assert kinds(commands) == [MouseCommandType.RIGHT_CLICK]


def test_folded_middle_near_thumb_remains_pointer_instead_of_false_right_click() -> None:
    engine = GestureEngine(AppConfig())

    commands = engine.process(
        observation(middle_tucked_pointer_points(), BASE),
    )

    assert kinds(commands) == [MouseCommandType.MOVE]
    assert engine.current_gesture.value == "pointer"


def test_right_pinch_requires_deliberate_open_then_close_sequence() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(middle_tucked_pointer_points(), BASE))

    still_pointer = engine.process(
        observation(middle_tucked_pointer_points(), BASE + 40_000_000),
    )

    assert kinds(still_pointer) == [MouseCommandType.MOVE]
    engine.process(observation(pointer_points(), BASE + 80_000_000))
    assert engine.process(
        observation(right_pinch_points(), BASE + 120_000_000),
    ) == []
    released = engine.process(
        observation(pointer_points(), BASE + 180_000_000),
    )
    assert kinds(released) == [MouseCommandType.RIGHT_CLICK]


def test_deliberate_right_pinch_can_start_from_open_palm() -> None:
    """自然张掌后直接用拇指和中指捏合，应被识别为右键。"""

    engine = GestureEngine(AppConfig())
    engine.process(observation(open_palm_points(), BASE))

    started = engine.process(
        observation(right_pinch_from_open_points(), BASE + 100_000_000),
    )
    released = engine.process(
        observation(pointer_points(), BASE + 180_000_000),
    )

    assert started == []
    assert kinds(released) == [MouseCommandType.RIGHT_CLICK]


def test_open_palm_to_pointer_transition_does_not_arm_false_right_click() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(open_palm_points(), BASE))

    commands = engine.process(
        observation(middle_tucked_pointer_points(), BASE + 80_000_000),
    )

    assert kinds(commands) == [MouseCommandType.MOVE]
    assert engine.current_gesture.value == "pointer"


def test_latched_right_pinch_is_not_stolen_by_ambiguous_index_contact() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(pointer_points(), BASE))
    engine.process(observation(right_pinch_points(), BASE + 40_000_000))

    ambiguous = engine.process(
        observation(ambiguous_right_pinch_points(), BASE + 80_000_000),
    )
    released = engine.process(
        observation(pointer_points(), BASE + 140_000_000),
    )

    assert ambiguous == []
    assert kinds(released) == [MouseCommandType.RIGHT_CLICK]


def test_fist_hold_toggles_pause_and_requires_release_before_resume() -> None:
    engine = GestureEngine(AppConfig(fist_hold_ms=500))
    engine.process(observation(fist_points(), BASE))
    paused = engine.process(observation(fist_points(), BASE + 510_000_000))
    still_paused = engine.process(observation(fist_points(), BASE + 800_000_000))
    engine.process(observation(pointer_points(), BASE + 900_000_000))
    engine.process(observation(fist_points(), BASE + 1_000_000_000))
    resumed = engine.process(observation(fist_points(), BASE + 1_510_000_000))

    assert kinds(paused) == [MouseCommandType.PAUSE]
    assert paused[0].paused is True
    assert still_paused == []
    assert kinds(resumed) == [MouseCommandType.PAUSE]
    assert resumed[0].paused is False


def test_open_palm_horizontal_swipe_triggers_one_next_page() -> None:
    engine = GestureEngine(AppConfig())
    start = open_palm_points()
    engine.process(observation(start, BASE))

    page = engine.process(
        observation(translate_points(start, dx=-0.20), BASE + 180_000_000),
    )
    held = engine.process(
        observation(translate_points(start, dx=-0.24), BASE + 230_000_000),
    )

    assert kinds(page) == [MouseCommandType.PAGE_NEXT]
    assert held == []


def test_open_palm_right_swipe_triggers_previous_page() -> None:
    engine = GestureEngine(AppConfig())
    start = open_palm_points()
    engine.process(observation(start, BASE))

    page = engine.process(
        observation(translate_points(start, dx=0.20), BASE + 180_000_000),
    )

    assert kinds(page) == [MouseCommandType.PAGE_PREVIOUS]


def test_page_swipe_can_be_disabled_without_becoming_scroll() -> None:
    engine = GestureEngine(AppConfig(page_gestures_enabled=False))
    start = open_palm_points()
    engine.process(observation(start, BASE))

    commands = engine.process(
        observation(translate_points(start, dx=-0.20), BASE + 180_000_000),
    )

    assert commands == []


def test_open_palm_vertical_motion_still_scrolls_without_page_event() -> None:
    engine = GestureEngine(AppConfig())
    start = open_palm_points()
    engine.process(observation(start, BASE))

    scroll = engine.process(
        observation(translate_points(start, dy=-0.08), BASE + 120_000_000),
    )

    assert kinds(scroll) == [MouseCommandType.WHEEL]
    assert scroll[0].delta > 0


def test_open_palm_rotation_scrolls_without_translating_hand_center() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(open_palm_points(), BASE))

    scroll = engine.process(
        observation(rotated_scroll_points(), BASE + 100_000_000),
    )

    assert kinds(scroll) == [MouseCommandType.WHEEL]
    assert scroll[0].delta != 0
    assert engine.current_gesture.value == "scroll"


def test_click_intent_has_priority_over_rotation_scroll() -> None:
    engine = GestureEngine(AppConfig())
    engine.process(observation(open_palm_points(), BASE))
    pinch = list(left_pinch_points())
    pinch[9] = Landmark(pinch[9].x, pinch[9].y, 0.10)

    commands = engine.process(
        observation(tuple(pinch), BASE + 100_000_000),
    )

    assert commands == []
    assert engine.current_gesture.value == "left_pinch"


def test_horizontal_page_swipe_has_priority_over_simultaneous_depth_change() -> None:
    engine = GestureEngine(AppConfig())
    start = open_palm_points()
    engine.process(observation(start, BASE))
    moved = list(translate_points(start, dx=-0.20))
    moved[9] = Landmark(moved[9].x, moved[9].y, 0.10)

    commands = engine.process(
        observation(tuple(moved), BASE + 180_000_000),
    )

    assert kinds(commands) == [MouseCommandType.PAGE_NEXT]


def test_pointer_rotation_keeps_moving_and_cannot_start_fist_pause() -> None:
    engine = GestureEngine(AppConfig(fist_hold_ms=500))
    engine.process(observation(pointer_points(), BASE))

    rotated = engine.process(
        observation(rotated_projected_closed_points(), BASE + 100_000_000),
    )
    held = engine.process(
        observation(rotated_projected_closed_points(), BASE + 800_000_000),
    )

    assert kinds(rotated) == [MouseCommandType.MOVE]
    assert kinds(held) == [MouseCommandType.MOVE]
    assert engine.paused is False
    assert engine.current_gesture.value == "pointer"


def test_rotation_scroll_transition_cannot_be_misjudged_as_pause() -> None:
    engine = GestureEngine(AppConfig(fist_hold_ms=500))
    engine.process(observation(open_palm_points(), BASE))

    scroll = engine.process(
        observation(rotated_projected_closed_points(), BASE + 100_000_000),
    )
    held = engine.process(
        observation(rotated_projected_closed_points(), BASE + 800_000_000),
    )

    assert kinds(scroll) == [MouseCommandType.WHEEL]
    assert held == []
    assert engine.paused is False
    assert engine.current_gesture.value == "scroll"


def test_partial_index_moves_but_partial_middle_only_reports_identity() -> None:
    engine = GestureEngine(AppConfig())
    index = HandObservation(
        landmarks=(),
        handedness="Right",
        timestamp_ns=BASE,
        inference_ms=8.0,
        partial_finger="Index",
        partial_position=Landmark(0.40, 0.30),
    )
    middle = HandObservation(
        landmarks=(),
        handedness="Right",
        timestamp_ns=BASE + 45_000_000,
        inference_ms=8.0,
        partial_finger="Middle",
        partial_position=Landmark(0.45, 0.30),
    )

    assert kinds(engine.process(index)) == [MouseCommandType.MOVE]
    assert engine.process(middle) == []
    assert engine.current_gesture.value == "partial_middle"


def test_left_pinch_falls_back_to_lost_when_hand_disappears() -> None:
    """r2 实机复现：捏合态下手离画（盖镜头/拔摄像头）后状态冻在 left_pinch。

    硬线 1/6 都要求"手离画能回 lost"，replay baseline-r2 时 58 秒无手却
    始终停在 left_pinch，说明失联保护在这条路径上不可达。
    """

    config = AppConfig()
    engine = GestureEngine(config)
    engine.process(observation(left_pinch_points(), BASE))
    assert engine.current_gesture.value == "left_pinch"

    engine.tick(BASE + (config.hand_lost_ms + 50) * 1_000_000)

    assert engine.current_gesture.value == "lost"


def test_pointer_falls_back_to_lost_when_hand_disappears() -> None:
    config = AppConfig()
    engine = GestureEngine(config)
    engine.process(observation(pointer_points(), BASE))
    assert engine.current_gesture.value == "pointer"

    engine.tick(BASE + (config.hand_lost_ms + 50) * 1_000_000)

    assert engine.current_gesture.value == "lost"


# ---------------- 进入门（测量域闸门）----------------


def _enter_gate(engine, start_ns: int = BASE) -> None:
    """喂一帧进入姿势、再喂一帧（相距 ≥300ms）完成开闸。"""

    engine.process(observation(entry_palm_points(), start_ns))
    engine.process(observation(entry_palm_points(), start_ns + 320_000_000))


def test_entry_gate_blocks_downstream_gestures_until_presented() -> None:
    """未进入控制态时，捏合/翻页/握拳等下游手势一律不执行。"""

    engine = GestureEngine(AppConfig(entry_gate_enabled=True))

    assert engine.process(observation(pointer_points(), BASE)) == []
    assert engine.process(observation(left_pinch_points(), BASE + 50_000_000)) == []
    assert engine.process(observation(fist_points(), BASE + 100_000_000)) == []
    assert engine.process(observation(open_palm_points(), BASE + 150_000_000)) == []
    assert engine.current_gesture.value == "gate_idle"


def test_entry_gate_opens_after_continuous_hold() -> None:
    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    assert engine.process(observation(entry_palm_points(), BASE)) == []
    assert engine.current_gesture.value == "gate_idle"

    engine.process(observation(entry_palm_points(), BASE + 320_000_000))
    assert engine.current_gesture.value != "gate_idle"

    commands = engine.process(observation(pointer_points(), BASE + 400_000_000))
    assert kinds(commands) == [MouseCommandType.MOVE]


def test_entry_gate_hold_restarts_when_pose_breaks() -> None:
    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    engine.process(observation(entry_palm_points(), BASE))
    engine.process(observation(pointer_points(), BASE + 100_000_000))
    engine.process(observation(entry_palm_points(), BASE + 200_000_000))

    engine.process(observation(entry_palm_points(), BASE + 450_000_000))
    assert engine.current_gesture.value == "gate_idle"

    engine.process(observation(entry_palm_points(), BASE + 520_000_000))
    assert engine.current_gesture.value != "gate_idle"


def test_entry_gate_rejects_back_of_hand() -> None:
    """误进防护：手背朝镜的五指张开不得开闸门。"""

    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    for step in range(20):
        engine.process(observation(entry_back_points(), BASE + step * 30_000_000))

    assert engine.current_gesture.value == "gate_idle"


def test_entry_gate_rejects_side_palm() -> None:
    """误进防护：侧掌不得开闸门。"""

    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    for step in range(20):
        engine.process(observation(side_palm_points(), BASE + step * 30_000_000))

    assert engine.current_gesture.value == "gate_idle"


def test_entry_gate_rejects_fist() -> None:
    """误进防护：握拳/半握不得开闸门。"""

    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    for step in range(20):
        engine.process(observation(fist_points(), BASE + step * 30_000_000))

    assert engine.current_gesture.value == "gate_idle"


def test_entry_gate_is_closed_when_handedness_missing() -> None:
    """左右手不可判定时保守关闸：宁可进不去，也不能让手背误开闸门。"""

    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    for step in range(20):
        engine.process(
            HandObservation(
                landmarks=entry_palm_points(),
                handedness="",
                timestamp_ns=BASE + step * 30_000_000,
                inference_ms=8.0,
            ),
        )

    assert engine.current_gesture.value == "gate_idle"


def test_fist_exits_gate_and_requires_reentry() -> None:
    """退出条件 1：握拳退出控制态，重新入画必须再做进入姿势。"""

    engine = GestureEngine(AppConfig(entry_gate_enabled=True, fist_hold_ms=500))
    _enter_gate(engine)
    engine.process(observation(pointer_points(), BASE + 400_000_000))

    engine.process(observation(fist_points(), BASE + 500_000_000))
    engine.process(observation(fist_points(), BASE + 1_020_000_000))
    assert engine.current_gesture.value == "gate_idle"

    assert engine.process(observation(pointer_points(), BASE + 1_100_000_000)) == []
    assert engine.current_gesture.value == "gate_idle"


def test_hand_out_exits_gate_and_requires_reentry() -> None:
    """退出条件 2：手出画 ≥ hand_lost_ms 退出控制态。"""

    config = AppConfig(entry_gate_enabled=True)
    engine = GestureEngine(config)
    _enter_gate(engine)
    engine.process(observation(pointer_points(), BASE + 400_000_000))

    engine.tick(BASE + 400_000_000 + (config.hand_lost_ms + 50) * 1_000_000)
    assert engine.current_gesture.value == "lost"

    assert engine.process(observation(pointer_points(), BASE + 1_000_000_000)) == []
    assert engine.current_gesture.value == "gate_idle"


def test_fist_during_click_or_drag_does_not_exit_gate() -> None:
    """误出防护：拖拽中握拳不得踢出控制态，否则一捏合就自锁。"""

    engine = GestureEngine(
        AppConfig(entry_gate_enabled=True, drag_hold_ms=300, fist_hold_ms=500),
    )
    _enter_gate(engine)
    engine.process(observation(left_pinch_points(), BASE + 400_000_000))
    engine.process(observation(left_pinch_points(), BASE + 800_000_000))
    assert engine._left_button_down is True

    engine.process(observation(fist_points(), BASE + 1_000_000_000))
    engine.process(observation(fist_points(), BASE + 2_000_000_000))

    assert engine.current_gesture.value != "gate_idle"
