import logging
import pytest
from math import sqrt

from handmouse.config import AppConfig
from handmouse.contracts import HandObservation, Landmark, MouseCommandType
from handmouse.gestures import GestureEngine, _expected_palm_sign, _palm_facing_unit

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
    # 第一次释放只挂起双击窗口，窗口内不提交单击。
    assert engine.process(
        observation(pointer_points(), BASE + 120_000_000),
    ) == []
    # 窗口到期后的下一帧补提交单击。
    commands = engine.process(
        observation(pointer_points(), BASE + 500_000_000),
    )

    assert kinds(commands) == [MouseCommandType.LEFT_CLICK, MouseCommandType.MOVE]


def test_pointer_extension_ignores_unstable_depth_coordinate() -> None:
    points = list(pointer_points())
    points[8] = Landmark(points[8].x, points[8].y, 0.5)
    engine = GestureEngine(AppConfig())

    commands = engine.process(observation(tuple(points), BASE))

    assert kinds(commands) == [MouseCommandType.MOVE]


def test_short_left_pinch_clicks_only_after_release_window() -> None:
    """短捏合释放后单击延迟一个双击窗口提交（VR 捏合同款取舍）。"""

    engine = GestureEngine(AppConfig())
    assert engine.process(observation(left_pinch_points(), BASE)) == []

    # 释放瞬间不点击：挂起窗口，等可能到来的第二次捏合。
    assert engine.process(
        observation(pointer_points(), BASE + 200_000_000),
    ) == []

    # 窗口到期后的后续帧补提交单击。
    commands = engine.process(
        observation(pointer_points(), BASE + 600_000_000),
    )

    assert kinds(commands) == [MouseCommandType.LEFT_CLICK, MouseCommandType.MOVE]


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
    assert released == []
    # 挂起的单击在双击窗口到期后补提交。
    flushed = engine.process(
        observation(pointer_points(), BASE + 1_200_000_000),
    )
    assert engine.paused is False
    assert kinds(flushed) == [MouseCommandType.LEFT_CLICK, MouseCommandType.MOVE]


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


def test_drag_cursor_stays_put_when_only_index_finger_curves() -> None:
    """方案 B 验收（等式级）：手整体不动、只弯食指，拖拽 MOVE 严格不变。

    掌心由 0/5/9/13/17 构成，食指尖弯缩不改变掌心屏幕坐标；锚定式 MOVE
    的 raw 逐帧恒等，OneEuro deadzone 内输出与输入严格相等——不是容差近似。
    """

    engine = GestureEngine(AppConfig(drag_hold_ms=300))

    def feed(points, ms):
        return engine.process(observation(points, BASE + int(ms * 1_000_000)))

    # 先有 pointer 帧建立真实光标，再接触式捏合进入拖拽。
    pinch = left_pinch_points()
    feed(pointer_points(), 0)
    feed(pinch, 33)
    entered = feed(pinch, 350)
    assert kinds(entered) == [MouseCommandType.LEFT_DOWN, MouseCommandType.MOVE]
    anchor = (entered[1].x, entered[1].y)

    move_positions = [anchor]
    t_ms = 383

    # 8 步把食指尖向 PIP 方向弯缩 0.020（拇指尖与手掌其余点完全不动）。
    bend_total = 0.020
    for step in range(1, 9):
        bent = list(pinch)
        bent[8] = Landmark(pinch[8].x, pinch[8].y + bend_total * step / 8)
        commands = feed(tuple(bent), t_ms)
        t_ms += 33
        move_positions.extend(
            (command.x, command.y)
            for command in commands
            if command.kind == MouseCommandType.MOVE
        )

    # 弯到底后再静止 6 帧，稳态输出同样必须严格等于锚点。
    fully_bent = list(pinch)
    fully_bent[8] = Landmark(pinch[8].x, pinch[8].y + bend_total)
    for _ in range(6):
        commands = feed(tuple(fully_bent), t_ms)
        t_ms += 33
        move_positions.extend(
            (command.x, command.y)
            for command in commands
            if command.kind == MouseCommandType.MOVE
        )

    assert len(move_positions) > 1
    assert all(position == anchor for position in move_positions)


def test_drag_cursor_follows_whole_hand_translation() -> None:
    """锚定式拖拽的正向保证：整手平移（掌心刚体位移）必须带动光标。"""

    engine = GestureEngine(AppConfig(drag_hold_ms=300))

    def feed(points, ms):
        return engine.process(observation(points, BASE + int(ms * 1_000_000)))

    pinch = left_pinch_points()
    feed(pointer_points(), 0)
    feed(pinch, 33)
    entered = feed(pinch, 350)
    anchor = (entered[1].x, entered[1].y)

    last_move = anchor
    for step in range(1, 5):
        moved = translate_points(pinch, dx=0.01 * step, dy=0.0)
        commands = feed(moved, 350 + step * 33)
        for command in commands:
            if command.kind == MouseCommandType.MOVE:
                last_move = (command.x, command.y)

    assert last_move[0] > anchor[0]


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

    硬线 H1 要求"手离画能回 lost"，replay baseline-r2 时 58 秒无手却
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


# ---------------- 时间域统一（笔1）：真 lost 回归 ----------------


def test_lost_not_triggered_while_hand_stays_in_view(arrival_clock) -> None:
    """手持续在画面：到达间隔（10Hz = 100ms）远小于 hand_lost_ms，不得失联。"""

    config = AppConfig()
    engine = GestureEngine(config)
    for step in range(5):
        timestamp_ns = BASE + step * 100_000_000
        arrival_clock.set_ns(timestamp_ns)
        engine.process(observation(pointer_points(), timestamp_ns))

    # 最后一帧到达后只隔一个帧间隔就 tick：手一直在交付路径上。
    engine.tick(BASE + 500_000_000)

    assert engine.current_gesture.value != "lost"


def test_lost_triggered_when_hand_leaves_view(arrival_clock) -> None:
    """真 lost 必须仍会触发：出画后到达间隔 ≥ hand_lost_ms。

    时间域统一不能把真出画也一起修没——r3 第 25 节拍那种手离画场景要能复现 lost。
    """

    config = AppConfig()
    engine = GestureEngine(config)
    arrival_clock.set_ns(BASE)
    engine.process(observation(pointer_points(), BASE))
    assert engine.current_gesture.value == "pointer"

    engine.tick(BASE + (config.hand_lost_ms + 50) * 1_000_000)

    assert engine.current_gesture.value == "lost"


def test_lost_not_triggered_by_packet_delivery_delay(arrival_clock) -> None:
    """本次 bug 的直接回归：取包延迟不得再被算成手部失联。

    采集时刻戳落后到达墙钟 200ms（r3 实测取包延迟 p95=78ms、10Hz 帧间隔 100ms），
    修复前 elapsed = 墙钟 now - 采集时刻 ≈ 200ms，刚好越过 hand_lost_ms 触发假性
    lost，把进入门的保持计时每轮清零。
    """

    config = AppConfig()
    engine = GestureEngine(config)
    capture_ns = BASE
    arrival_clock.set_ns(BASE + 200_000_000)
    engine.process(observation(pointer_points(), capture_ns))

    # 同一时刻 tick：手仍在交付路径上，取包延迟不构成失联。
    engine.tick(BASE + 200_000_000)

    assert engine.current_gesture.value != "lost"


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


# ---------------- 掌心朝向判定（0-5-17 三角朝向 + nz 幅度门槛）----------------


def _palm_landmarks(*, nz_value: float) -> tuple[Landmark, ...]:
    """构造 0-5-17 三角形可控的手：``_palm_facing_unit`` 精确等于 ``nz_value``。

    取 0→5 = (s, 0, 0)（s=0.1），5→17 = (bx, by, bz)（bx=-0.2）：
      nz = ax*by - ay*bx = s*by，nx = ay*bz - az*by = 0，
      ny = az*bx - ax*bz = -s*bz ⇒ nz_unit = by / sqrt(by²+bz²)。
    故令 by = nz_value、bz = sqrt(1-nz_value²)，nz_unit 即等于 nz_value，符号随
    nz_value——z 只进分母（幅度归一化），与生产实现同源。

    by 乘 (1+1e-9)：sqrt 的浮点舍入会让实测 |nz| 略低于目标，令 0.30 落到门槛
    下侧、边界用例抖动；这点相对偏置保证实测 |nz| ≥ 目标，使"阈值含等号"的
    0.29/0.30/0.31 边界可稳定断言。掌宽(0.2)/腕-中指距离(0.6)≈0.33 ≥ 0.25
    通过正对面判定。
    """

    points = [Landmark(0.5, 0.5, 0.0)] * 21
    points[0] = Landmark(0.5, 0.9, 0.0)
    points[12] = Landmark(0.5, 0.3, 0.0)
    points[5] = Landmark(0.6, 0.9, 0.0)
    by = nz_value * (1.0 + 1e-9)
    bz = sqrt(max(0.0, 1.0 - by * by))
    points[17] = Landmark(points[5].x - 0.2, points[5].y + by, bz)
    return tuple(points)


def _entry_engine() -> GestureEngine:
    """进入门开启的引擎（符号锁存走真实产路径）。"""

    return GestureEngine(AppConfig(entry_gate_enabled=True))


def _presented(
    engine: GestureEngine,
    nz_value: float,
    handedness: str = "Left",
    mirror: bool = True,
    in_hold: bool = False,
) -> bool:
    """用单帧 ``_palm_landmarks`` 驱动 ``_palm_presented``（可复用同一引擎）。"""

    return engine._palm_presented(
        _palm_landmarks(nz_value=nz_value),
        _expected_palm_sign(handedness, mirror),
        in_hold=in_hold,
    )


def test_palm_presented_matches_measured_signs_under_mirror() -> None:
    """mirror=True：四组 sign-ref 实测符号必须判对。

    实测（2026-10-16 sign-ref 三轮回算，模型坐标 nz_unit 符号）：
      右手掌心 label=Left nz>0；右手背 label=Left nz<0；左手掌心
      label=Right nz<0；左手背 label=Right nz>0。
    """

    # 每条断言用新引擎：锁存有状态，冷启动首帧免 K 立即采纳。
    # 右手掌心：label=Left、nz>0 → 接受。
    assert _presented(_entry_engine(), +0.90, "Left", True) is True
    # 右手背：label=Left、nz<0 → 拒绝。
    assert _presented(_entry_engine(), -0.90, "Left", True) is False
    # 左手掌心：label=Right、nz<0 → 接受。
    assert _presented(_entry_engine(), -0.90, "Right", True) is True
    # 左手背：label=Right、nz>0 → 拒绝。
    assert _presented(_entry_engine(), +0.90, "Right", True) is False


def test_palm_presented_matches_signs_without_mirror() -> None:
    """mirror=False 回归：未翻转像素，标签即解剖学手别，约定仍成立。

    无镜像时右手掌心 nz>0（与 mirror=True 的左手掌心同号），左手掌心 nz<0。
    """

    assert _presented(_entry_engine(), +0.90, "Right", False) is True
    assert _presented(_entry_engine(), -0.90, "Left", False) is True
    # 无镜像下的手背：同一标签、符号相反 → 拒绝。
    assert _presented(_entry_engine(), -0.90, "Right", False) is False


def test_palm_presented_nz_threshold_boundary() -> None:
    """幅度门槛边界：|nz|=0.29 拒绝、0.30 与 0.31 放行（门槛含等号）。

    门槛只杀几何退化帧，不判掌/背；这里用符号正确的掌心帧只测门槛本身。
    """

    for nz_value, expected in ((0.29, False), (0.30, True), (0.31, True)):
        points = _palm_landmarks(nz_value=nz_value)
        assert _palm_facing_unit(points) == pytest.approx(nz_value, abs=1e-9)
        assert _presented(_entry_engine(), nz_value) is expected


def test_palm_facing_unit_returns_zero_for_collinear_frame() -> None:
    """0/5/17 三点近共线（几何退化帧）返回 0.0，并被 |nz| 门槛拒掉。

    退化帧返回 0.0 是刻意的：符号在三点共线时纯属噪声，返回 0 让幅度门槛
    直接拒绝，而不是把一个随机符号喂给掌/背判定。
    """

    points = [Landmark(0.5, 0.5, 0.0)] * 21
    points[0] = Landmark(0.5, 0.9, 0.0)
    points[12] = Landmark(0.5, 0.3, 0.0)
    points[5] = Landmark(0.5, 0.7, 0.0)
    points[17] = Landmark(0.5, 0.5, 0.0)  # 与 0、5 同线

    assert _palm_facing_unit(points) == 0.0
    assert _entry_engine()._palm_presented(
        points, _expected_palm_sign("Left", True),
    ) is False


# ---------------- 符号锁存（③：|nz| 门控 + K 帧确认）----------------
# 下面用默认 "Left"+mirror：期望符号 = +1，故 nz>0 为"匹配"、nz<0 为"错号"。


def test_facing_lock_adopts_first_frame_without_k() -> None:
    """冷启动首帧采纳：|nz|≥min 一帧即锁定，不等 K 帧。

    若首帧也要 K 帧确认，开闸会从 300ms 拖到 300ms+K 帧 ≈ 1.2s——那是新 bug。
    """

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True
    assert engine._locked_facing == +1.0


def test_facing_lock_ignores_transition_block_shorter_than_k() -> None:
    """过渡块不采纳：已锁 +1 后，6 帧错号且 |nz|=0.50 不得切换锁存。"""

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True   # 首帧锁 +1（= 期望符号）

    for _ in range(6):
        assert _presented(engine, -0.50) is True   # 过渡块，锁未变 → 仍"匹配"

    assert engine._locked_facing == +1.0
    assert engine._candidate_count == 6           # 候选在累积但未达 K=8


def test_facing_lock_switches_after_k_continuous_high_confidence_frames() -> None:
    """持续 K 帧采纳：8 帧同向且 |nz|≥commit 才切换锁存。"""

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True

    results = [_presented(engine, -0.90) for _ in range(8)]

    assert results[:7] == [True] * 7      # 前 7 帧仍是旧锁 +1
    assert engine._locked_facing == -1.0  # 第 8 帧切换
    assert results[-1] is False           # 切换后与期望 +1 不符


def test_facing_lock_ignores_frames_below_commit_magnitude() -> None:
    """中等幅值不参与：8 帧同向但 |nz|=0.40 < commit → 不切换。"""

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True

    for _ in range(8):
        assert _presented(engine, -0.40) is True   # <commit，冻结候选

    assert engine._locked_facing == +1.0
    assert engine._candidate_count == 0


def test_facing_lock_candidate_resets_on_sub_commit_frames() -> None:
    """退化/中等幅值帧清零候选：交替 0.40/0.20 序列 → 候选始终清零。"""

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True

    for nz_value in (-0.40, -0.20) * 4:
        _presented(engine, nz_value)

    assert engine._locked_facing == +1.0
    assert engine._candidate_facing is None
    assert engine._candidate_count == 0


def test_facing_lock_requires_strictly_consecutive_frames() -> None:
    """严格连续：退化帧（|nz| < min）必须打断候选，不得跨谷累加。

    序列 ``-0.9 / -0.25`` 交替 8 组（-0.9 共出现 8 次但**不连续**）：
    若退化帧对锁存不可见（|nz|<min 早退在 _facing_committed 之前），-0.9 会被
    累加到 K=8 而切换锁存——违反"连续 K 帧"的本意。锁存必须先于 min 门更新，
    让 -0.25 走 ``<commit`` 分支清零候选。
    """

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True   # 锁 +1

    for _ in range(8):
        _presented(engine, -0.90)   # 错号、过 commit
        _presented(engine, -0.25)   # 退化帧（< min），必须先打断候选

    assert engine._locked_facing == +1.0
    assert engine._candidate_count == 0


def test_facing_lock_follows_sign_immediately_during_hold() -> None:
    """hold 期：单帧过 commit 的错号立即翻号、免 K（防抖交给 entry_hold_ms）。"""

    engine = _entry_engine()
    assert _presented(engine, +0.90) is True          # 冷启动锁 +1
    assert engine._locked_facing == +1.0

    # 非 hold 期维持 K 迟滞：1 帧错号不翻。
    assert _presented(engine, -0.90, in_hold=False) is True
    assert engine._locked_facing == +1.0
    assert engine._candidate_count == 1

    # hold 期：同样一帧过 commit 的错号立即翻号，候选清零。
    assert _presented(engine, -0.90, in_hold=True) is False
    assert engine._locked_facing == -1.0
    assert engine._candidate_count == 0


def test_entry_gate_does_not_open_when_sign_flips_during_hold() -> None:
    """H4 竞速回归：冷启动锁错符号后，hold 内翻成手背不得在 300ms 顶开闸门。

    实机 sign-ref 20261016-1708 复现序列：手出画后回画首帧仍在翻手过渡态，
    冷启动首帧免 K 锁了错号；随后连续手背高置信帧在 K=8（≈0.94s）内来不及
    翻号，而 entry_hold_ms=300ms 先到——闸门比锁存先动，手背朝镜进了控制态
    （seg2 gate_active=2，硬条件 FAIL）。

    修法：hold 计时累积期间（entry_started_ns 已置位）符号立即跟随，防抖由
    entry_hold_ms 本身提供（presented 一旦翻 False 计时清零）；非 hold 期才
    走 K=8。本测试用 entry_palm（nz≈-1，匹配期望 -1）冷启动锁号，第二帧起
    改喂 entry_back（nz≈+1，错号），350ms 内闸门必须始终不开。
    """

    engine = GestureEngine(
        AppConfig(entry_gate_enabled=True, entry_hold_ms=300),
    )

    # t=0：回画首帧（掌心）→ 冷启动免 K 立即锁 -1，presented=True，hold 开始。
    assert engine.process(observation(entry_palm_points(), BASE)) == []
    assert engine._entry_started_ns == BASE
    assert engine._gate_active is False

    # t=100/200ms：已翻成手背（连续高置信错号帧）。hold 期符号立即跟随 →
    # presented 翻 False、entry 计时清零；闸门继续关着。
    engine.process(observation(entry_back_points(), BASE + 100_000_000))
    assert engine._gate_active is False
    assert engine.current_gesture.value == "gate_idle"
    engine.process(observation(entry_back_points(), BASE + 200_000_000))
    assert engine._gate_active is False

    # t=350ms：旧实现这里 held_ms≥300 直接开闸（H4 漏检）；修复后必须仍在门外。
    engine.process(observation(entry_back_points(), BASE + 350_000_000))
    assert engine._gate_active is False
    assert engine.current_gesture.value == "gate_idle"


def test_double_pinch_fires_double_click_without_early_single() -> None:
    """硬线 H6'：窗口内同位置捏合两下 → 双击，第一次捏合不得提前提交单击。"""

    engine = GestureEngine(AppConfig())
    # 第一次捏合：无任何指令。
    assert engine.process(observation(left_pinch_points(), BASE)) == []
    # 第一次释放：单击挂起，窗口内不落地（这是 H6' 的核心）。
    assert engine.process(
        observation(pointer_points(), BASE + 100_000_000),
    ) == []
    # 第二次捏合（中间已完全释放：_left_active 只在越过 pinch_release
    # 阈值后才会清除），仍不得提前出单击。
    assert engine.process(
        observation(left_pinch_points(), BASE + 200_000_000),
    ) == []

    commands = engine.process(
        observation(pointer_points(), BASE + 300_000_000),
    )

    assert kinds(commands) == [MouseCommandType.LEFT_DOUBLE_CLICK]


def test_second_pinch_outside_window_becomes_two_single_clicks() -> None:
    """超过双击窗口后的第二次捏合按两次独立单击处理。"""

    engine = GestureEngine(AppConfig())
    engine.process(observation(left_pinch_points(), BASE))
    assert engine.process(
        observation(pointer_points(), BASE + 100_000_000),
    ) == []
    # 窗口到期帧：补提交第一次单击。
    first = engine.process(
        observation(pointer_points(), BASE + 500_000_000),
    )
    assert kinds(first) == [MouseCommandType.LEFT_CLICK, MouseCommandType.MOVE]

    # 此后再来一次捏合：是全新的一轮，释放后只挂起单击。
    assert engine.process(
        observation(left_pinch_points(), BASE + 600_000_000),
    ) == []
    assert engine.process(
        observation(pointer_points(), BASE + 700_000_000),
    ) == []
    second = engine.process(
        observation(pointer_points(), BASE + 1_100_000_000),
    )
    assert kinds(second) == [MouseCommandType.LEFT_CLICK, MouseCommandType.MOVE]


def test_second_pinch_with_large_drift_becomes_two_single_clicks() -> None:
    """两次捏合落点相差超过容差：不算双击，第一次立即补提交。"""

    drifted = translate_points(left_pinch_points(), dx=0.20)
    engine = GestureEngine(AppConfig())
    engine.process(observation(left_pinch_points(), BASE))
    engine.process(observation(pointer_points(), BASE + 100_000_000))

    # 第二次捏合位置远离第一次：本次捏合瞬间补发第一次单击。
    reengage = engine.process(observation(drifted, BASE + 200_000_000))
    assert kinds(reengage) == [MouseCommandType.LEFT_CLICK]

    # 第二次释放只挂起自己的单击，窗口到期后补提交，全程无双击。
    assert engine.process(
        observation(translate_points(pointer_points(), dx=0.20),
                    BASE + 300_000_000),
    ) == []
    flushed = engine.process(
        observation(translate_points(pointer_points(), dx=0.20),
                    BASE + 700_000_000),
    )
    assert kinds(flushed) == [MouseCommandType.LEFT_CLICK, MouseCommandType.MOVE]


def test_second_pinch_held_too_long_becomes_single_click_then_drag() -> None:
    """双击的第二次捏合被按住越过拖拽阈值：双击不成立，先补单击再拖拽。"""

    engine = GestureEngine(AppConfig(drag_hold_ms=300))
    engine.process(observation(left_pinch_points(), BASE))
    engine.process(observation(pointer_points(), BASE + 100_000_000))
    engine.process(observation(left_pinch_points(), BASE + 200_000_000))

    held = engine.process(
        observation(left_pinch_points(), BASE + 560_000_000),
    )

    assert kinds(held) == [
        MouseCommandType.LEFT_CLICK,
        MouseCommandType.LEFT_DOWN,
        MouseCommandType.MOVE,
    ]


def test_pending_single_click_flushes_when_hand_lost() -> None:
    """挂起窗口内手离画：失联时把挂起的单击兑现，点击不得凭空消失。"""

    config = AppConfig(hand_lost_ms=200)
    engine = GestureEngine(config)
    engine.process(observation(left_pinch_points(), BASE))
    assert engine.process(
        observation(pointer_points(), BASE + 100_000_000),
    ) == []

    lost = engine.tick(BASE + 400_000_000)

    assert kinds(lost) == [MouseCommandType.LEFT_CLICK]
    assert engine.current_gesture.value == "lost"


def test_sign_ref_is_log_only_switch(monkeypatch, caplog) -> None:
    """参照模式是纯日志开关：开/关 SIGN_REF，判定与注入命令必须逐字一致。

    参照开关的唯一职责是"让 frontal 不足时也照打 cross/sin"。若它在任何路径
    上改动了 ``entry_palm_presented`` / gate / 注入，这条断言立刻变红——参照拿的是
    数据，不能顺手改了判定的行为。
    """

    from handmouse import gestures as gestures_module

    obs = observation(entry_palm_points(), BASE)

    monkeypatch.setattr(gestures_module, "VERBOSE_GATE", False)
    monkeypatch.setattr(gestures_module, "SIGN_REF", False)
    off = GestureEngine(AppConfig()).process(obs)

    monkeypatch.setattr(gestures_module, "SIGN_REF", True)
    with caplog.at_level(logging.INFO, logger="handmouse.gate-diag"):
        on = GestureEngine(AppConfig()).process(obs)

    def signature(cmds):
        return [(command.kind, command.x, command.y) for command in cmds]

    assert signature(on) == signature(off)
    # 开关确实生效：参照模式下打了 [sign-ref] 行，否则本测试是空跑、无牙。
    assert any("[sign-ref]" in record.getMessage() for record in caplog.records)


def test_sign_ref_line_carries_handedness_score(monkeypatch, caplog) -> None:
    """④ 分叉判据：SIGN_REF 逐帧行必须带 hn_score（观测用，非修法）。

    用途见 gestures._log_sign_ref_facing：统计"追踪链错锚首帧 vs 正确锚首帧"
    的 handedness 置信度分布，决定走 score 门槛还是几何判手别。缺这个字段，
    下一轮实机拿不到分叉证据。有效/无效手别两条分支都必须带值。
    """

    from handmouse import gestures as gestures_module

    monkeypatch.setattr(gestures_module, "VERBOSE_GATE", False)
    monkeypatch.setattr(gestures_module, "SIGN_REF", True)

    valid = observation(entry_palm_points(), BASE)
    assert valid.handedness_score == 0.98
    invalid = HandObservation(
        landmarks=entry_palm_points(),
        handedness="",
        timestamp_ns=BASE + 100_000_000,
        inference_ms=12.0,
        handedness_score=0.42,
    )

    engine = GestureEngine(AppConfig())
    with caplog.at_level(logging.INFO, logger="handmouse.gate-diag"):
        engine.process(valid)
        engine.process(invalid)

    messages = [
        record.getMessage()
        for record in caplog.records
        if "[sign-ref]" in record.getMessage()
    ]
    assert any("hn=Right" in line and "hn_score=0.9800" in line for line in messages)
    assert any("nz_unit=N/A" in line and "hn_score=0.4200" in line for line in messages)


def _flip_observation(points, handedness: str, timestamp_ns: int) -> HandObservation:
    return HandObservation(
        landmarks=points,
        handedness=handedness,
        timestamp_ns=timestamp_ns,
        inference_ms=12.0,
        handedness_score=0.98,
    )


def test_handedness_flip_logged_before_filter(monkeypatch, caplog, arrival_clock) -> None:
    """KI-002 观测点必须在 handedness 过滤器之前。

    config.handedness="Right" 时 Right→Left 翻转帧被过滤器按「无有效观测」
    吞掉；观测若放过滤器之后等于漏掉最该观测的事件。本测试是问题 A 的
    验收判据：被吞的帧仍须留下 [handedness-flip] 记录。"""
    from handmouse import gestures as gestures_module

    monkeypatch.setattr(gestures_module, "VERBOSE_GATE", True)
    engine = GestureEngine(AppConfig(handedness="Right"))
    points = entry_palm_points()
    with caplog.at_level(logging.INFO, logger="handmouse.gate-diag"):
        engine.process(observation(points, BASE))  # Right，通过过滤器
        arrival_clock.advance_ms(100)
        # Left，被过滤器吞——但翻转事件必须先于过滤被记录
        engine.process(_flip_observation(points, "Left", BASE + 100_000_000))
    messages = [record.getMessage() for record in caplog.records]
    assert any("[handedness-flip] Right->Left" in m for m in messages)


def test_handedness_flip_within_window_logged(monkeypatch, caplog, arrival_clock) -> None:
    """追踪链内存活期（Δt ≤ 500ms）的标签翻转 → 记日志，含腕部位移与 dt。"""
    from handmouse import gestures as gestures_module

    monkeypatch.setattr(gestures_module, "VERBOSE_GATE", True)
    engine = GestureEngine(AppConfig())
    points = entry_palm_points()
    with caplog.at_level(logging.INFO, logger="handmouse.gate-diag"):
        engine.process(observation(points, BASE))
        arrival_clock.advance_ms(100)
        engine.process(_flip_observation(points, "Left", BASE + 100_000_000))
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "[handedness-flip] Right->Left wrist_dist=" in m and "dt_ms=100" in m
        for m in messages
    )


def test_handedness_flip_outside_window_not_logged(monkeypatch, caplog, arrival_clock) -> None:
    """Δt > 500ms 的标签变化是间隙回画（lost 后重检采纳新标签），不算翻转。

    协议间隙 4s 完全出画，回画标签变化是 KI-002 设计里的正常行为；无时间窗
    会把每次回画记成假事件。与上一条配套才有牙：单看本条，无实现时也通过。"""
    from handmouse import gestures as gestures_module

    monkeypatch.setattr(gestures_module, "VERBOSE_GATE", True)
    engine = GestureEngine(AppConfig())
    points = entry_palm_points()
    with caplog.at_level(logging.INFO, logger="handmouse.gate-diag"):
        engine.process(observation(points, BASE))
        arrival_clock.advance_ms(600)
        engine.process(_flip_observation(points, "Left", BASE + 600_000_000))
    assert not any("[handedness-flip]" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# KI-002 轨迹纠正
# ---------------------------------------------------------------------------


def test_trajectory_correction_suppresses_flip(arrival_clock) -> None:
    """同一 wrist 轨迹下标签翻转 → 纠正为上一帧标签（默认关时透传）。"""
    engine = GestureEngine(AppConfig(enable_trajectory_correction=True))
    points = entry_palm_points()

    # 帧1：Right，建立轨迹锚点。
    engine.process(observation(points, BASE))
    assert engine._trajectory_state is not None
    assert engine._trajectory_state[0] == "Right"

    # 帧2：Left，wrist 微移（dist≈0.014 < 0.15），Δt=100ms ≤ 500ms。
    arrival_clock.advance_ms(100)
    shifted = tuple(Landmark(p.x + 0.01, p.y + 0.01, p.z) for p in points)
    engine.process(_flip_observation(shifted, "Left", BASE + 100_000_000))
    # 纠正后实际采纳的 handedness 仍为 Right（引擎内部状态已锁定）。
    assert engine._trajectory_state[0] == "Right"


def test_trajectory_correction_accepts_large_jump(arrival_clock) -> None:
    """wrist 大跳跃（dist > 阈值）视为换手，采纳新标签。"""
    engine = GestureEngine(AppConfig(enable_trajectory_correction=True))
    points = entry_palm_points()

    engine.process(observation(points, BASE))
    assert engine._trajectory_state[0] == "Right"

    # 帧2：Left，wrist 大跳跃（dx=0.30 > 0.15），Δt=100ms。
    arrival_clock.advance_ms(100)
    shifted = tuple(Landmark(p.x + 0.30, p.y, p.z) for p in points)
    engine.process(_flip_observation(shifted, "Left", BASE + 100_000_000))
    # 换手后采纳新标签。
    assert engine._trajectory_state[0] == "Left"


def test_trajectory_correction_accepts_after_window(arrival_clock) -> None:
    """时间窗口外（Δt > 500ms）视为追踪链断裂，采纳新标签。"""
    engine = GestureEngine(AppConfig(enable_trajectory_correction=True))
    points = entry_palm_points()

    engine.process(observation(points, BASE))
    assert engine._trajectory_state[0] == "Right"

    # 帧2：Left，wrist 微移，但 Δt=600ms > 500ms。
    arrival_clock.advance_ms(600)
    shifted = tuple(Landmark(p.x + 0.01, p.y, p.z) for p in points)
    engine.process(_flip_observation(shifted, "Left", BASE + 600_000_000))
    assert engine._trajectory_state[0] == "Left"


def test_trajectory_correction_disabled_by_default(arrival_clock) -> None:
    """默认关闭：标签翻转直接透传，不纠正。"""
    engine = GestureEngine(AppConfig())
    points = entry_palm_points()

    engine.process(observation(points, BASE))
    assert engine._trajectory_state is not None

    arrival_clock.advance_ms(100)
    shifted = tuple(Landmark(p.x + 0.01, p.y, p.z) for p in points)
    engine.process(_flip_observation(shifted, "Left", BASE + 100_000_000))
    # 关闭时仍维护锚点（供开闸即用），但透传标签。
    assert engine._trajectory_state[0] == "Left"


# ==================== 摇杆式光标 ====================


def test_joystick_deadzone_keeps_cursor_still() -> None:
    """食指尖在中心死区内时，摇杆光标不动（过滤生理手抖）。"""

    engine = GestureEngine(AppConfig(cursor_joystick_enabled=True))
    engine._last_smoothed = (0.5, 0.5)

    center = Landmark(0.50, 0.50)
    # 第一帧初始化摇杆位置到当前光标 (0.5, 0.5)。
    assert engine._joystick_update(center, 1_000_000_000) == (0.5, 0.5)
    # 第二帧仍在死区内：位置不变。
    assert engine._joystick_update(center, 1_100_000_000) == (0.5, 0.5)


def test_joystick_moves_in_direction_of_finger() -> None:
    """食指尖偏离中心时，光标向该方向移动（x 增大 = 向右）。"""

    engine = GestureEngine(AppConfig(cursor_joystick_enabled=True))
    engine._last_smoothed = (0.5, 0.5)

    engine._joystick_update(Landmark(0.50, 0.50), 1_000_000_000)
    x, y = engine._joystick_update(Landmark(0.70, 0.50), 1_100_000_000)
    assert x > 0.5
    assert abs(y - 0.5) < 1e-6


def test_joystick_speed_multiplier_scales_movement() -> None:
    """速度倍数越大，相同时间内光标移动越远。"""

    def run(multiplier: float) -> float:
        engine = GestureEngine(
            AppConfig(cursor_joystick_enabled=True, cursor_speed_multiplier=multiplier),
        )
        engine._last_smoothed = (0.5, 0.5)
        engine._joystick_update(Landmark(0.50, 0.50), 1_000_000_000)
        x, _ = engine._joystick_update(Landmark(0.70, 0.50), 1_100_000_000)
        return x

    assert run(2.0) > run(1.0) > 0.5


def test_joystick_dt_clamped_to_200ms() -> None:
    """卡顿（dt > 200ms）时光标位移被钳制，不会飞出去。"""

    engine = GestureEngine(
        AppConfig(cursor_joystick_enabled=True, cursor_max_speed=5.0),
    )
    engine._last_smoothed = (0.5, 0.5)

    engine._joystick_update(Landmark(0.50, 0.50), 1_000_000_000)
    # dt=1000ms（严重卡顿），实际位移应按 200ms 计算。
    x, _ = engine._joystick_update(Landmark(0.70, 0.50), 2_000_000_000)
    # 即使 max_speed=5.0、dt 名义 1s，钳制后位移不超过 5.0 * 0.2 = 1.0。
    assert x <= 1.0


def test_joystick_reinitializes_after_idle() -> None:
    """摇杆闲置超过 500ms 后重新锚定到当前光标位置，避免跳变。"""

    engine = GestureEngine(AppConfig(cursor_joystick_enabled=True))
    engine._last_smoothed = (0.3, 0.7)

    engine._joystick_update(Landmark(0.50, 0.50), 1_000_000_000)
    # 闲置 600ms 后回到摇杆：位置应重新锚定为 (0.3, 0.7)。
    engine._last_smoothed = (0.3, 0.7)
    x, y = engine._joystick_update(Landmark(0.70, 0.50), 1_600_000_000)
    assert (x, y) == (0.3, 0.7)


def test_pointer_uses_joystick_when_enabled(arrival_clock) -> None:
    """POINTER 路径在摇杆开启时使用摇杆算法，关闭时走绝对映射。"""

    points = pointer_points()  # 食指尖 (0.42, 0.20)，偏离画面中心

    # 摇杆开启：第一帧初始化摇杆到 (0.5, 0.5)，后续帧按偏移累加。
    engine_on = GestureEngine(AppConfig(cursor_joystick_enabled=True))
    first = engine_on.process(observation(points, BASE))
    move_first = [c for c in first if c.kind == MouseCommandType.MOVE][0]
    assert move_first.x == 0.5

    # 第二帧：食指尖偏向左上方，光标应向左上方移动。
    arrival_clock.advance_ms(100)
    second = engine_on.process(observation(points, BASE + 100_000_000))
    move_second = [c for c in second if c.kind == MouseCommandType.MOVE][0]
    assert move_second.x < 0.5  # 食指尖 x=0.42 < 0.5，光标向左
    assert move_second.y < 0.5  # 食指尖 y=0.20 < 0.5，光标向上

    # 摇杆关闭：绝对映射，食指尖直接映射到屏幕坐标。
    engine_off = GestureEngine(AppConfig(cursor_joystick_enabled=False))
    commands_off = engine_off.process(observation(points, BASE))
    move_off = [c for c in commands_off if c.kind == MouseCommandType.MOVE][0]
    # 绝对映射下食指尖在画面左侧，映射到屏幕左侧。
    assert move_off.x < 0.5


# ==================== 捏合防误触 ====================


def test_short_pinch_below_hold_ms_does_not_click(arrival_clock) -> None:
    """短于 pinch_hold_ms 的快速掠过不触发点击，也不挂起双击窗口。"""

    engine = GestureEngine(AppConfig(pinch_hold_ms=100))

    # 捏合开始。
    assert engine.process(observation(left_pinch_points(), BASE)) == []

    # 50ms 后释放（< 100ms 门槛）：不应挂起双击窗口。
    arrival_clock.advance_ms(50)
    commands = engine.process(observation(pointer_points(), BASE + 50_000_000))
    assert MouseCommandType.LEFT_CLICK not in kinds(commands)
    assert MouseCommandType.LEFT_DOUBLE_CLICK not in kinds(commands)
    # 双击窗口未被挂起，后续帧也不会补提交单击。
    arrival_clock.advance_ms(500)
    commands = engine.process(observation(pointer_points(), BASE + 550_000_000))
    assert MouseCommandType.LEFT_CLICK not in kinds(commands)


def test_pinch_above_hold_ms_still_clicks(arrival_clock) -> None:
    """捏合持续时间超过 pinch_hold_ms 时，正常触发点击（双击窗口机制）。"""

    engine = GestureEngine(AppConfig(pinch_hold_ms=100))

    assert engine.process(observation(left_pinch_points(), BASE)) == []

    # 200ms 后释放（> 100ms 门槛）：挂起双击窗口。
    arrival_clock.advance_ms(200)
    assert engine.process(
        observation(pointer_points(), BASE + 200_000_000),
    ) == []

    # 窗口到期后补提交单击。
    arrival_clock.advance_ms(500)
    commands = engine.process(
        observation(pointer_points(), BASE + 700_000_000),
    )
    assert MouseCommandType.LEFT_CLICK in kinds(commands)
