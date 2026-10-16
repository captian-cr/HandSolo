import pytest
from math import sqrt

from handmouse.config import AppConfig
from handmouse.contracts import HandObservation, Landmark, MouseCommandType
from handmouse.gestures import GestureEngine, _palm_axis_cross, _palm_presented

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


# ---------------- 掌心朝向判定（mirror 语义修正 + sin 幅度门槛）----------------


def _palm_landmarks(*, cross_sign: float, sin_value: float) -> tuple[Landmark, ...]:
    """构造掌轴(5→17)沿 +x、拇指矢量(5→4)幅角可控的手。

    across=(0.2, 0)，故叉积符号 = sign(thumb_y) = cross_sign；sinθ =
    |thumb_y|/|thumb| 与 thumb 尺度无关，取 thumb=(√(1-s²), s)（单位长）即
    得 sinθ≈sin_value。掌宽/腕-中指距离比 0.2/0.6≈0.33 ≥ 0.25 通过正对面判定。

    thumb_y 乘 (1+1e-9)：√(1-s²) 的浮点舍入会让实测 sin 比 sin_value 低
    约 1 ulp，导致 0.10 落在门槛下侧、边界用例抖动。这点相对偏置保证实测
    sin ≥ 目标值，让"阈值含等号"的边界可稳定断言。
    """

    points = [Landmark(0.5, 0.5, 0.0)] * 21
    points[0] = Landmark(0.5, 0.95, 0.0)
    points[12] = Landmark(0.5, 0.35, 0.0)
    points[5] = Landmark(0.40, 0.70, 0.0)
    points[17] = Landmark(0.60, 0.70, 0.0)
    thumb_y = cross_sign * sin_value * (1.0 + 1e-9)
    thumb_x = sqrt(max(0.0, 1.0 - sin_value * sin_value))
    points[4] = Landmark(points[5].x + thumb_x, points[5].y + thumb_y, 0.0)
    return tuple(points)


def test_palm_presented_matches_measured_signs_under_mirror() -> None:
    """mirror=True：三组 flip-diag 实测符号必须判对。

    实测（2026-10-16 flip-diag-20261016-114804，模型坐标）：
      右手掌心 label=Left cross+；右手背 label=Left cross−；左手掌心
      label=Right cross−。
    """

    # 右手掌心：label=Left、叉积为正 → 接受。
    assert _palm_presented(
        _palm_landmarks(cross_sign=+1.0, sin_value=0.65),
        "Left", mirror=True, min_sin=0.10,
    ) is True
    # 右手背：label=Left、叉积为负 → 拒绝。
    assert _palm_presented(
        _palm_landmarks(cross_sign=-1.0, sin_value=0.30),
        "Left", mirror=True, min_sin=0.10,
    ) is False
    # 左手掌心：label=Right、叉积为负 → 接受。
    assert _palm_presented(
        _palm_landmarks(cross_sign=-1.0, sin_value=0.25),
        "Right", mirror=True, min_sin=0.10,
    ) is True
    # 左手背：label=Right、叉积为正 → 拒绝。
    assert _palm_presented(
        _palm_landmarks(cross_sign=+1.0, sin_value=0.25),
        "Right", mirror=True, min_sin=0.10,
    ) is False


def test_palm_presented_matches_signs_without_mirror() -> None:
    """mirror=False 回归：未翻转像素，标签即解剖学手别，约定仍成立。

    无镜像时右手掌心 cross 为正（与 mirror=True 相反），左手掌心 cross 为负。
    """

    assert _palm_presented(
        _palm_landmarks(cross_sign=+1.0, sin_value=0.40),
        "Right", mirror=False, min_sin=0.10,
    ) is True
    assert _palm_presented(
        _palm_landmarks(cross_sign=-1.0, sin_value=0.40),
        "Left", mirror=False, min_sin=0.10,
    ) is True
    # 无镜像下的手背：同一标签、符号相反 → 拒绝。
    assert _palm_presented(
        _palm_landmarks(cross_sign=-1.0, sin_value=0.40),
        "Right", mirror=False, min_sin=0.10,
    ) is False


def test_palm_presented_sin_threshold_boundary() -> None:
    """幅度门槛边界：sin=0.09 拒绝、0.10 与 0.11 放行（门槛含等号）。

    门槛只杀几何退化帧，不判掌/背；这里用符号正确的掌心帧只测门槛本身。
    """

    for sin_value, expected in ((0.09, False), (0.10, True), (0.11, True)):
        points = _palm_landmarks(cross_sign=+1.0, sin_value=sin_value)
        _, measured = _palm_axis_cross(points)
        assert measured == pytest.approx(sin_value, abs=1e-9)
        assert _palm_presented(
            points, "Left", mirror=True, min_sin=0.10,
        ) is expected


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
