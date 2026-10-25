import logging
import pytest
from math import pi, sin, sqrt

from handmouse.config import AppConfig
from handmouse.contracts import HandObservation, Landmark, MouseCommandType
from handmouse.gestures import (  # noqa: E402
    GestureEngine,
    _aspect_scale,
    _distance_2d,
    _expected_palm_sign,
    _finger3d_feats,
    _finger_feats,
    _is_five_open,
    _is_two_finger,
    _isotropic,
    _palm_facing_unit,
    _palm_width,
    _two_finger_by_fingers,
    _two_finger_by_world,
    _two_finger_diag,
    _two_finger_hit,
)

from conftest import (
    ambiguous_right_pinch_points,
    entry_back_points,
    entry_palm_points,
    fist_like_left_pinch_points,
    fist_points,
    frontal_two_finger_points,
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
    thumb_open_pointer_points,
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

    assert len(move_positions) >= 1
    # 注意：2026-10-22 之后静止帧的 MOVE 会被「瞄准死区」吞掉，所以这里只
    # 保证"发出来的那些 MOVE 位置全等于锚点"。死区本身有独立用例覆盖。
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
    engine = GestureEngine(AppConfig(right_click_enabled=True))
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
    # 本用例只关心"右键要不要先张开再合"，与瞄准死区无关；静止姿势重复帧
    # 的 MOVE 会被死区吞掉，所以显式关掉死区，避免两个机制互相干扰。
    engine = GestureEngine(AppConfig(right_click_enabled=True, cursor_micro_deadband=0.0))
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

    engine = GestureEngine(AppConfig(right_click_enabled=True))
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
    engine = GestureEngine(AppConfig(right_click_enabled=True))
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


def test_right_click_disabled_ignores_deliberate_pinch() -> None:
    """总开关关闭时，完整"张开→中指捏合→松开"序列不得产出右键。

    默认关闭的原因：实机中握拳与拇指碰中指在 2.5D 投影上难以彻底分开，
    且 ``right_shape_valid`` 曾采用 105° 的宽松阈值，握拳过程中的中间帧
    可蹭过判据；一旦命中，``_right_active`` 会 latch 到松手，表现为
    "握拳就弹右键"。右键改由键盘或鼠标侧键承担。
    """

    engine = GestureEngine(AppConfig())
    assert engine.config.right_click_enabled is False

    engine.process(observation(open_palm_points(), BASE))
    engine.process(
        observation(right_pinch_from_open_points(), BASE + 100_000_000),
    )
    commands = engine.process(
        observation(pointer_points(), BASE + 200_000_000),
    )

    assert MouseCommandType.RIGHT_CLICK not in kinds(commands)


def test_right_click_disabled_still_reports_left_click() -> None:
    """关闭右键不得连带压掉左键：食指捏合仍须正常出左键。

    单击会挂起 ``double_click_window_ms``（默认 350ms）以等待可能的第二次
    捏合，因此断言要推进到窗口到期之后再取命令。
    """

    engine = GestureEngine(AppConfig())
    engine.process(observation(pointer_points(), BASE))
    engine.process(observation(left_pinch_points(), BASE + 100_000_000))
    engine.process(observation(pointer_points(), BASE + 250_000_000))
    commands = engine.process(
        observation(pointer_points(), BASE + 700_000_000),
    )

    assert MouseCommandType.LEFT_CLICK in kinds(commands)


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
    # 待机姿势要先连续保持 FIVE_OPEN_CONFIRM_MS(300ms) 才成立（2026-10-22
    # 加时序确认：单帧判据会被"只伸食指"的侧手投影误触发，见 gestures.py）。
    engine.process(observation(start, BASE))
    engine.process(observation(start, BASE + 350_000_000))

    page = engine.process(
        observation(translate_points(start, dx=-0.20), BASE + 530_000_000),
    )
    held = engine.process(
        observation(translate_points(start, dx=-0.24), BASE + 580_000_000),
    )

    assert kinds(page) == [MouseCommandType.PAGE_NEXT]
    assert held == []


def test_open_palm_right_swipe_triggers_previous_page() -> None:
    engine = GestureEngine(AppConfig())
    start = open_palm_points()
    engine.process(observation(start, BASE))
    engine.process(observation(start, BASE + 350_000_000))

    page = engine.process(
        observation(translate_points(start, dx=0.20), BASE + 530_000_000),
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


def test_open_palm_vertical_motion_no_longer_emits_wheel() -> None:
    """滚轮模式已整体移除：张掌上下平移不产生任何输出。"""

    engine = GestureEngine(AppConfig())
    start = open_palm_points()
    engine.process(observation(start, BASE))
    engine.process(observation(start, BASE + 350_000_000))

    commands = engine.process(
        observation(translate_points(start, dy=-0.08), BASE + 470_000_000),
    )

    assert commands == []
    assert engine.current_gesture.value == "open_palm"


def test_open_palm_rotation_no_longer_emits_wheel() -> None:
    """滚轮模式已整体移除：原地翻腕只改变深度，不再产生滚轮事件。"""

    engine = GestureEngine(AppConfig())
    engine.process(observation(open_palm_points(), BASE))
    engine.process(observation(open_palm_points(), BASE + 350_000_000))

    commands = engine.process(
        observation(rotated_scroll_points(), BASE + 450_000_000),
    )

    assert commands == []
    assert engine.current_gesture.value == "open_palm"


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
    engine.process(observation(start, BASE + 350_000_000))
    moved = list(translate_points(start, dx=-0.20))
    moved[9] = Landmark(moved[9].x, moved[9].y, 0.10)

    commands = engine.process(
        observation(tuple(moved), BASE + 530_000_000),
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


def test_open_palm_then_projected_closed_never_wheels() -> None:
    """五指张开 → 翻腕投影闭合：任何时刻都不得甩滚轮。"""

    engine = GestureEngine(AppConfig(fist_hold_ms=500))
    engine.process(observation(open_palm_points(), BASE))

    first = engine.process(
        observation(rotated_projected_closed_points(), BASE + 100_000_000),
    )
    held = engine.process(
        observation(rotated_projected_closed_points(), BASE + 800_000_000),
    )

    assert MouseCommandType.WHEEL not in kinds(first)
    assert MouseCommandType.WHEEL not in kinds(held)
    # 800ms 已超过握拳时长，进入门关闭时按设计会切换暂停，属预期而非
    # 误判——这里只锁死"不发滚轮"。
    assert engine.current_gesture.value in ("fist", "paused")


def test_five_open_then_pointer_moves_immediately() -> None:
    """回归：五指张开后收回成只伸食指，必须立刻回到指向模式。

    旧实现存在 ``_rotation_scroll_active`` 锁死：手收回食指后 HUD 仍永久
    显示「滚轮模式」。滚轮移除后此路径不应再有任何残留状态。
    """

    engine = GestureEngine(AppConfig(fist_hold_ms=5000))
    engine.process(observation(entry_palm_points(), BASE))
    engine.process(
        observation(rotated_projected_closed_points(), BASE + 100_000_000),
    )

    commands = [
        engine.process(observation(pointer_points(), BASE + step * 120_000_000))
        for step in range(1, 7)
    ]

    # 第一帧必须立刻发 MOVE（这才是本用例要的"立刻回到指向"）。
    # 后面几帧姿势完全没变，会被「瞄准死区」吞掉——那是 2026-10-22 用户
    # 要求的"手指不动就别抖"，不是回归。
    assert kinds(commands[0]) == [MouseCommandType.MOVE]
    assert engine.current_gesture.value == "pointer"


@pytest.mark.parametrize(
    "pose",
    [pointer_points, thumb_open_pointer_points],
    ids=["只伸食指", "食指+拇指张开"],
)
def test_single_index_poses_never_enter_scroll_mode(pose) -> None:
    """回归：只伸食指 / 食指+拇指张开永远不得进入滚轮或张掌待机。

    用户原话：「我只伸出食指和拇指食指展开还是滚动模式呢」。根因是屈伸
    分类只看二维夹角，收拢的四指在侧手投影下近似共线，被误判成五指张开。
    """

    engine = GestureEngine(AppConfig())
    base = pose()
    # 叠加翻腕（改变 9 号点深度）+ 掌心上下平移，模拟最容易误判的输入。
    for step in range(1, 9):
        moved = list(translate_points(base, dy=0.02 * step))
        moved[9] = Landmark(moved[9].x, moved[9].y, 0.05 * step)
        commands = engine.process(observation(tuple(moved), BASE + step * 60_000_000))
        assert MouseCommandType.WHEEL not in kinds(commands)
        assert engine.current_gesture.value not in ("scroll", "open_palm")


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


def test_pointing_down_does_not_trigger_pinch() -> None:
    """回归：手指朝下时食指尖勾向拇指，不得被判成捏合。

    2026-10-19 用户实测：向下指时光标乱点。原因是食指尖自然勾向拇指，
    二维距离骤降到 0.55 正好踩进 pinch_engage，而捏合判定从不看食指形态。
    """

    engine = GestureEngine(AppConfig())
    engine.process(observation(pointer_points(), BASE))

    down = list(pointer_points())
    down[8] = Landmark(0.36, 0.65)   # 食指尖落到掌指关节下方
    down[4] = Landmark(0.34, 0.66)   # 拇指尖自然贴近

    engine.process(observation(tuple(down), BASE + 60_000_000))
    engine.process(observation(tuple(down), BASE + 120_000_000))

    assert engine.current_gesture.value != "left_pinch"
    assert engine.current_gesture.value != "pinch_armed"


def test_straight_index_pinch_still_works() -> None:
    """反向保证：食指伸直、拇指凑上来的真捏合必须照常触发。"""

    engine = GestureEngine(AppConfig(pinch_hold_ms=100))
    engine.process(observation(pointer_points(), BASE))

    engine.process(observation(left_pinch_points(), BASE + 60_000_000))
    engine.process(observation(left_pinch_points(), BASE + 160_000_000))

    assert engine.current_gesture.value == "left_pinch"


def test_pinch_clears_thumb_stop_latch() -> None:
    """回归：捏合必须清掉「停住」锁存，否则拖拽时光标冻住。

    2026-10-20 用户实测：HUD 同时显示「正在拖拽」和「●已停住」，但光标不动。
    根因是捏合时 `_allow_cursor_move` 提前 return True 却没清 latch，
    pipeline._advance_motion 一直读到 thumb_stop_latched=True 就反复 cancel
    插值器。
    """

    engine = GestureEngine(AppConfig())
    t = BASE
    for _ in range(8):  # 开门
        engine.process(observation(entry_palm_points(), t))
        t += 60_000_000

    # 拇指往外撇到 spread≈0.95（默认阈值 0.88）→ 进入停住
    open_thumb = list(pointer_points())
    open_thumb[4] = Landmark(0.231, 0.467)
    engine.process(observation(tuple(open_thumb), t))
    assert engine.thumb_stop_latched is True

    # 捏合 → 必须立刻解除停住
    t += 60_000_000
    engine.process(observation(left_pinch_points(), t))
    assert engine.thumb_stop_latched is False


def test_pinch_anchor_freezes_cursor_after_stop() -> None:
    """停住后准备捏合时，光标要被钉住，别让捏合动作把落点带偏。

    2026-10-20 用户要求：拇指张开停住之后，要捏合时光标再多停 1 秒，
    否则捏合时食指一动，光标就移开了。
    """

    engine = GestureEngine(AppConfig(pinch_anchor_hold_ms=10_000))
    t = BASE
    for _ in range(8):
        engine.process(observation(entry_palm_points(), t))
        t += 60_000_000

    # 拇指外撇 → 停住
    open_thumb = list(pointer_points())
    open_thumb[4] = Landmark(0.231, 0.467)
    engine.process(observation(tuple(open_thumb), t))
    assert engine.thumb_stop_latched is True

    # 手指开始收拢（捏合比落在 release~arm 之间）→ 触发锚定
    t += 60_000_000
    closing = list(pointer_points())
    closing[4] = Landmark(0.42, 0.346)   # 到食指尖约 0.146，/掌宽 ≈ 0.60
    engine.process(observation(tuple(closing), t))
    assert engine._pinch_anchor_active() is True

    # 真正捏上：停住锁存解除，但锚定窗口内光标仍必须冻结
    t += 60_000_000
    engine.process(observation(left_pinch_points(), t))
    assert engine.thumb_stop_latched is False
    assert engine.cursor_frozen is True


def test_pinch_anchor_disabled_when_hold_is_zero() -> None:
    """把锚定时长设成 0 就退回旧行为，捏合后立刻能移动。"""

    engine = GestureEngine(AppConfig(pinch_anchor_hold_ms=0))
    t = BASE
    for _ in range(8):
        engine.process(observation(entry_palm_points(), t))
        t += 60_000_000

    open_thumb = list(pointer_points())
    open_thumb[4] = Landmark(0.231, 0.467)
    engine.process(observation(tuple(open_thumb), t))
    t += 60_000_000

    closing = list(pointer_points())
    closing[4] = Landmark(0.42, 0.346)
    engine.process(observation(tuple(closing), t))
    t += 60_000_000

    engine.process(observation(left_pinch_points(), t))
    # 锚定确实关掉了；但「捏住期间继续冻结」是新加的独立保护，
    # 所以这里断言的是锚定窗口，而不是 cursor_frozen。
    assert engine.pinch_anchor_active is False


def test_cursor_stays_frozen_while_pinching() -> None:
    """捏住不放时，即使锚定窗口已过也要继续冻结。

    2026-10-20 用户要求：停住后"这一秒钟可以用捏合"，否则一秒钟一过
    光标又跟着食指飞了。所以冻结范围从"锚定 1 秒"扩展到"捏住期间"。
    """

    engine = GestureEngine(AppConfig(pinch_anchor_hold_ms=0, drag_hold_ms=5_000))
    t = BASE
    for _ in range(8):
        engine.process(observation(entry_palm_points(), t))
        t += 60_000_000

    engine.process(observation(left_pinch_points(), t))
    t += 60_000_000
    engine.process(observation(left_pinch_points(), t))

    assert engine._left_button_down is False   # 还没到 5s，不是拖拽
    assert engine.cursor_frozen is True        # 但捏住期间光标必须钉住


# ---------------------------------------------------------------------------
# 拖拽右移：锚点必须用画面原始坐标，不能用映射后（会被 clamp）的坐标
# ---------------------------------------------------------------------------


def _shift_hand(dx: float) -> tuple:
    """把整只手水平平移，用来模拟"掌心往右推"。"""

    from conftest import left_pinch_points

    # 必须是「捏住」的手型：拖拽分支在 _left_active 里，松开就会走释放路径。
    return tuple(
        Landmark(p.x + dx, p.y, p.z) for p in left_pinch_points()
    )


def test_drag_keeps_moving_right_after_mapping_saturates() -> None:
    """掌心已在映射饱和区时，继续右推仍必须产生位移。

    旧实现把锚点存成 ``self._map(palm)``，而 `_map` 会 clamp 到 [0,1]：
    掌心推到画面右侧后映射值恒为 1.0，delta 恒为 0 ——
    用户实测「只要拖拽，向右拉都拉不动」（2026-10-21，31002 帧记录里
    46 次拖拽全部受影响）。
    """

    from handmouse.gestures import _palm_center

    config = AppConfig()
    config.pointer_margin_x = 0.2
    config.pointer_margin_y = 0.2
    config.drag_gain = 1.0
    engine = GestureEngine(config)
    engine.set_paused(False)

    # 手工置入拖拽态：锚点掌心 x=0.85 已经落在映射饱和区（(0.85-0.2)/0.6 > 1）。
    base = _palm_center(_shift_hand(0.85 - _palm_center(_shift_hand(0)).x))
    assert base.x > 0.8
    engine._left_active = True
    engine._left_started_ns = BASE
    engine._left_button_down = True
    engine._drag_anchor_palm = (base.x, base.y)
    engine._drag_anchor_cursor = (0.5, 0.5)
    engine._last_smoothed = (0.5, 0.5)

    # 再往右推 0.10（画面归一化）
    hand = _shift_hand(0.85 - _palm_center(_shift_hand(0)).x + 0.10)
    commands = engine.process(observation(hand, BASE + 33_000_000))

    moves = [c for c in commands if c.kind == MouseCommandType.MOVE]
    assert moves, "拖拽中必须下发 MOVE"
    assert moves[0].x > 0.5 + 1e-6, "右移被映射饱和吃掉了"


# ---------------------------------------------------------------------------
# 单击是否立即落地（click_immediate）
# ---------------------------------------------------------------------------


def _pinch_then_release(immediate: bool) -> list:
    """捏一下再松开，返回整段产生的指令。"""

    from conftest import left_pinch_points, pointer_points

    config = AppConfig()
    config.entry_gate_enabled = False
    config.click_immediate = immediate
    config.pinch_hold_ms = 70
    engine = GestureEngine(config)
    engine.set_paused(False)

    pinch = left_pinch_points()
    open_hand = pointer_points()
    commands: list = []
    # 8fps 节奏捏住两帧（约 250ms，超过 pinch_hold_ms）
    for i in range(3):
        commands += engine.process(observation(pinch, BASE + i * 125_000_000))
    # 松开
    for i in range(3, 6):
        commands += engine.process(observation(open_hand, BASE + i * 125_000_000))
    return commands


def test_click_immediate_fires_without_waiting_for_double_window() -> None:
    """click_immediate=True 时，松开捏合必须**当场**发 LEFT_CLICK。

    旧实现会先挂起 `double_click_window_ms`(450ms) 等第二次捏合，
    实测该用户等不到反馈就再捏一次，单点被做成了双击（单击仅 4 次、
    双击 24 次），再等不到就按更久变成拖拽（46 次）。
    """

    kinds = [c.kind for c in _pinch_then_release(True)]

    assert MouseCommandType.LEFT_CLICK in kinds
    assert MouseCommandType.LEFT_DOUBLE_CLICK not in kinds


def test_default_path_still_defers_single_click() -> None:
    """默认（click_immediate=False）保持旧语义：不当场发单击，先挂起。"""

    kinds = [c.kind for c in _pinch_then_release(False)]

    assert MouseCommandType.LEFT_CLICK not in kinds


# ---------------------------------------------------------------------------
# 握拳是否退出控制态（fist_exits_gate）
# ---------------------------------------------------------------------------


def _fist_keeps_gate(keep: bool) -> bool:
    """握拳 1 秒后，进入门是否还开着。"""

    from conftest import fist_points

    config = AppConfig()
    config.entry_gate_enabled = True
    config.fist_exits_gate = keep
    engine = GestureEngine(config)
    engine.set_paused(False)
    _enter_gate(engine)
    assert engine._gate_active, "前置条件：门已开"

    fist = fist_points()
    for i in range(20):  # 2 秒，远超 fist_hold_ms
        engine.process(observation(fist, BASE + 400_000_000 + i * 100_000_000))
    return engine._gate_active


def test_fist_exits_gate_by_default() -> None:
    """默认保留旧语义：握拳 = 退出控制态。"""

    assert _fist_keeps_gate(True) is False


def test_fist_can_be_made_harmless_for_daily_use() -> None:
    """关掉后握拳只冻结动作，不再把用户踢出控制态。

    2026-10-21 用户实测：日常使用中手一放松就半握，被判成握拳 →
    HUD 报「未进入控制态」、光标定住，他以为卡死了（31002 帧里 93 次、76 秒）。
    """

    assert _fist_keeps_gate(False) is True


# ---------------------------------------------------------------------------
# 双指滑动导航（食指+中指伸出，其余握起）
# ---------------------------------------------------------------------------


def _two_finger_hand(dx: float = 0.0, dy: float = 0.0) -> tuple:
    """构造双指手势：食指与中指都竖直伸出、指尖同高；无名指小指握着。"""

    from conftest import pointer_points

    points = list(pointer_points())
    # 中指伸直并与食指尖同高（默认姿势里中指是收着的）
    points[10] = Landmark(0.52, 0.45)
    points[11] = Landmark(0.52, 0.32)
    points[12] = Landmark(0.52, 0.20)
    return tuple(Landmark(p.x + dx, p.y + dy, p.z) for p in points)


def _enter_swipe(engine, points: tuple, base_ns: int) -> list:
    """喂两帧双指姿势，让引擎真正进入 SWIPE，返回第二帧的指令。

    判据要求"两次成立落在 ``_SWIPE_CONFIRM_MS``(150ms) 内"（2026-10-22
    第五次修订），单帧只记账不进入——这是为了防止孤立的单帧误报变成
    一次真的滚动。所以任何"要进 SWIPE"的用例都必须喂两帧。
    """

    first = engine.process(observation(points, base_ns))
    assert first == [], "确认窗内的第一帧只记账，不该输出指令"
    second = engine.process(observation(points, base_ns + 40_000_000))
    assert second == [], "确认帧只设锚点，不该输出指令"
    assert engine.current_gesture.value == "swipe", "两帧都没进入双指"
    return second


def _swipe(dx: float, dy: float) -> list:
    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    _enter_swipe(engine, _two_finger_hand(), BASE)
    return engine.process(
        observation(_two_finger_hand(dx, dy), BASE + 200_000_000),
    )


def test_swipe_up_scrolls_down() -> None:
    """手指向上划 → 看下面的内容 → 滚轮向下（负 delta）。"""

    commands = _swipe(0.0, -0.25)

    wheels = [c for c in commands if c.kind == MouseCommandType.WHEEL]
    assert wheels, "向上划应该发 WHEEL"
    assert wheels[0].delta < 0


def test_swipe_down_scrolls_up() -> None:
    commands = _swipe(0.0, +0.25)

    wheels = [c for c in commands if c.kind == MouseCommandType.WHEEL]
    assert wheels and wheels[0].delta > 0


def test_swipe_left_goes_next() -> None:
    """手指向左划 → 看右边的内容 → 下一页。"""

    commands = _swipe(-0.25, 0.0)

    assert MouseCommandType.PAGE_NEXT in [c.kind for c in commands]


def test_swipe_right_goes_previous() -> None:
    commands = _swipe(+0.25, 0.0)

    assert MouseCommandType.PAGE_PREVIOUS in [c.kind for c in commands]


def test_two_finger_does_not_move_cursor() -> None:
    """双指姿势绝不能顺带移动光标（它排在 pointer 之前）。"""

    commands = _swipe(0.0, -0.25)
    assert MouseCommandType.MOVE not in [c.kind for c in commands]


def _rotate_hand(points: tuple, degrees: float, *, pivot: int = 0) -> tuple:
    """绕某个关键点旋转整只手，模拟"手斜着伸出去"。"""

    from math import cos, radians, sin

    cx, cy = points[pivot].x, points[pivot].y
    angle = radians(degrees)
    cos_a, sin_a = cos(angle), sin(angle)
    out = []
    for p in points:
        dx, dy = p.x - cx, p.y - cy
        out.append(
            Landmark(cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a, p.z),
        )
    return tuple(out)


def test_two_finger_detected_when_hand_is_rotated() -> None:
    """双指朝左上斜着伸出去，也必须识别成滑动姿势。

    2026-10-21 用户实测截图中手指是斜的，而当时的判据比较"两指尖 y 坐标"
    （隐含要求手指竖直），于是完全认不出来。判据改成旋转不变的伸展长度后
    这条必须通过。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    tilted = _rotate_hand(_two_finger_hand(), -40)
    _enter_swipe(engine, tilted, BASE)


def test_rotated_pointer_is_still_pointer() -> None:
    """只伸食指、同样斜着伸，仍然必须是移动光标，不能被双指抢走。"""

    from conftest import pointer_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    tilted = _rotate_hand(pointer_points(), -40)
    commands = engine.process(observation(tilted, BASE))

    assert engine.current_gesture.value == "pointer"
    assert MouseCommandType.MOVE in [c.kind for c in commands]


def _side_view_two_finger_hand() -> tuple:
    """侧手双指：食指+中指伸开，无名指/小指收拢但投影近似共线。

    2026-10-21 用户实测截图：侧手时收拢手指的指节投影近似共线，角度判据
    会把它们判成"伸直"，整只手被抢进五指张开待机。收拢必须用伸展长度比
    （指尖折回掌心 → tip→MCP 距离被压短）才能分开。
    """

    points = list(_two_finger_hand())
    # 无名指：13→14→16 排成近似一条直线（角度≈180°，角度判据说"伸直"），
    # 但 tip 离 MCP 很近（收拢，reach≈0.29）。
    points[13] = Landmark(0.58, 0.61)
    points[14] = Landmark(0.585, 0.575)
    points[16] = Landmark(0.59, 0.54)
    # 小指同理。
    points[17] = Landmark(0.66, 0.64)
    points[18] = Landmark(0.662, 0.60)
    points[20] = Landmark(0.665, 0.56)
    return tuple(points)


def test_two_finger_survives_side_view_projection() -> None:
    """侧手双指绝不能被"五指张开待机"抢走（用户 2026-10-21 实测 bug）。"""

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    commands = _enter_swipe(engine, _side_view_two_finger_hand(), BASE)

    assert engine.current_gesture.value == "swipe", (
        "侧手双指被误判成五指张开待机（角度判据把收拢指认成伸直）"
    )
    assert commands == []


def test_frontal_two_finger_is_recognized() -> None:
    """正对镜头持双指必须能识别——用户复报"手移到某个高度就失灵"的根因回归。

    旧判据是 ``|MCP→tip| ÷ 掌宽(5→17) >= 1.3``。它的分母含着一个**会被透视
    压缩的量**：侧手时掌宽只剩一半，比值虚高到 1.3 以上；正对镜头时掌宽是满
    宽，同一根伸直的手指只能算出

    * 食指 ``reach_index  = 1.150``
    * 中指 ``reach_middle = 1.227``

    双双卡在 1.3 以下 → 手指明明伸直了，"双指"却永远不触发。用户描述的
    "手移到某个高度以下就失灵"其实就是手腕角度变了、掌宽投影变宽，不是位置
    本身。本用例用 掌宽/手长 = 1.043 的正手夹具把这条钉死。
    """

    from handmouse.gestures import _distance_2d, _is_two_finger, _two_finger_diag

    points = frontal_two_finger_points()
    diag = _two_finger_diag(points)
    assert diag[0] > 0, "夹具退化，掌宽或手长为零"
    assert _is_two_finger(diag), (
        f"新判据都没认出正手双指：peak={diag[0]:.3f} "
        f"i={diag[1]:.3f} m={diag[2]:.3f} r={diag[3]:.3f} p={diag[4]:.3f} "
        f"a_m={diag[5]:.1f}"
    )

    # 反证：这个夹具确实会被旧判据拒掉，否则本用例就不算回归。
    palm_width = _distance_2d(points[5], points[17])
    hand_length = _distance_2d(points[0], points[9])
    assert palm_width / hand_length > 0.9, "夹具不是正手（掌宽没被压扁）"
    for mcp, tip, name in ((5, 8, "食指"), (9, 12, "中指")):
        reach = _distance_2d(points[mcp], points[tip]) / palm_width
        assert reach < 1.3, (
            f"{name}旧判据 reach={reach:.3f} 竟然 >= 1.3，"
            f"说明夹具没复现出正手失灵、本用例失去回归意义"
        )

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    _enter_swipe(engine, points, BASE)
    assert engine.current_gesture.value == "swipe", "正手双指没有进入滑动模式"


def test_half_curled_ring_finger_still_enters_swipe() -> None:
    """无名指只是**自然半弯**（没真收拢）也必须认作双指——0.65→0.80 的边界。

    2026-10-22 把最差的那几次录制逐帧拆开才发现：他失败帧的无名指比值中位是
    0.871、小指 0.853，也就是**半弯而不是收拢**。旧阈值 0.65 把这类真实姿势
    全拒了，占非"移动姿势"失败帧的 60~69%。

    本夹具的 ``t_r/peak = 0.762``：旧阈值 0.65 下必被拒，新阈值 0.80 下放行。
    抓到"为了躲开五指张开而把收拢闸收得过紧"这种回归。
    """

    from conftest import swipe_points

    points = list(swipe_points())
    points[16] = Landmark(0.58, 0.30)

    diag = _two_finger_diag(points)
    assert 0.65 < diag[3] <= 0.80, f"夹具的无名指比值 {diag[3]:.3f} 不在新旧阈值之间"
    assert _is_two_finger(diag), "半弯的无名指又被当成伸直的拒掉了"

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    _enter_swipe(engine, tuple(points), BASE)
    assert engine.current_gesture.value == "swipe"


def test_finger_straightness_path_rescues_pose_the_distance_ruler_rejects() -> None:
    """并联的第二把尺子必须能单独抓住主判据漏掉的帧。

    2026-10-22 用户提出「直接看中指和食指伸出来就行了」。逐指判据
    （伸直度 + 指根角）单独用**更差**（零误报下 73.8% vs 84.9%），
    但它和主判据的盲区互补——``collect-20261022-161133`` 里有一批帧
    两指都很直、指根角也正常，却因为无名指被算成"没完全收拢"而被主判据
    拒掉，这一批只有逐指判据能救（该会话 71.2% → 87.5%）。

    本用例构造同一类帧：无名指半弯到 ``t_r/peak`` 超过收拢闸 0.80，
    但它的**伸直度**明显偏低 → 主判据拒、并联路径收。
    """

    from conftest import swipe_points

    points = list(swipe_points())
    # 无名指指尖拉到 ``t_r/peak = 0.811``（越过 0.80 的收拢闸），但三节
    # 明显折着（伸直度只有 0.767）→ 主判据拒、逐指判据收。
    points[16] = Landmark(0.60, 0.28)
    points = tuple(points)

    diag = _two_finger_diag(points)
    assert not _is_two_finger(diag), "主判据竟然放行了，本用例失去意义"
    assert _two_finger_by_fingers(points), "并联路径没能接住主判据漏掉的帧"

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    _enter_swipe(engine, points, BASE)
    assert engine.current_gesture.value == "swipe"


def test_open_palm_not_eaten_by_finger_straightness_path() -> None:
    """五指张开时四指伸直度都 ≈1.00，逐指路径绝不能说"无名指没伸出来"。

    这里曾踩过一个真坑：无名指/小指的"伸没伸出"若也用**指根角**判断，
    外侧手指天然呈扇形外张，这个方向天然与手指方向成大角——实测确实张开
    的手会被判成"指尖没收拢"，五指张开待机被误吃 **44.4%**。
    所以外侧两指只看伸直度（五指张开时四指都 ≈1.00，没有这个偏向）。

    顺带钉住上面那句"扇形外张"：全伸直的手里，小指指根角只有 79°，
    刚刚越过 ``_FINGER_ROOT_MIN``(75) —— 说明这条线对指根角一点都不宽裕。
    """

    from conftest import entry_palm_points

    hands = entry_palm_points()
    feats = _finger_feats(hands)
    for name in ("index", "middle", "ring", "pinky"):
        assert feats[name][0] >= 0.99, f"{name} 伸直度只有 {feats[name][0]:.3f}，夹具不是真张开"
    assert feats["pinky"][1] < 85.0, "小指指根角没复现出扇形外张偏低的现象"

    assert not _two_finger_by_fingers(hands), "五指张开被逐指路径吃成了双指"


def test_foreshortened_straight_middle_is_not_two_finger() -> None:
    """三节共线但指尖还缩在掌心附近（手指指着镜头）不得算"伸出来"。

    这种姿势伸直度能到 ≈0.98，光看伸直度会被骗；``_FINGER_MIN_REACH``
    就是为此加的——实测代价只有 0.1 个百分点。
    """

    from conftest import pointer_points

    points = list(pointer_points())
    points[10] = Landmark(0.50, 0.48)
    points[11] = Landmark(0.51, 0.46)
    points[12] = Landmark(0.52, 0.44)
    points = tuple(points)

    feats = _finger_feats(points)
    assert feats["middle"][0] >= 0.90, "夹具的中指不够直，测不到「指向镜头」这条"

    assert not _two_finger_by_fingers(points), "指向镜头的短中指被当成伸出来了"


def test_world_two_finger_accepts_index_middle_only() -> None:
    """3D 判据：食指+中指伸直、无名指小指蜷着 → 是双指。

    这是用户 2026-10-22 的原话——「直接看我的中指食指手指本体，不要靠 2D」。
    ``world_landmarks`` 是米制 3D 骨架，判据在真三维里量，与手腕角度无关。
    """

    from conftest import world_hand

    hand = world_hand()
    feats = _finger3d_feats(hand)
    assert feats["index"] >= 0.99 and feats["middle"] >= 0.99
    assert feats["ring"] <= 0.60 and feats["pinky"] <= 0.60
    assert _two_finger_by_world(hand)


def test_world_two_finger_rejects_open_palm() -> None:
    """五指张开时四指 3D 伸直度都是 1.0 —— 绝不能判成双指。"""

    from conftest import world_hand

    hand = world_hand(ring_extended=True, pinky_extended=True)
    feats = _finger3d_feats(hand)
    for name in ("index", "middle", "ring", "pinky"):
        assert feats[name] >= 0.99
    assert not _two_finger_by_world(hand)


def test_world_two_finger_rejects_folded_middle() -> None:
    """中指蜷着就不算双指（哪怕食指笔直）。"""

    from conftest import world_hand

    assert not _two_finger_by_world(world_hand(middle_extended=False))


def test_world_two_finger_needs_full_landmarks() -> None:
    """3D 数据缺失（回放/测试/局部跟踪降级）时必须老实返回 False，由 2D 兜底。"""

    from conftest import world_hand

    assert not _two_finger_by_world(())
    assert not _two_finger_by_world(world_hand()[:20])


def test_engine_enters_swipe_from_world_landmarks_alone() -> None:
    """接线证明：2D 判据全拒，但 3D 说是双指 → 引擎必须进滑动。

    用"2D 是关键点姿势、3D 是双指"这种不一致的合成输入，只为证明 3D 这条路
    真的接进了状态机、而不是躺在模块里没人调。真实输入里两者是一致的，
    这时 3D 只是把"2D 因为投影问题判断不了的帧"救回来。
    """

    from conftest import pointer_points, world_hand

    two_d = pointer_points()
    assert not _two_finger_hit(two_d, _two_finger_diag(two_d)), "2D 判据不该放行这个姿势"

    world = world_hand()
    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    first = engine.process(observation(two_d, BASE, world=world))
    assert first == [], "确认窗内第一帧只记账"
    engine.process(observation(two_d, BASE + 40_000_000, world=world))
    assert engine.current_gesture.value == "swipe", "3D 判据没接进状态机"


def test_micro_deadband_holds_cursor_when_finger_barely_moves() -> None:
    """瞄准死区：手指只动一丁点时不出新落点，光标钉住（用户 2026-10-22 要求）。

    原话：「当我的手指移动微弱的时候，就要控制抖动，不然我瞄准小目标太难了」。
    """

    from conftest import pointer_points, translate_points

    config = AppConfig(cursor_micro_deadband=0.05)
    engine = GestureEngine(config)

    first = engine.process(observation(pointer_points(), BASE))
    assert kinds(first) == [MouseCommandType.MOVE]
    anchor = (first[0].x, first[0].y)

    # 指尖只挪 0.002（映射后远小于 0.05）→ 一帧 MOVE 都不发。
    jittered = translate_points(pointer_points(), dx=0.002)
    suppressed = engine.process(observation(jittered, BASE + 33_000_000))
    assert suppressed == [], "微小位移不该发出 MOVE，否则光标会在目标上抖"
    assert engine.micro_locked is True


def test_micro_deadband_releases_immediately_on_real_movement() -> None:
    """死区只是"不抖"，不能变成"粘住"：一挪大就立刻跟上，且无需等待。"""

    from conftest import pointer_points, translate_points

    config = AppConfig(cursor_micro_deadband=0.05)
    engine = GestureEngine(config)
    engine.process(observation(pointer_points(), BASE))

    moved = translate_points(pointer_points(), dx=0.20)
    commands = engine.process(observation(moved, BASE + 33_000_000))

    assert kinds(commands) == [MouseCommandType.MOVE], "真实移动被死区吞掉了"
    assert engine.micro_locked is False


def test_micro_deadband_never_swallows_clicks() -> None:
    """死区只作用于 MOVE。点击是事件，被吞掉就等于"手一抖点不中"。"""

    from conftest import left_pinch_points, pointer_points

    config = AppConfig(cursor_micro_deadband=0.05)
    engine = GestureEngine(config)
    engine.process(observation(pointer_points(), BASE))

    # 捏合（手指几乎没位移）+ 释放 → 仍然必须出单击。
    engine.process(observation(left_pinch_points(), BASE + 33_000_000))
    released = engine.process(observation(pointer_points(), BASE + 300_000_000))
    clicked = engine.process(observation(pointer_points(), BASE + 700_000_000))

    assert MouseCommandType.LEFT_CLICK in kinds(clicked), "死区把单击吞了"


def test_micro_deadband_resets_anchor_when_an_event_occurs() -> None:
    """出现事件（按下/点击）时必须重置死区锚点，否则拖拽的第一下会被吞。"""

    from conftest import left_pinch_points, pointer_points, translate_points

    # 用线上真实档位（0.0025），不要用夸张值：死区叠在 OneEuro 平滑之后，
    # 死区过大时"真实移动的第一帧"本身还没被滤到超过死区，会看起来像被吞掉。
    config = AppConfig(cursor_micro_deadband=0.0025, drag_hold_ms=300)
    engine = GestureEngine(config)
    engine.process(observation(pointer_points(), BASE))

    engine.process(observation(left_pinch_points(), BASE + 33_000_000))
    entered = engine.process(observation(left_pinch_points(), BASE + 350_000_000))
    assert kinds(entered)[0] == MouseCommandType.LEFT_DOWN
    # 按下那一批是"事件"，锚点被重置 → 同一批里的 MOVE 必须还在。
    assert MouseCommandType.MOVE in kinds(entered), "左键按下后光标第一下不能不动"

    # 按下之后真实平移，必须持续跟手。
    moved = translate_points(left_pinch_points(), dx=0.05)
    commands = engine.process(observation(moved, BASE + 400_000_000))
    assert MouseCommandType.MOVE in kinds(commands)


def test_true_open_palm_not_eaten_by_swipe() -> None:
    """真正的五指张开必须仍是待机，不能被排在前面的双指判定抢走。"""

    from conftest import open_palm_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    engine.process(observation(open_palm_points(), BASE))
    # 待机要连续保持 FIVE_OPEN_CONFIRM_MS(300ms) 才成立。
    engine.process(observation(open_palm_points(), BASE + 350_000_000))

    assert engine.current_gesture.value == "open_palm"


def test_dwell_signal_engages_snap_only_after_pause(arrival_clock) -> None:
    """光标原地停留满阈值才点亮"预捏合磁吸窗口"（平时不吸）。

    2026-10-21 用户选定：停留 = 要捏合了，磁吸只在临门一脚时工作。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    # 第一帧建立 POINTER 状态。
    engine.process(observation(pointer_points(), BASE))
    assert engine.current_gesture.value == "pointer"

    # 刚进入控制、还没停留够久：磁吸窗口应关闭。
    assert engine.snap_engaged is False

    # 同一姿势原地停留，累计超过 pinch_pre_dwell_ms(1000ms)。
    for _ in range(5):
        arrival_clock.advance_ms(250)  # 合计 1250ms > 1000ms
        engine.process(observation(pointer_points(), arrival_clock.now_ns))

    assert engine.snap_engaged is True


def test_dwell_resets_when_pointer_moves(arrival_clock) -> None:
    """停留窗口激活后，手指一动（位移超过 epsilon）立即解除磁吸窗口。"""

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    engine.process(observation(pointer_points(), BASE))
    for _ in range(5):
        arrival_clock.advance_ms(250)
        engine.process(observation(pointer_points(), arrival_clock.now_ns))
    assert engine.snap_engaged is True

    # 手指平移 0.10（>> pinch_pre_dwell_epsilon 的 0.005），应重置停留检测。
    arrival_clock.advance_ms(250)
    engine.process(
        observation(translate_points(pointer_points(), dx=0.10), arrival_clock.now_ns),
    )
    assert engine.snap_engaged is False


def test_snap_engaged_gated_to_pointer_state() -> None:
    """snap_engaged 只在 POINTER 状态下有意义，其它手势一律不吸。"""

    from handmouse.contracts import GestureType

    config = AppConfig()
    engine = GestureEngine(config)
    engine._dwell_active = True

    engine.current_gesture = GestureType.OPEN_PALM
    assert engine.snap_engaged is False
    engine.current_gesture = GestureType.SWIPE
    assert engine.snap_engaged is False

    engine.current_gesture = GestureType.POINTER
    assert engine.snap_engaged is True


def test_update_dwell_timing_math() -> None:
    """白盒：_update_dwell 的"位移阈值 + 时间阈值"逻辑自洽。"""

    config = AppConfig()
    config.pinch_pre_dwell_ms = 1000
    config.pinch_pre_dwell_epsilon = 0.005
    engine = GestureEngine(config)

    # 第一次调用永远是"刚落位"，重置计时、未激活。
    engine._update_dwell((0.5, 0.5), 0)
    assert engine._dwell_active is False

    # 位置没动、时间未到：仍不激活。
    engine._update_dwell((0.5, 0.5), 500_000_000)
    assert engine._dwell_active is False

    # 时间到了：激活。
    engine._update_dwell((0.5, 0.5), 1_100_000_000)
    assert engine._dwell_active is True

    # 位移超过 epsilon：清零、未激活。
    engine._update_dwell((0.6, 0.5), 1_200_000_000)
    assert engine._dwell_active is False


def test_dwell_lights_up_under_slow_drift(arrival_clock) -> None:
    """回归（2026-10-23）：真实手"停住"时仍有缓慢漂移，判据必须认得出。

    他原话：「我鼠标静止一秒钟也没有出现捏合的趋向」。用他自己 460 秒
    真实日志回放：旧阈值 0.005（≈9.6px）**激活 0 次**，连整段里最安静的
    1 秒（漂移 24.8px，p1=33.8px）都达不到——手在空中不可能像素级静止。

    这里复现"真实停住"的形状：**来回微幅晃动**（幅度 0.018 画面坐标，
    约等于实测 p1 量级的漂移），不是一路匀速走。旧判据必挂，新判据必须亮。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    engine.process(observation(pointer_points(), arrival_clock.now_ns))

    # 1.4 秒的微幅晃动（43 帧 × 33ms，约 1.4s 一个来回）。
    for index in range(1, 44):
        arrival_clock.advance_ms(33)
        offset = 0.018 * sin(2 * pi * index / 43)
        engine.process(
            observation(
                translate_points(pointer_points(), dx=offset),
                arrival_clock.now_ns,
            ),
        )

    assert engine.dwell_active is True, "微幅晃动下停留也必须能点亮"
    assert engine.snap_engaged is True, "停留亮了磁吸窗口就该开"

    # 真正开始移动（一帧 0.05，远超短窗阈值 0.025）必须立刻解除。
    arrival_clock.advance_ms(33)
    engine.process(
        observation(
            translate_points(pointer_points(), dx=0.05),
            arrival_clock.now_ns,
        ),
    )
    assert engine.dwell_active is False, "他明明动了，停留必须立刻解除"


def test_pinch_anchor_alone_does_not_open_snap_window(arrival_clock) -> None:
    """回归（2026-10-23）：光有捏合趋向锚定，**不许**打开磁吸窗口。

    这就是「夹在两个小图标中间左右弹跳」的机制：捏合比在他正常移动时
    就会频繁掉到 0.90 以下（实测 move 类帧约 10% 命中），若单靠锚定开窗，
    光标会被反复拽到某个控件中心、一秒后又松回随手位置 → 一左一右。
    现在锚定只负责**冻结**光标，开磁吸窗口还要求"已经停稳"。
    """

    from handmouse.contracts import GestureType

    config = AppConfig()
    engine = GestureEngine(config)
    engine.current_gesture = GestureType.POINTER
    engine._pinch_anchor_ns = 1
    assert engine.pinch_anchor_active is True

    # 光标刚动过（停留计时刚清零）→ 窗口不开。
    engine._dwell_started_ns = arrival_clock.now_ns
    assert engine.snap_engaged is False, "移动中的捏合不该把光标拽到控件上"

    # 停稳超过 pinch_snap_still_ms 之后才允许（锚定仍在 1000ms 持有窗口内）。
    arrival_clock.advance_ms(config.pinch_snap_still_ms + 50)
    assert engine.pinch_anchor_active is True, "前提：锚定还没过期"
    assert engine.snap_engaged is True


# ---------------------------------------------------------------------------
# 双指"滑轮式"连续滚动（2026-10-22 用户要求）
# ---------------------------------------------------------------------------

def _swipe_feeder(offset_x: float = 0.0, offset_y: float = 0.0):
    """构造一个从"掌心在画面正中"出发的双指序列喂帧器。

    从正中出发只是为了让 dy 的取值和画面归一化坐标对齐、用例好写；
    **静止/解锁已与画面位置无关**（2026-10-22 移除了回中解锁），
    所以这里是否在正中不影响任何判定。
    """

    from handmouse.gestures import _palm_center

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    base = _palm_center(_two_finger_hand(0.0, 0.0))
    home = (offset_x + (0.5 - base.x), offset_y + (0.5 - base.y))
    return engine, home


def _wheels(commands):
    return [c for c in commands if c.kind == MouseCommandType.WHEEL]


def test_pump_scroll_is_continuous_and_ignores_return_stroke() -> None:
    """连续向上抽拉必须持续滚动，且"回手"不能被算成反向滚动。

    这是整个改动的核心：抽拉式操作每一划之间手指必然要向下复位，
    若复位被计入，用户抽得再快也会被自己抵消。
    """

    engine, (hx, hy) = _swipe_feeder()
    engine.process(observation(_two_finger_hand(hx, hy), BASE))

    forward: list = []
    for i in range(1, 7):
        forward.extend(
            engine.process(
                observation(_two_finger_hand(hx, hy - 0.05 * i), BASE + i * 33_000_000),
            ),
        )
    up = _wheels(forward)
    assert len(up) >= 2, "应该是逐帧连续给量，而不是整段只发一次"
    assert all(c.delta < 0 for c in up), "向上划应当看下方内容（负 delta）"

    # 回手向下复位：不得产生任何正向滚轮。
    returning: list = []
    for i in range(7, 12):
        dy = -0.30 + 0.05 * (i - 6)
        returning.extend(
            engine.process(
                observation(_two_finger_hand(hx, hy + dy), BASE + i * 33_000_000),
            ),
        )
    assert _wheels(returning) == [], "回手动作被算成了滚动，连续抽拉会自我抵消"


def test_scroll_amount_scales_with_stroke_speed() -> None:
    """划得越快/越远 → 滚得越多；这是"滑动速度与距离决定内容量"的核心。"""

    def scroll_total(step: float, frames: int = 6) -> int:
        engine, (hx, hy) = _swipe_feeder()
        engine.process(observation(_two_finger_hand(hx, hy), BASE))
        total = 0
        for i in range(1, frames + 1):
            commands = engine.process(
                observation(_two_finger_hand(hx, hy - step * i), BASE + i * 33_000_000),
            )
            total += sum(abs(c.delta) for c in _wheels(commands))
        return total

    slow = scroll_total(0.03)
    fast = scroll_total(0.09)

    assert slow > 0, "小幅抽拉也该有滚动"
    assert fast > slow * 2, "大幅/快速抽拉必须滚得明显更多"


def test_passing_through_center_does_not_unlock_direction(arrival_clock) -> None:
    """向下划穿过画面中央时不得解锁滚动方向（2026-10-22 用户实测回归）。

    用户原话："把双指指向中央静止改掉，因为我向下划很容易指到中央"。
    旧逻辑把"掌心路过画面中央并停留 swipe_rest_hold_ms(180ms)"当成"回中解锁"，
    正在滚动的方向被半路清零 → 滚动断掉/乱跳，正是他向下划失灵的原因。
    现在解锁只看"停手超时"，穿过中央不再影响方向。
    """

    engine, (hx, hy) = _swipe_feeder()

    def feed(dy: float, ms: float = 33.0):
        arrival_clock.advance_ms(ms)
        return engine.process(
            observation(_two_finger_hand(hx, hy + dy), arrival_clock.now_ns),
        )

    feed(-0.30)  # 从画面中央上方起手

    # 缓慢向下划、穿过中央：围绕中央（±0.06）停留约 7 帧 ≈ 230ms，
    # 已超过旧的 180ms 解锁阈值 —— 旧实现会在这里把方向清零。
    cmds: list = []
    for i in range(1, 21):
        cmds.extend(feed(-0.30 + 0.02 * i))

    wheels = _wheels(cmds)
    assert wheels, "向下划应该有滚动"
    assert all(c.delta > 0 for c in wheels), "向下划只能正向滚动，不能出现反向"
    assert engine._swipe_dir == 1, "路过画面中央不得解锁滚动方向"


def test_direction_unlocks_after_long_idle(arrival_clock) -> None:
    """停手超时是唯一解锁途径：停够 swipe_window_ms 后，反向划即可反向滚动。"""

    engine, (hx, hy) = _swipe_feeder()
    engine.process(observation(_two_finger_hand(hx, hy), 0))

    for i in range(1, 7):
        arrival_clock.advance_ms(33)
        engine.process(
            observation(_two_finger_hand(hx, hy - 0.05 * i), arrival_clock.now_ns),
        )
    assert engine._swipe_dir == -1

    # 静止超过 swipe_window_ms（默认 800ms）
    arrival_clock.advance_ms(900)
    engine.process(
        observation(_two_finger_hand(hx, hy - 0.30), arrival_clock.now_ns),
    )
    assert engine._swipe_dir == 0

    # 解锁后向下划 → 应当变成正向滚动（反向成功）
    down: list = []
    for i in range(1, 7):
        arrival_clock.advance_ms(33)
        down.extend(
            engine.process(
                observation(
                    _two_finger_hand(hx, hy - 0.30 + 0.05 * i),
                    arrival_clock.now_ns,
                ),
            ),
        )
    assert _wheels(down) and all(c.delta > 0 for c in _wheels(down))


# ---------------------------------------------------------------------------
# 双指判据鲁棒性（2026-10-22 用户实机截图：V 字双指被判成"只伸食指"）
# ---------------------------------------------------------------------------

def _flicker_two_finger() -> tuple:
    """双指姿势，但食指尖被运动模糊"压短"到过不了进入判据。

    真实抖动最常见的形态不是"无名指突然伸直"，而是**食指在快速挥动时
    指尖位置被糊短**：``t_i/peak`` 从 1.0 掉到 0.689（< ``_SWIPE_TIP_MAIN``
    0.70），而 ``peak`` 仍有 2.256（≥ 0.80 进入闸、≥ 0.45 持有闸）。

    用于滞回测试：滚动中进入判据失败、但手仍半张着 → 必须持有而非掉回移动。
    """

    from conftest import swipe_points

    points = list(swipe_points())
    points[8] = Landmark(0.42, 0.36)
    return tuple(points)


def _v_sign_projected() -> tuple:
    """用户实机的 V 字双指：收拢指投影呈"半伸直"。

    旧判据（÷ 掌宽、绝对阈值）会把它漏掉（判成"只伸食指"，光标跟着飞）；
    相对长度判据放行。当前 ``t_r/peak = 0.634``，在收拢闸 0.80 之内。
    """

    from conftest import swipe_points

    points = list(swipe_points())
    # 无名指指尖 16 放在 (0.58, 0.36)：
    #   * 旧判据（÷ 掌宽）|13→16|/掌宽 = 1.027 —— **超过**当时的绝对阈值 1.0
    #   * 相对判据 tr = 0.634 —— 在收拢闸 0.80 之内
    # 这正是"侧视下收拢指投影天然偏长"的真实情形，用来钉住这类帧不得漏判。
    points[16] = Landmark(0.58, 0.36)
    return tuple(points)


def test_v_sign_with_projected_curled_fingers_still_swipe() -> None:
    """收拢指投影 reach 超过旧的绝对阈值 1.0 时，双指仍必须被认出。"""

    from conftest import translate_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    v_sign = _v_sign_projected()
    engine.process(observation(v_sign, BASE))
    engine.process(
        observation(translate_points(v_sign, 0.0, -0.25), BASE + 200_000_000),
    )

    assert engine.current_gesture.value == "swipe", (
        "V 字双指被误判成'只伸食指'（收拢判据的绝对阈值太脆）"
    )


def test_bent_middle_cannot_enter_swipe_even_with_long_projection(monkeypatch) -> None:
    """弯掉的中指就算投影 reach ≥1.3 也绝不能触发双指（2026-10-22 14:17 事故）。

    用户原话："我用这个角度向右划就会触发……一只手指为什么也能触发"。
    他只伸食指、中指半弯（指节翘起），|MCP→tip| 投影照样 ≥1.3，长度判据
    全体失守、模型也报 swipe@0.91。修复依据他的观察："每个点都跟着我的
    关节，系统是能知道我在弯曲的"——中指改用**关节夹角**判伸直：
    本测试中指 PIP 处弯成 90°（MCP→PIP 向上 0.28、PIP→TIP 向右 0.28，
    |MCP→tip|≈0.396 → reach≈1.63 ≥1.3），几何与模型都必须否决。
    """

    from conftest import pointer_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    monkeypatch.setattr(engine, "_model_label", lambda points, now: "swipe")

    points = list(pointer_points())
    points[10] = Landmark(0.50, 0.32)
    points[11] = Landmark(0.64, 0.32)
    points[12] = Landmark(0.78, 0.32)

    commands = engine.process(observation(tuple(points), BASE))

    assert engine.current_gesture.value == "pointer"
    assert MouseCommandType.MOVE in [c.kind for c in commands]


def test_model_cannot_extend_swipe_hold_beyond_window(monkeypatch, arrival_clock) -> None:
    """模型票**不能无限续命**：持有窗只有时间一条出路，模型说了不算。

    2026-10-22 第五次修订撤掉了旧的 ``or model_label == "swipe"`` 分支。
    理由：持有窗的语义是"吞掉判据抖动"（``swipe_hold_ms``），而模型在手指折起来时
    照样会自信地报 swipe——留着那条分支等于让它把用户永久按在滑动态，
    光标一直冻着。撤回后持有时长严格由 ``swipe_hold_ms`` 决定。
    """

    from conftest import swipe_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    monkeypatch.setattr(engine, "_model_label", lambda points, now: "swipe")

    good = swipe_points()
    _enter_swipe(engine, good, BASE)
    arrival_clock.advance_ms(66)
    engine.process(observation(good, arrival_clock.now_ns))
    assert engine.current_gesture.value == "swipe"

    # 中指在 PIP 处折成 90° 往侧面翘：指尖离掌心依旧很远（tm/peak 高达
    # 0.9 以上），模型也说 swipe —— 但持有窗到期后必须放行。
    points = list(swipe_points())
    points[10] = Landmark(0.50, 0.40)
    points[11] = Landmark(0.59, 0.40)
    points[12] = Landmark(0.68, 0.40)
    bent = tuple(points)

    # 持有窗内：继续算滑动（吞抖动），这是故意的。
    arrival_clock.advance_ms(150)
    engine.process(observation(bent, arrival_clock.now_ns))
    assert engine.current_gesture.value == "swipe", "持有窗内不该掉回移动"

    # 超过 swipe_hold_ms 且判据仍不成立 → 必须放行，模型无权续命。
    arrival_clock.advance_ms(config.swipe_hold_ms)
    engine.process(observation(bent, arrival_clock.now_ns))
    assert engine.current_gesture.value != "swipe", (
        "中指折起来了还被按在滑动态（模型不得续命）"
    )


def test_swipe_holds_through_criterion_flicker() -> None:
    """滚动中判据抖动不得掉回"移动"——那会把光标瞬移到食指位置。"""

    from conftest import swipe_points, translate_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    good = swipe_points()
    engine.process(observation(good, BASE))
    engine.process(
        observation(translate_points(good, 0.0, -0.15), BASE + 33_000_000),
    )
    assert engine.current_gesture.value == "swipe"

    # 判据抖动帧：无名指被撑直 → 双指判定失败；但中指仍伸直 → 必须持有。
    flicker = _flicker_two_finger()
    for i in range(2, 5):
        commands = engine.process(
            observation(
                translate_points(flicker, 0.0, -0.02 * i),
                BASE + i * 33_000_000,
            ),
        )
        assert engine.current_gesture.value == "swipe", "判据抖动不该掉回移动"
        assert MouseCommandType.MOVE not in kinds(commands), (
            "持有帧绝不能移动光标（光标乱飞的根源）"
        )

    # 判据恢复 → 无缝继续滚动
    commands = engine.process(
        observation(translate_points(good, 0.0, -0.30), BASE + 6 * 33_000_000),
    )
    assert engine.current_gesture.value == "swipe"
    assert any(c.kind == MouseCommandType.WHEEL for c in commands)


def test_swipe_hold_expires_back_to_pointer(arrival_clock) -> None:
    """持有窗口过期后必须正常掉回移动，不能永久卡在双指状态。"""

    from conftest import swipe_points, translate_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    good = swipe_points()
    engine.process(observation(good, 0))
    arrival_clock.advance_ms(33)
    engine.process(
        observation(translate_points(good, 0.0, -0.15), arrival_clock.now_ns),
    )
    assert engine.current_gesture.value == "swipe"

    # 超过 swipe_hold_ms 的抖动 → 掉回移动。这里从配置读时长，避免以后
    # 再调 swipe_hold_ms 时这条用例变成"其实没越过窗口"的假通过。
    arrival_clock.advance_ms(config.swipe_hold_ms + 80)
    commands = engine.process(
        observation(translate_points(_flicker_two_finger(), 0.0, -0.06), arrival_clock.now_ns),
    )
    assert engine.current_gesture.value == "pointer"


def test_swipe_holds_through_middle_fold_within_hold_window() -> None:
    """收中指不再是"立刻"放行——这是 2026-10-22 第五次修订**故意**换来的取舍。

    旧实现把"中指关节角弯了"当成立即放行通道。但它和进入判据高度相关：
    判据抖动的那几帧，恰恰就是中指看起来弯了的那几帧。实测把形状闸加进
    持有判据后，双指可用度从 81.2% 掉回 65.0%（``scripts/audit_swipe_final.py``）
    ——持有等于失效。所以现在持有只判断"手还半张着"（``peak ≥ 0.45``），
    放行靠 ``swipe_hold_ms`` 兜底，最长就是那一个窗口（当前 450ms）。

    这条用例把这个取舍钉死：收中指后**仍在持有窗内**必须继续算 SWIPE
    （否则长划动会掉帧、光标瞬移）；超过窗口必须回移动（见上一条用例）。
    """

    from conftest import pointer_points, swipe_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    _enter_swipe(engine, swipe_points(), BASE)

    commands = engine.process(observation(pointer_points(), BASE + 66_000_000))
    assert engine.current_gesture.value == "swipe", "持有窗内不该掉回移动"
    assert not any(c.kind == MouseCommandType.MOVE for c in commands), (
        "持有帧绝不能移动光标（光标乱飞的根源）"
    )


def test_swipe_releases_immediately_when_hand_closes() -> None:
    """手收起来（四指都回到掌心）→ 立即离开滑动态，不等持有窗到期。

    持有窗只在"手还张着"时成立：``peak``（四指最长指尖到掌心 ÷ 手长）
    掉到 0.45 以下就说明他已经放下手了。注意"四指收进掌心"在几何上等价于
    握拳，所以这一帧会先被握拳分支接走——这同样算立即放行（关键是**没有**
    继续按在滑动态、也没有继续发滚轮）。
    """

    from conftest import pointer_points, swipe_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    _enter_swipe(engine, swipe_points(), BASE)

    folded = list(pointer_points())
    for tip, x, y in ((8, 0.56, 0.60), (12, 0.52, 0.60), (16, 0.58, 0.62), (20, 0.60, 0.66)):
        folded[tip] = Landmark(x, y)

    commands = engine.process(observation(tuple(folded), BASE + 66_000_000))
    assert engine.current_gesture.value != "swipe", "手已经收起来了还按在滑动态"
    assert not [c for c in commands if c.kind == MouseCommandType.WHEEL], (
        "收手那一刻不该再产生滚轮"
    )


def test_swipe_releases_on_confirmed_open_palm() -> None:
    """滚动中张开五指 = 要进待机：确认满 300ms 后放行。

    2026-10-22 起"进待机"必须过时序确认（单帧误报会把光标冻住，见
    ``_confirm_five_open``），所以不再是"立即"。但这 300ms **不会多滚**：
    滚动量只来自掌心位移，张开手时掌心是停住的。退出待机仍然是即时的。
    """

    from conftest import open_palm_points, swipe_points, translate_points

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    engine.process(observation(swipe_points(), BASE))
    engine.process(
        observation(translate_points(swipe_points(), 0.0, -0.15), BASE + 33_000_000),
    )
    assert engine.current_gesture.value == "swipe"

    # 第一帧张开：仍在确认窗口内，悬置（不跳光标、也不立刻进待机）。
    engine.process(observation(open_palm_points(), BASE + 66_000_000))
    assert engine.current_gesture.value != "open_palm"

    # 连续保持超过确认窗口 → 放行进入待机。
    engine.process(observation(open_palm_points(), BASE + 400_000_000))
    assert engine.current_gesture.value == "open_palm", "确认满 300ms 后应进待机"


# ---------------------------------------------------------------------------
# 立即落地模式下：双击必须把光标送回第一次的落点（2026-10-22）
# ---------------------------------------------------------------------------


def _click_immediate_engine() -> GestureEngine:
    config = AppConfig()
    config.entry_gate_enabled = False
    config.click_immediate = True
    config.double_click_window_ms = 700
    engine = GestureEngine(config)
    engine.set_paused(False)
    return engine


def _pinch_episode(engine, points, start_ns: int):
    """捏住 → 松开，返回这一整段产生的指令。"""

    commands = []
    # 捏住（拇指贴到食指尖）
    tight = list(points)
    tight[4] = Landmark(tight[8].x + 0.05, tight[8].y)
    commands += engine.process(observation(tuple(tight), start_ns))
    commands += engine.process(observation(tuple(tight), start_ns + 120_000_000))
    # 松开
    commands += engine.process(observation(tuple(points), start_ns + 200_000_000))
    return commands


def test_second_pinch_returns_cursor_to_first_click_position() -> None:
    """两次捏合之间光标被带偏了，也要凑成双击——回到第一次的落点。

    用他 31002 帧使用日志还原的实测：相邻两次捏合**落点漂移中位 56px**，
    而系统双击容差只有 4px，所以 77% 的双击根本合并不了
    （"对一个应用捏合两次他都不会打开"）。
    """

    engine = _click_immediate_engine()
    base = pointer_points()

    first = _pinch_episode(engine, base, BASE)
    assert MouseCommandType.LEFT_CLICK in kinds(first), "第一次捏合应当是一次单击"
    anchor = (base[8].x, base[8].y)

    # 第二次捏合：整只手往右下方漂移（正是捏合动作会把指尖带下去）
    drifted = translate_points(base, dx=0.03, dy=0.03)
    second = _pinch_episode(engine, drifted, BASE + 400_000_000)

    assert MouseCommandType.LEFT_DOUBLE_CLICK in kinds(second), "第二次捏合没凑成双击"
    moves = [c for c in second if c.kind == MouseCommandType.MOVE]
    assert moves, "双击前必须先把光标送回第一次落点，否则系统不认双击"
    assert (moves[-1].x, moves[-1].y) == (anchor[0], anchor[1])


def test_two_pinches_far_apart_stay_two_single_clicks() -> None:
    """两次捏合落点相距太远 = 两个不同的目标，绝不能合并成双击。"""

    engine = _click_immediate_engine()
    base = pointer_points()
    _pinch_episode(engine, base, BASE)

    far = translate_points(base, dx=0.30, dy=0.0)
    second = _pinch_episode(engine, far, BASE + 300_000_000)

    assert MouseCommandType.LEFT_DOUBLE_CLICK not in kinds(second)
    assert MouseCommandType.LEFT_CLICK in kinds(second)


def test_two_pinches_beyond_window_stay_two_single_clicks() -> None:
    """间隔超过双击窗口就算两次单击（他中位间隔 422ms，窗口取 700ms）。"""

    engine = _click_immediate_engine()
    base = pointer_points()
    _pinch_episode(engine, base, BASE)

    second = _pinch_episode(engine, base, BASE + 1_500_000_000)

    assert MouseCommandType.LEFT_DOUBLE_CLICK not in kinds(second)
    assert MouseCommandType.LEFT_CLICK in kinds(second)


# ---------------------------------------------------------------- 光标锚点
#
# 2026-10-23：锚点从"食指尖"改成可选的"掌心 + 补偿偏移"。
# 离线实测（scripts/audit_pointer_anchor.py）：掌心的形状抖动只有食指尖的
# 17%~36%，捏合段最低 —— 这同时是"磁吸释放瞬间跳回随手位置"和
# "捏合 1.42→0.90 这段光标跟着食指走"两个老问题的根因。


def _anchor_track(config, frames):
    """直接驱动锚点 + 映射，绕过手势状态机，只看光标轨迹。"""

    engine = GestureEngine(config)
    xs = []
    for now, points in frames:
        anchor = engine._pointer_point(points, now)
        x, _y = engine._cursor(anchor, now)
        xs.append(x)
    return xs


def _steady_frames(count=60):
    base = pointer_points()
    return [(BASE + index * 33_000_000, base) for index in range(count)]


def _wobble_frames(count=30, amplitude=0.02):
    """掌心不动、只有食指指尖左右摆 —— 模拟捏合时食指自己动。"""

    base = pointer_points()
    frames = []
    for index in range(count):
        points = list(base)
        # 以第 8 号点（食指尖）为轴摆动：改它的 x，其余关键点不动。
        offset = amplitude if index % 2 else -amplitude
        points[8] = Landmark(base[8].x + offset, base[8].y, base[8].z)
        frame_time = BASE + index * 33_000_000
        frames.append((frame_time, tuple(points)))
    return frames


def test_pointer_anchor_ships_as_palm() -> None:
    """出厂默认是掌心锚点（2026-10-23 换的）。

    这不是"顺手改的手感"：两个老问题（磁吸释放跳、捏合时光标跟着食指走）
    都是"锚在食指尖"造成的，不锚移到掌心就修不掉。
    ``pointer_anchor="index_tip"`` 是完整回滚开关。
    """

    assert AppConfig().pointer_anchor == "palm"


def test_pointer_anchor_index_tip_returns_tip_verbatim() -> None:
    """回滚开关必须**逐位**返回食指尖，不带任何滤波/补偿。"""

    config = AppConfig.from_mapping({"pointer_anchor": "index_tip"})
    config.validate()
    engine = GestureEngine(config)
    points = pointer_points()

    assert engine._pointer_point(points, BASE) == points[8]


def test_pointer_anchor_rejects_unknown_value() -> None:
    """拼错的取值必须退回 ``index_tip``，不能悄悄进 palm 模式。"""

    config = AppConfig.from_mapping({"pointer_anchor": "palm_centre"})
    config.validate()

    assert config.pointer_anchor == "index_tip"


def test_palm_anchor_keeps_cursor_position_after_converging() -> None:
    """掌心锚点的**平均位置**必须和食指尖一致，否则屏幕边缘就够不到了。

    靠 ``掌心 + EMA(食指尖 - 掌心)`` 补偿：姿势不变时补偿量收敛到真实的
    指尖-掌心偏移，锚点就落回食指尖上。
    """

    frames = _steady_frames()

    tip_config = AppConfig()
    tip_config.pointer_anchor = "index_tip"
    palm_config = AppConfig()
    palm_config.pointer_anchor = "palm"

    tip_x = _anchor_track(tip_config, frames)[-1]
    palm_x = _anchor_track(palm_config, frames)[-1]

    assert abs(tip_x - palm_x) < 0.01, f"锚点换到掌心后光标位置偏了 {tip_x - palm_x:.3f}"


def test_palm_anchor_damps_index_tip_wobble() -> None:
    """食指自己抖（掌心不动）时，掌心锚点必须明显更稳。"""

    settle = _steady_frames(count=60)
    wobble = _wobble_frames()

    tip_config = AppConfig()
    tip_config.pointer_anchor = "index_tip"
    palm_config = AppConfig()
    palm_config.pointer_anchor = "palm"

    tip_x = _anchor_track(tip_config, settle + wobble)[len(settle):]
    palm_x = _anchor_track(palm_config, settle + wobble)[len(settle):]

    tip_span = max(tip_x) - min(tip_x)
    palm_span = max(palm_x) - min(palm_x)

    assert tip_span > 0.01, "测试本身没造出抖动，先修测试"
    assert palm_span < tip_span * 0.5, (
        f"掌心锚点没压住抖动：食指 {tip_span:.4f} vs 掌心 {palm_span:.4f}"
    )



# ---------------- 宽高比自适应（换摄像头不用重标）----------------
#
# 用户原话：「我的摄像头的硬件有限，像 720p 的，1080p 的就使用不了我们这个，
# 使用我们还要再做一个适配他们版本」。
#
# 根因是归一化坐标的**各向异性**：x 除以画面宽、y 除以画面高。掌宽（5→17）
# 是纯横向的，捏合比（4→8）是斜向的，16:9 画面里横向长度比 4:3 缩水 25%，
# 两者受影响幅度不同 → 比值整体漂移，所有捏合阈值失准。
# 补偿方式是把 x 乘上 ``宽高比 ÷ (4/3)``，4:3 时倍率恰为 1.0（旧行为零变化）。


def _squeeze_x(points: tuple[Landmark, ...], factor: float) -> tuple[Landmark, ...]:
    """模拟 16:9 画面里横向归一化坐标被多压一次。"""

    return tuple(Landmark(point.x * factor, point.y, point.z) for point in points)


def _pinch_ratio(points: tuple[Landmark, ...]) -> float:
    return _distance_2d(points[4], points[8]) / _palm_width(points)


def test_aspect_scale_is_exactly_one_at_reference_aspect() -> None:
    """4:3 是标定基准，倍率必须恰好 1.0，否则全部历史阈值当场失准。"""

    assert _aspect_scale(4.0 / 3.0) == 1.0
    assert _aspect_scale(16.0 / 9.0) == pytest.approx(4.0 / 3.0)


def test_aspect_scale_clamps_garbage_input() -> None:
    """坏数据（0/负数/NaN）宁可不动，也不要把光标搞飞。"""

    assert _aspect_scale(0.0) == 1.0
    assert _aspect_scale(-1.5) == 1.0
    assert _aspect_scale(float("nan")) == 1.0
    assert _isotropic(pointer_points(), 1.0) is not None

    clamped = _aspect_scale(100.0)
    assert 0.5 <= clamped <= 2.4


def test_isotropic_at_unit_scale_returns_the_same_tuple() -> None:
    """k=1 时必须原样返回，不加分配——这是"4:3 零回归"的结构性保证。"""

    points = pointer_points()
    assert _isotropic(points, 1.0) is points


@pytest.mark.parametrize(
    "points_factory",
    [entry_palm_points, right_pinch_from_open_points, side_palm_points],
)
def test_pinch_ratio_drifts_badly_without_compensation(points_factory) -> None:
    """先证明问题真实存在：16:9 下不补偿会漂 18%~32%。"""

    base = points_factory()
    reference = _pinch_ratio(base)
    assert reference > 1e-3, "测试素材本身没有捏合距离，先修测试"

    squeezed = _squeeze_x(base, 0.75)
    drifted = _pinch_ratio(squeezed)

    assert drifted > reference * 1.15, (
        f"16:9 未补偿应明显偏大：4:3 {reference:.3f} vs 裸算 {drifted:.3f}"
    )


@pytest.mark.parametrize(
    "points_factory",
    [entry_palm_points, right_pinch_from_open_points, side_palm_points],
)
def test_pinch_ratio_is_invariant_after_compensation(points_factory) -> None:
    """补偿后 4:3 与 16:9 量出来必须是同一个数（阈值不用重标）。"""

    base = points_factory()
    reference = _pinch_ratio(base)

    squeezed = _squeeze_x(base, 0.75)
    compensated = _pinch_ratio(_isotropic(squeezed, _aspect_scale(16.0 / 9.0)))

    assert compensated == pytest.approx(reference, rel=1e-9)


def test_engine_picks_up_aspect_from_observation() -> None:
    engine = GestureEngine(AppConfig(entry_gate_enabled=True))
    _enter_gate(engine)

    engine.process(observation(pointer_points(), BASE + 400_000_000))
    assert engine._aspect_k == 1.0

    engine.process(
        observation(pointer_points(), BASE + 500_000_000, aspect=16.0 / 9.0),
    )
    assert engine._aspect_k == pytest.approx(4.0 / 3.0)


def test_pinch_ratio_matches_between_4_3_and_16_9_sessions() -> None:
    """端到端：同一姿势在两种宽高比的摄像头上量出同一个捏合比。"""

    def ratio_in(aspect: float, points: tuple[Landmark, ...]) -> float:
        engine = GestureEngine(AppConfig(entry_gate_enabled=True))
        _enter_gate(engine)
        engine.process(observation(points, BASE + 400_000_000, aspect=aspect))
        return engine._last_pinch_ratio

    base = right_pinch_from_open_points()
    reference = ratio_in(4.0 / 3.0, base)
    wide = ratio_in(16.0 / 9.0, _squeeze_x(base, 0.75))

    assert reference > 1e-3
    assert wide == pytest.approx(reference, rel=0.02)


def test_five_open_survives_16_9_squeeze() -> None:
    """五指张开进入门同样走比值判据，16:9 下也不能被压死。"""

    base = entry_palm_points()
    assert _is_five_open(base) is True

    squeezed = _squeeze_x(base, 0.75)
    compensated = _isotropic(squeezed, _aspect_scale(16.0 / 9.0))
    assert _is_five_open(compensated) is True
