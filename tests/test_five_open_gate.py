"""五指张开的**两道闸**：投影长度闸 + 时序确认闸。

2026-10-22 用户反馈：只伸食指，HUD 却显示「五指张开（待机，左右挥动翻页）」，
光标不动。根因是 ``_is_five_open`` 当年只用二维夹角判断手指伸没伸直，而他
侧手投影下收拢的四指三个关节点近似共线、夹角照样 ≥135°。

他自己 5700 帧真实样本上的假阳率（``scripts/audit_gesture_conflicts.py``）：
    move（只伸食指）  9.8%   → 光标冻结
    stop（食指+拇指张开）40.3%   → 想做"停住"反被抢进待机
    swipe（双指）     13.7%   → 划到一半被判待机

修完后：move 4.2% / stop 0% / swipe 3.8% / open 仍识别 96.8%。

本文件锁死修复后的契约，用的是**他自己的真实帧**（不是合成手），防止将来
"调阈值"把这两个闸又调没了。
"""

from __future__ import annotations

import pytest
from conftest import observation, open_palm_points, pointer_points

from handmouse.config import AppConfig
from handmouse.contracts import Landmark, MouseCommandType
from handmouse.gestures import GestureEngine, _finger_extensions, _is_five_open

BASE = 1_000_000_000

#: 他在「食指+拇指张开」(stop) 姿势下的一帧真实关键点，40 维归一化特征
#: （相对手腕平移、除以手长）。四指 reach = 1.74 / 1.03 / 0.93 / **0.80**
#: —— 角度全伸，长度只有小指暴露真相。是长度闸的边界用例（0.80 离门槛
#: 1.10 还有余量，但比它更接近的帧在 stop 里已经没有了）。
_STOP_FRAME = [
    -0.182229, -0.174702, -0.275016, -0.443733, -0.190334, -0.545578,
    -0.098268, -0.593372, -0.124248, -1.039828, -0.054162, -1.475807,
    0.010305, -1.754934, 0.072761, -2.000560, 0.058582, -0.998283,
    0.122303, -0.844292, 0.028104, -0.531771, -0.038349, -0.426538,
    0.210880, -0.872570, 0.226077, -0.664523, 0.111962, -0.418826,
    0.038436, -0.378082, 0.337716, -0.715800, 0.334634, -0.553202,
    0.217199, -0.354395, 0.135029, -0.313233,
]


def _points_from_features(features: list[float]) -> tuple[Landmark, ...]:
    """40 维归一化特征 → 21 点（0 号点即原点；判据只用 x/y，尺度不影响）。"""

    assert len(features) == 40
    points = [Landmark(x=0.0, y=0.0, z=0.0)]
    points += [
        Landmark(x=features[2 * i], y=features[2 * i + 1], z=0.0) for i in range(20)
    ]
    return tuple(points)


def _engine() -> GestureEngine:
    config = AppConfig()
    # 进入门会先拦一道，这里只想测待机判定本身。
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    return engine


def _kinds(commands) -> list[MouseCommandType]:
    return [command.kind for command in commands]


# ---- 闸一：投影长度 ----


def test_real_stop_frame_is_not_five_open() -> None:
    """真实"食指+拇指张开"帧：角度全伸，但投影长度不够 → 不许判成待机。

    这就是 40.3% 假阳的来源：三指收拢但关节共线，角度判据看不出来。
    """

    points = _points_from_features(_STOP_FRAME)
    # 前提：纯角度判据确实把四指全判成伸直（当年 bug 的现场）。
    assert all(_finger_extensions(points))
    # 修复：长度闸把它挡下来。
    assert _is_five_open(points) is False


def test_true_open_palm_still_detected() -> None:
    """加了长度闸之后，真正的五指张开必须照旧识别（不能修过头）。"""

    assert all(_finger_extensions(open_palm_points()))
    assert _is_five_open(open_palm_points()) is True


def test_five_open_prefers_3d_when_world_landmarks_present() -> None:
    """有 3D 时五指张开只听 3D：召回更高，且双指帧不再被判成待机。

    2026-10-22 在他 701 帧带 3D 的采集上实测（`scripts/audit_2d_vs_3d_real.py`）：

    ====================  ========  ============
                         五指张开   双指被误判
    2D 全判据              82.2%          7.0%
    3D≥0.86 + 拇指闸       92.5%          0.0%
    ====================  ========  ============

    后者正是他"划着划着就跳去待机、光标冻住"的来源之一。
    """

    from conftest import world_hand

    open_hand = world_hand(
        index_extended=True,
        middle_extended=True,
        ring_extended=True,
        pinky_extended=True,
    )
    two_finger = world_hand()  # 无名指/小指蜷着

    # 3D 说四指全伸 + 拇指闸成立 → 五指张开。
    assert _is_five_open(open_palm_points(), open_hand) is True
    # 3D 说无名指/小指蜷着 → 绝不是五指张开（2D 版本在这里会误判）。
    assert _is_five_open(open_palm_points(), two_finger) is False


def test_five_open_falls_back_to_2d_without_world_landmarks() -> None:
    """3D 缺失时必须退回 2D，不能因为"没有 3D"就永远判不成五指张开。"""

    from conftest import world_hand

    assert _is_five_open(open_palm_points()) is True
    assert _is_five_open(open_palm_points(), ()) is True
    assert _is_five_open(open_palm_points(), world_hand()[:20]) is True


# ---- 闸二：时序确认（进严出松）----


def test_single_five_open_frame_does_not_freeze_cursor() -> None:
    """单帧误报不许切待机，也不许按移动处理（悬置）。

    按移动处理会在张开姿势下把光标拽到食指尖（"飞一下再冻住"）；
    直接待机就是老毛病（单帧误报 → 光标瞬冻）。
    """

    engine = _engine()
    engine.process(observation(pointer_points(), BASE))
    assert engine.current_gesture.value == "pointer"

    commands = engine.process(observation(open_palm_points(), BASE + 66_000_000))

    assert engine.current_gesture.value != "open_palm"
    assert commands == [], "确认窗口内不该发 MOVE"


def test_five_open_confirmed_after_hold_window() -> None:
    """连续保持满 300ms（他实机 15fps 下约 5 帧）才切待机。"""

    engine = _engine()
    engine.process(observation(open_palm_points(), BASE))
    assert engine.current_gesture.value != "open_palm"

    engine.process(observation(open_palm_points(), BASE + 310_000_000))
    assert engine.current_gesture.value == "open_palm"


def test_open_palm_stays_while_pose_held() -> None:
    """停留在待机姿势时不能被重新计时踢出去。

    如果确认计时在"已在待机"时也重来一遍，用户张开手停着会每隔 300ms
    掉出待机一帧、光标突然暴走——比原 bug 更难查。
    """

    engine = _engine()
    engine.process(observation(open_palm_points(), BASE))
    engine.process(observation(open_palm_points(), BASE + 310_000_000))
    assert engine.current_gesture.value == "open_palm"

    # 他的实机 15fps：连续 30 帧 ≈ 2 秒。
    for step in range(1, 31):
        engine.process(observation(open_palm_points(), BASE + 310_000_000 + step * 66_000_000))
        assert engine.current_gesture.value == "open_palm", f"第 {step} 帧掉出待机"


def test_leaving_open_palm_is_immediate() -> None:
    """出松：张开的手一变回食指，下一帧就恢复移动，不加任何延迟。"""

    engine = _engine()
    engine.process(observation(open_palm_points(), BASE))
    engine.process(observation(open_palm_points(), BASE + 310_000_000))
    assert engine.current_gesture.value == "open_palm"

    commands = engine.process(observation(pointer_points(), BASE + 376_000_000))

    assert engine.current_gesture.value == "pointer"
    assert MouseCommandType.MOVE in _kinds(commands)


def test_lost_clears_five_open_timer() -> None:
    """手离画必须清零确认计时。

    否则手回来第一帧会拿"出画前"的起始时刻算出已保持很久，直接把光标
    冻进待机——用户看到的是"手一回到画面鼠标就不动了"。
    """

    engine = _engine()
    engine.process(observation(open_palm_points(), BASE))
    assert engine._five_open_since_ns == BASE

    engine.tick(BASE + 2_000_000_000)  # 远超 hand_lost_ms

    assert engine._five_open_since_ns is None


# ---- 五指张开唤醒（2026-10-23 用户要求「开机比五根手指就能启动」）----


def test_open_palm_wakes_from_pause_after_hold() -> None:
    """暂停时比五指张开并保持一段时间 → 恢复控制。"""

    config = AppConfig()
    config.entry_gate_enabled = False
    config.wake_by_open_palm = True
    config.wake_hold_ms = 400
    engine = GestureEngine(config)
    engine.set_paused(True)

    engine.process(observation(open_palm_points(), BASE))
    assert engine.paused is True, "刚比出来就唤醒 = 太灵敏"

    engine.process(observation(open_palm_points(), BASE + 200_000_000))
    assert engine.paused is True, "还没保持够时长就唤醒了"

    engine.process(observation(open_palm_points(), BASE + 500_000_000))
    assert engine.paused is False, "保持够了却没唤醒"


def test_brief_open_palm_does_not_wake() -> None:
    """手从画面前掠过（张开一帧）不能把控制唤醒。"""

    config = AppConfig()
    config.entry_gate_enabled = False
    config.wake_by_open_palm = True
    config.wake_hold_ms = 600
    engine = GestureEngine(config)
    engine.set_paused(True)

    engine.process(observation(open_palm_points(), BASE))
    engine.process(observation(pointer_points(), BASE + 100_000_000))

    assert engine.paused is True


def test_wake_disabled_by_default_config_flag() -> None:
    """显式关掉开关后，五指张开不能再唤醒（保留逃生口）。"""

    config = AppConfig()
    config.entry_gate_enabled = False
    config.wake_by_open_palm = False
    config.wake_hold_ms = 200
    engine = GestureEngine(config)
    engine.set_paused(True)

    for step in range(1, 6):
        engine.process(observation(open_palm_points(), BASE + step * 200_000_000))

    assert engine.paused is True
