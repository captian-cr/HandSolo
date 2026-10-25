"""``handmouse.grid_judge`` 的捏合判定与目标去抖测试。"""

from __future__ import annotations

import pytest

from handmouse.grid_judge import PinchJudger, TargetStabilizer, pinch_ratio


# ---- pinch_ratio ----


def test_pinch_ratio_divides_by_index_length() -> None:
    ratio = pinch_ratio(
        thumb_tip=(0.0, 0.0),
        index_tip=(0.1, 0.0),
        index_root=(0.5, 0.0),
    )

    # 食指长度 0.4（0.5→0.1），捏合距离 0.1 ⇒ 0.25。
    assert ratio == pytest.approx(0.25)


def test_pinch_ratio_rejects_degenerate_index() -> None:
    """食指长度趋近 0（完全侧对镜头）时报 inf，而不是给出虚假的小比值。"""

    ratio = pinch_ratio(
        thumb_tip=(0.0, 0.0),
        index_tip=(0.3, 0.3),
        index_root=(0.3, 0.3),
    )

    assert ratio == float("inf")


# ---- PinchJudger ----


def make_judge(**kwargs: float) -> PinchJudger:
    params = {"enter_ratio": 0.40, "release_ratio": 0.62, "hold_ms": 100.0}
    params.update(kwargs)
    return PinchJudger(**params)  # type: ignore[arg-type]


def test_holds_before_firing() -> None:
    judge = make_judge()

    assert judge.update(0.30, 0.0) is False
    assert judge.is_closed is True
    assert judge.update(0.30, 50.0) is False


def test_fires_once_after_hold() -> None:
    judge = make_judge()

    judge.update(0.30, 0.0)
    assert judge.update(0.30, 100.0) is True
    # 捏住不放不得连续触发。
    assert judge.update(0.30, 200.0) is False
    assert judge.update(0.30, 500.0) is False


def test_release_then_pinch_fires_again() -> None:
    judge = make_judge()

    judge.update(0.30, 0.0)
    assert judge.update(0.30, 100.0) is True
    assert judge.update(0.70, 200.0) is False
    assert judge.is_closed is False
    judge.update(0.30, 300.0)
    assert judge.update(0.30, 400.0) is True


def test_hysteresis_band_keeps_state() -> None:
    judge = make_judge()

    judge.update(0.30, 0.0)
    # 落在 enter/release 之间的滞回带：既不算松开，也不重新开始计时。
    assert judge.update(0.50, 50.0) is False
    assert judge.is_closed is True
    assert judge.update(0.50, 150.0) is True


def test_brief_pinch_below_hold_never_fires() -> None:
    judge = make_judge(hold_ms=200.0)

    judge.update(0.30, 0.0)
    assert judge.update(0.30, 100.0) is False
    assert judge.update(0.70, 150.0) is False
    assert judge.update(0.70, 300.0) is False


def test_hand_lost_resets_state() -> None:
    judge = make_judge()

    judge.update(0.30, 0.0)
    assert judge.update(float("inf"), 60.0) is False
    assert judge.is_closed is False
    # 复位后必须重新计时，不能凭旧的 closed_since 立刻触发。
    judge.update(0.30, 70.0)
    assert judge.update(0.30, 120.0) is False
    assert judge.update(0.30, 180.0) is True


def test_reset_clears_pending() -> None:
    judge = make_judge()

    judge.update(0.30, 0.0)
    judge.reset()

    assert judge.is_closed is False
    assert judge.update(0.30, 100.0) is False


def test_invalid_thresholds_rejected() -> None:
    with pytest.raises(ValueError):
        PinchJudger(enter_ratio=0.6, release_ratio=0.4)
    with pytest.raises(ValueError):
        PinchJudger(enter_ratio=0.0)


# ---- TargetStabilizer ----


def test_stabilizer_requires_consecutive_frames() -> None:
    stabilizer = TargetStabilizer(confirm_frames=3)

    assert stabilizer.update(0) is None
    assert stabilizer.update(0) is None
    assert stabilizer.update(0) == 0


def test_stabilizer_ignores_single_frame_flicker() -> None:
    stabilizer = TargetStabilizer(confirm_frames=3)

    for _ in range(3):
        stabilizer.update(0)
    # 单帧跳到 1 又跳回 0：稳定值应保持 0，不产生闪烁。
    assert stabilizer.update(1) == 0
    assert stabilizer.update(0) == 0


def test_stabilizer_switches_after_enough_frames() -> None:
    stabilizer = TargetStabilizer(confirm_frames=2)

    stabilizer.update(0)
    stabilizer.update(0)
    assert stabilizer.update(1) == 0
    assert stabilizer.update(1) == 1


def test_stabilizer_none_also_needs_frames() -> None:
    stabilizer = TargetStabilizer(confirm_frames=2)

    stabilizer.update(0)
    stabilizer.update(0)
    # 进入安全区同样要跨帧确认，避免手刚滑出图标就被判成无目标。
    assert stabilizer.update(None) == 0
    assert stabilizer.update(None) is None


def test_stabilizer_reset() -> None:
    stabilizer = TargetStabilizer(confirm_frames=2)

    stabilizer.update(0)
    stabilizer.update(0)
    stabilizer.reset()

    assert stabilizer.stable is None
    assert stabilizer.update(0) is None


def test_stabilizer_rejects_zero_frames() -> None:
    with pytest.raises(ValueError):
        TargetStabilizer(confirm_frames=0)
