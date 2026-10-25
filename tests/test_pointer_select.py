"""``pointer_select`` 的平滑与冻结行为测试。

重点：捏合时食指向拇指靠会带动锚点，光标必须冻结，否则点击落点会滑走。
"""

from __future__ import annotations

import copy
import threading
import time

from handmouse.grid_judge import PinchJudger
from handmouse.grid_select import DEFAULT_CONFIG, HandSample
from handmouse.grid_ui import PointerState
from handmouse.pointer_select import PointerRunner


class FakePoller:
    def __init__(self) -> None:
        self.sample: HandSample | None = None

    def snapshot(self) -> tuple[HandSample | None, float, str]:
        return self.sample, 30.0, ""

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


def make_runner() -> tuple[PointerRunner, FakePoller]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["roi"] = [0.0, 1.0, 0.0, 1.0]
    config["sensitivity"] = 1.0
    # 收紧时间常数，让测试能在几次 tick 内收敛。
    config["pointer_smoothing_s"] = 0.01
    # 收紧双击窗口，避免测试等半秒以上。
    config["double_click_window_ms"] = 300
    config["snap_enabled"] = False
    runner = PointerRunner(
        config=config,
        judge=PinchJudger(enter_ratio=0.40, release_ratio=0.62, hold_ms=100.0),
        state=PointerState(),
        stop_event=threading.Event(),
        screen_w=1000,
        screen_h=1000,
        dry_run=True,
    )
    poller = FakePoller()
    runner.poller = poller  # type: ignore[assignment]
    return runner, poller


def sample_at(x: float, y: float, ratio: float) -> HandSample:
    return HandSample(
        index_tip=(x, y),
        thumb_tip=(x, y),
        index_root=(x, y - 0.2),
        ratio=ratio,
        captured_ms=0.0,
    )


def feed(runner: PointerRunner, poller: FakePoller, x: float, y: float,
         ratio: float, frames: int = 8, gap: float = 0.02) -> None:
    for _ in range(frames):
        poller.sample = sample_at(x, y, ratio)
        runner.tick()
        time.sleep(gap)


def test_smoothing_converges_towards_target() -> None:
    runner, poller = make_runner()

    feed(runner, poller, 0.85, 0.80, 1.60, frames=12)

    assert runner.state.x > 0.70
    assert runner.state.y > 0.65


def test_cursor_freezes_while_pinching() -> None:
    runner, poller = make_runner()
    feed(runner, poller, 0.40, 0.40, 1.60, frames=12)
    settled_x = runner.state.x
    assert settled_x > 0.30

    # 捏合，同时手指往拇指方向滑（真实捏合就是这样）。
    poller.sample = sample_at(0.40, 0.55, 0.28)
    runner.tick()
    assert runner.state.pinching is True
    frozen_x = runner.state.x

    # 继续在别处保持捏合：坐标不得再被拉走。
    for _ in range(6):
        poller.sample = sample_at(0.75, 0.90, 0.28)
        runner.tick()
        time.sleep(0.02)
        assert runner.state.x == frozen_x

    assert frozen_x == settled_x


def test_resume_follows_again_after_release() -> None:
    runner, poller = make_runner()
    feed(runner, poller, 0.30, 0.30, 1.60, frames=10)

    poller.sample = sample_at(0.30, 0.45, 0.28)
    runner.tick()
    assert runner.judge.is_closed is True

    # 松开并移到右下：应当重新开始跟随。
    feed(runner, poller, 0.90, 0.85, 1.60, frames=12)

    assert runner.judge.is_closed is False
    assert runner.state.x > 0.70


def test_paused_does_not_move() -> None:
    runner, poller = make_runner()
    feed(runner, poller, 0.30, 0.30, 1.60, frames=10)
    before = runner.state.x

    runner.state.paused = True
    feed(runner, poller, 0.90, 0.90, 1.60, frames=8)

    assert runner.state.x == before


def test_single_click_fires_after_window() -> None:
    runner, poller = make_runner()

    poller.sample = sample_at(0.5, 0.5, 0.28)
    runner.tick()
    time.sleep(0.12)
    poller.sample = sample_at(0.5, 0.5, 0.28)
    runner.tick()

    # 第一次确认只进入待定，不立即点击——否则手势双击永远凑不出来。
    assert runner._clicks == 0
    assert runner.state.pending is True

    time.sleep(0.35)
    poller.sample = sample_at(0.5, 0.5, 1.60)
    runner.tick()

    assert runner._clicks == 1
    assert runner.state.pending is False


def test_double_pinch_sends_double_click() -> None:
    runner, poller = make_runner()

    # 第一次捏合并触发确认
    poller.sample = sample_at(0.5, 0.5, 0.28)
    runner.tick()
    time.sleep(0.12)
    poller.sample = sample_at(0.5, 0.5, 0.28)
    runner.tick()
    assert runner._clicks == 0

    # 松开，再捏一次（保持在双击窗口内）
    poller.sample = sample_at(0.5, 0.5, 1.60)
    runner.tick()
    poller.sample = sample_at(0.5, 0.5, 0.28)
    runner.tick()
    time.sleep(0.12)
    poller.sample = sample_at(0.5, 0.5, 0.28)
    runner.tick()

    assert runner._clicks == 1
    assert runner.state.message == "双击"


def test_hand_lost_stops_following() -> None:
    runner, poller = make_runner()
    feed(runner, poller, 0.60, 0.60, 1.60, frames=8)
    held = runner.state.x

    poller.sample = None
    runner.tick()

    assert runner.state.hand_visible is False
    assert runner.state.x == held


# ---- 磁吸 ----


def make_target(left: int, top: int, right: int, bottom: int):
    from handmouse.uia_targets import UITarget

    return UITarget(
        name="目标", control_type="ButtonControl",
        left=left, top=top, right=right, bottom=bottom,
    )


def enable_snap(runner: PointerRunner) -> None:
    runner.config["snap_enabled"] = True
    runner.config["pointer_snap_px"] = 30
    runner.config["pointer_snap_release_px"] = 60
    runner._targets = [make_target(500, 500, 540, 540)]  # 中心 (520, 520)


def test_snap_pulls_cursor_to_center() -> None:
    runner, _ = make_runner()
    enable_snap(runner)

    assert runner._apply_snap(525, 522) == (520, 520)
    assert runner._snapped is not None


def test_snap_ignores_far_target() -> None:
    runner, _ = make_runner()
    enable_snap(runner)

    assert runner._apply_snap(700, 700) == (700, 700)
    assert runner._snapped is None


def test_snap_hysteresis_keeps_hold_until_release_radius() -> None:
    runner, _ = make_runner()
    enable_snap(runner)

    runner._apply_snap(520, 520)
    assert runner._snapped is not None

    # 距离 40：超出进入半径 30，但在脱离半径 60 之内 -> 仍吸住。
    assert runner._apply_snap(560, 520) == (520, 520)
    assert runner._snapped is not None

    # 距离 100：超过脱离半径 -> 松手。
    assert runner._apply_snap(620, 520) == (620, 520)
    assert runner._snapped is None


def test_snap_disabled_passes_through() -> None:
    runner, _ = make_runner()
    runner.config["snap_enabled"] = False
    runner._targets = [make_target(500, 500, 540, 540)]

    assert runner._apply_snap(525, 522) == (525, 522)
    assert runner._snapped is None
