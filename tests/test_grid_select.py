"""``handmouse.grid_select`` 的目标锁定行为测试。

重点验证：捏合过程中食指往拇指方向移动（这是捏合的物理必然），不得把
已锁定的目标改判成相邻图标。
"""

from __future__ import annotations

import copy
import threading
import time

from handmouse.grid_select import DEFAULT_CONFIG, HandSample, build_runner


class FakePoller:
    """替换真实的摄像头+推理线程，只回放预设样本。"""

    def __init__(self) -> None:
        self.sample: HandSample | None = None

    def snapshot(self) -> tuple[HandSample | None, float, str]:
        return self.sample, 30.0, ""

    def start(self) -> None:
        return None

    def join(self, timeout: float | None = None) -> None:
        return None


def make_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    # 归一化映射，让样本坐标可以直接当作屏幕坐标使用。
    config["roi"] = [0.0, 1.0, 0.0, 1.0]
    config["sensitivity"] = 1.0
    config["items"] = [
        {"label": "A", "command": ""},
        {"label": "B", "command": ""},
        {"label": "C", "command": ""},
        {"label": "D", "command": ""},
    ]
    return config


def make_runner() -> tuple[object, FakePoller]:
    runner = build_runner(make_config(), threading.Event())
    runner.dry_run = True
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


def open_hand_at(runner: object, index: int, frames: int = 4) -> None:
    """在指定图标中心连续喂"张开"样本，直到目标稳定。"""

    cx, cy = runner.layout.center(index)  # type: ignore[attr-defined]
    for _ in range(frames):
        runner.poller.sample = sample_at(cx, cy, 1.60)  # type: ignore[attr-defined]
        runner.tick()  # type: ignore[attr-defined]


def test_target_locks_when_pinch_starts() -> None:
    runner, poller = make_runner()
    open_hand_at(runner, 0)
    assert runner.state.target == 0

    # 捏合的真实位移：食指向拇指靠拢，指尖大约移动 0.07~0.10（食指长度的一半）。
    cx, cy = runner.layout.center(0)
    poller.sample = sample_at(cx, cy + 0.09, 0.30)
    runner.tick()

    assert runner.state.target == 0
    assert runner.judge.is_closed is True


def test_lock_holds_across_pinch_frames() -> None:
    runner, poller = make_runner()
    open_hand_at(runner, 0)

    cx, cy = runner.layout.center(0)
    for _ in range(6):
        poller.sample = sample_at(cx, cy + 0.09, 0.30)
        runner.tick()
        assert runner.state.target == 0


def test_pinch_fires_on_locked_target() -> None:
    runner, poller = make_runner()
    open_hand_at(runner, 0)

    cx, cy = runner.layout.center(0)
    poller.sample = sample_at(cx, cy + 0.09, 0.30)
    runner.tick()
    assert runner.state.target == 0

    # 越过 pinch_hold_ms 后触发，命中的应是锁定的 A。
    time.sleep(0.14)
    poller.sample = sample_at(cx, cy + 0.09, 0.30)
    runner.tick()

    assert "A" in runner.state.message


def test_far_drift_cancels_the_pinch() -> None:
    """手整体移开很远时放弃本次捏合，而不是点原来的格子。"""

    runner, poller = make_runner()
    open_hand_at(runner, 0)

    poller.sample = sample_at(*runner.layout.center(0), 0.30)
    runner.tick()
    assert runner.state.target == 0

    # 对角线另一端：远超 2 倍磁吸半径。
    poller.sample = sample_at(0.95, 0.95, 0.30)
    runner.tick()

    assert runner.state.target is None


def test_unlock_after_release() -> None:
    runner, poller = make_runner()
    open_hand_at(runner, 0)

    poller.sample = sample_at(*runner.layout.center(0), 0.30)
    runner.tick()
    assert runner.judge.is_closed is True

    # 松开：回到张开，并移到图标 3，此时应恢复实时跟随。
    open_hand_at(runner, 3)
    assert runner.judge.is_closed is False
    assert runner.state.target == 3


def test_message_marks_locked_state() -> None:
    runner, poller = make_runner()
    open_hand_at(runner, 0)

    poller.sample = sample_at(*runner.layout.center(0), 0.30)
    runner.tick()

    assert "已锁定" in runner.state.message
