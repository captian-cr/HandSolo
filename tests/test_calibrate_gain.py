"""标定工具里「捏合趋向锚定 + 完成判定」的行为约定。

为什么要单独测：这段逻辑是 v2 标定能用的前提。
v2 初版曾把释放条件写成 ``ratio >= pinch_release(0.55)``，
而锚定阈值是 ``0.70`` —— 0.55 < 0.70，于是"一锚定就立刻释放"，
锚定形同失效，用户表现为「捏合没反应」。这里把它钉住。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.calibrate_gain import PinchTracker  # noqa: E402

ANCHOR, ENGAGE, RELEASE = 0.70, 0.42, 0.55


def _tracker() -> PinchTracker:
    return PinchTracker(ANCHOR, ENGAGE, RELEASE)


def test_anchor_starts_when_ratio_approaches() -> None:
    tracker = _tracker()

    assert tracker.update(1.50, "tip", (0.5, 0.5)) is None
    assert tracker.anchored is False

    assert tracker.update(0.68, "tip@068", (0.4, 0.6)) == "start"
    assert tracker.anchored is True
    assert tracker.tip == "tip@068"
    assert tracker.cursor == (0.4, 0.6)


def test_anchor_survives_between_anchor_and_engage() -> None:
    """锚定后捏合比落在 0.42~0.70 之间时，锚定必须保持住。"""

    tracker = _tracker()
    tracker.update(0.68, "tip", (0.4, 0.6))

    # 0.50 同时满足 "<= 0.70" 和 ">= 0.55"，旧实现会在这里立刻释放。
    assert tracker.update(0.50, "tip2", (0.9, 0.9)) is None
    assert tracker.anchored is True, "锚定不该在 0.50 就解除"
    assert tracker.tip == "tip", "锚定期间记录点不能被后续帧覆盖"


def test_complete_requires_real_pinch_then_release() -> None:
    tracker = _tracker()
    tracker.update(0.68, "tip", (0.4, 0.6))

    assert tracker.update(0.40, "x", (0, 0)) is None   # 捏到 engage 以下
    assert tracker.pinched is True
    assert tracker.update(0.30, "x", (0, 0)) is None
    assert tracker.update(0.60, "x", (0, 0)) == "complete"
    assert tracker.anchored is False
    assert tracker.pinched is False


def test_release_without_pinching_does_not_complete() -> None:
    """只逼近没捏下去，不算点击。"""

    tracker = _tracker()
    tracker.update(0.68, "tip", (0.4, 0.6))

    # 捏合比在 0.42~0.55 之间来回，从未真正跌破 engage
    assert tracker.update(0.50, "x", (0, 0)) is None
    assert tracker.pinched is False
    assert tracker.anchored is True


def test_retreat_cancels_anchor() -> None:
    """逼近后又退回 0.70 以上 = 放弃，锚定取消且记录点清空。"""

    tracker = _tracker()
    tracker.update(0.68, "tip", (0.4, 0.6))

    assert tracker.update(0.80, "tip2", (0.9, 0.9)) == "cancel"
    assert tracker.anchored is False
    assert tracker.tip is None


def test_reset_clears_everything() -> None:
    tracker = _tracker()
    tracker.update(0.68, "tip", (0.4, 0.6))
    tracker.update(0.30, "x", (0, 0))

    tracker.reset()

    assert tracker.anchored is False
    assert tracker.pinched is False
    assert tracker.tip is None
    assert tracker.cursor is None
