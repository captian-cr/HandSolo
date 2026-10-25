"""渲染去重的行为约定。

背景（2026-10-20 实测）：主循环原本靠 ``cv2.waitKey(1)`` 自然节流到约
113 次/秒，而摄像头只有 30fps——约 3/4 的 ``build_view`` + ``imshow``
是在把同一帧重复画一遍。这既白烧 CPU（build_view 单次 2.4~2.66ms），
又持续迫使Windows DWM 重绘窗口、把独显从 P8 拉到 P0。

修复方式是按 ``FramePacket.sequence`` 判新帧：无新帧时跳过渲染，
但**仍要走 waitKey 喂按键**，否则 M/R/ESC/Q 会失灵。
"""

import numpy as np
import pytest

from handmouse import debug_view


class _FakeCv2:
    """记录 imshow / waitKey 调用的最小替身。"""

    INTER_LINEAR = 1

    def __init__(self) -> None:
        self.imshow_calls: list[str] = []
        self.waitkey_calls: list[int] = []
        self.destroyed: list[str] = []

    def imshow(self, name, frame):  # noqa: ANN001
        self.imshow_calls.append(name)

    def waitKey(self, delay):  # noqa: ANN001
        self.waitkey_calls.append(delay)
        return 255  # 模拟"没有按键"

    def destroyWindow(self, name):  # noqa: ANN001
        self.destroyed.append(name)


@pytest.fixture
def fake_cv2(monkeypatch):
    fake = _FakeCv2()
    monkeypatch.setattr(debug_view, "cv2", fake)
    monkeypatch.setattr(debug_view, "_window_opened", False)
    return fake


def _frame() -> np.ndarray:
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_none_frame_without_open_window_only_polls_keys(fake_cv2) -> None:
    """窗口还没建起来时不能调 imshow（会抛异常），但也不该崩。"""

    key = debug_view.present("win", None, 1.0)

    assert key == -1
    assert fake_cv2.imshow_calls == []
    assert fake_cv2.waitkey_calls == []


def test_frame_none_still_polls_keys_when_window_exists(fake_cv2) -> None:
    """这是限速后按键不失灵的关键：跳帧也必须继续读键盘。"""

    debug_view.present("win", _frame(), 1.0)  # 建窗
    assert fake_cv2.imshow_calls == ["win"]

    key = debug_view.present("win", None, 1.0)

    assert fake_cv2.waitkey_calls == [1, 1]  # 正常帧 + 跳帧，各喂一次
    assert fake_cv2.imshow_calls == ["win"]  # 没多画一帧
    assert key == 255


def test_new_frame_always_redraws(fake_cv2) -> None:
    """有新帧时必须重画，不能被去重逻辑误伤。"""

    for _ in range(3):
        debug_view.present("win", _frame(), 1.0)

    assert fake_cv2.imshow_calls == ["win"] * 3
    assert fake_cv2.waitkey_calls == [1, 1, 1]


def test_scale_above_one_resizes_before_show(fake_cv2, monkeypatch) -> None:
    """DPI 放大路径仍然按原逻辑 resize，不受 frame=None 分支影响。"""

    resized: list[tuple] = []
    monkeypatch.setattr(
        debug_view.cv2,
        "resize",
        lambda frame, None_, fx, fy, interpolation: resized.append((fx, fy))
        or _frame(),
        raising=False,
    )

    debug_view.present("win", _frame(), 2.0)

    assert resized == [(2.0, 2.0)]
    assert fake_cv2.imshow_calls == ["win"]