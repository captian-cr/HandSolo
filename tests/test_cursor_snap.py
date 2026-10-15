import threading
import time
from dataclasses import replace

import pytest

from handmouse import cursor_snap as cursor_snap_module
from handmouse.cursor_snap import (
    CANDIDATE_TTL_SECONDS,
    CursorSnapService,
    MagneticSnapState,
    SnapCandidate,
    WindowsUiAutomationProbe,
)


class FakeProbe:
    def __init__(self, candidate: SnapCandidate | None) -> None:
        self.candidate = candidate
        self.requests: list[tuple[int, int, int]] = []

    def initialize(self) -> None:
        pass

    def close(self) -> None:
        pass

    def find_target(
        self,
        x: int,
        y: int,
        radius_px: int,
    ) -> SnapCandidate | None:
        self.requests.append((x, y, radius_px))
        return self.candidate


class FakeRectangle:
    def __init__(self, left: int, top: int, right: int, bottom: int) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom


class FakeControl:
    def __init__(
        self,
        control_type: int,
        name: str,
        rect: tuple[int, int, int, int],
        clickable: tuple[int, int],
        control_type_name: str,
    ) -> None:
        self.ControlType = control_type
        self.Name = name
        self.BoundingRectangle = FakeRectangle(*rect)
        self.IsEnabled = True
        self.IsOffscreen = False
        self.ControlTypeName = control_type_name
        self._clickable = clickable

    def GetClickablePoint(self) -> tuple[int, int, bool]:
        return (*self._clickable, True)

    def GetParentControl(self) -> None:
        return None


class FakeAutomation:
    def __init__(self, tab: FakeControl, close_button: FakeControl) -> None:
        self.tab = tab
        self.close_button = close_button

    def ControlFromPoint(self, x: int, _y: int) -> FakeControl:
        return self.close_button if x >= 733 else self.tab


def button_candidate() -> SnapCandidate:
    return SnapCandidate(
        x=950,
        y=20,
        left=930,
        top=0,
        right=970,
        bottom=40,
        name="关闭",
        control_type="ButtonControl",
    )


def tab_candidate() -> SnapCandidate:
    return SnapCandidate(
        x=665,
        y=26,
        left=560,
        top=0,
        right=773,
        bottom=52,
        name="浏览器标签页",
        control_type="TabItemControl",
    )


def test_magnetic_state_latches_to_clickable_point_near_button_edge() -> None:
    state = MagneticSnapState(radius_px=30)

    x, y, snapped = state.resolve(920, 20, button_candidate())

    assert (x, y, snapped) == (950, 20, True)
    assert state.target_name == "关闭"


def test_magnetic_state_releases_after_pointer_leaves_release_zone() -> None:
    state = MagneticSnapState(radius_px=30, release_multiplier=1.8)
    state.resolve(920, 20, button_candidate())

    x, y, snapped = state.resolve(800, 200, None)

    assert (x, y, snapped) == (800, 200, False)
    assert state.target_name is None


def test_tab_item_does_not_capture_pointer_near_its_close_button() -> None:
    state = MagneticSnapState(radius_px=30)

    x, y, snapped = state.resolve(750, 20, tab_candidate())

    assert (x, y, snapped) == (750, 20, False)
    assert state.target_name is None


def test_pointer_can_leave_tab_center_and_snap_to_close_button() -> None:
    state = MagneticSnapState(radius_px=30, release_multiplier=1.8)
    close_button = SnapCandidate(
        x=743,
        y=25,
        left=733,
        top=15,
        right=753,
        bottom=35,
        name="关闭标签页",
        control_type="ButtonControl",
    )
    assert state.resolve(665, 26, tab_candidate()) == (665, 26, True)

    x, y, snapped = state.resolve(750, 20, close_button)

    assert (x, y, snapped) == (743, 25, True)
    assert state.target_name == "关闭标签页"


def test_windows_probe_ignores_tab_background_and_finds_close_child() -> None:
    tab = FakeControl(
        50019,
        "浏览器标签页",
        (560, 0, 773, 52),
        (665, 26),
        "TabItemControl",
    )
    close_button = FakeControl(
        50000,
        "关闭标签页",
        (733, 15, 753, 35),
        (743, 25),
        "ButtonControl",
    )
    probe = WindowsUiAutomationProbe()
    probe._auto = FakeAutomation(tab, close_button)

    candidate = probe.find_target(720, 20, 30)

    assert candidate is not None
    assert candidate.name == "关闭标签页"
    assert candidate.control_type == "ButtonControl"
    assert (candidate.x, candidate.y) == (743, 25)


def test_snap_service_converts_between_normalized_and_physical_coordinates() -> None:
    probe = FakeProbe(button_candidate())
    service = CursorSnapService(
        probe,
        screen_width=1000,
        screen_height=500,
        enabled=True,
        radius_px=30,
        start_worker=False,
    )
    try:
        raw = service.transform(0.92, 0.04)
        service.refresh_once()
        snapped = service.transform(0.92, 0.04)
    finally:
        service.shutdown()

    assert raw == (0.92, 0.04)
    assert abs(snapped[0] - 950 / 999) < 1e-9
    assert abs(snapped[1] - 20 / 499) < 1e-9
    assert service.target_name == "关闭"
    assert probe.requests[-1] == (919, 20, 30)


def test_disabled_snap_service_never_calls_probe() -> None:
    probe = FakeProbe(button_candidate())
    service = CursorSnapService(
        probe,
        screen_width=1000,
        screen_height=500,
        enabled=False,
        radius_px=30,
        start_worker=False,
    )
    try:
        assert service.transform(0.5, 0.5) == (0.5, 0.5)
        service.refresh_once()
    finally:
        service.shutdown()

    assert probe.requests == []


# ---------------------------------------------------------------------------
# UIA 三个放行保护的回归测试
# ---------------------------------------------------------------------------


class _SequenceProbe(FakeProbe):
    """按顺序返回候选，用完后返回 None；用于模拟"控件已消失"。"""

    def __init__(self, candidates: list[SnapCandidate | None]) -> None:
        super().__init__(None)
        self._sequence = list(candidates)

    def find_target(self, x: int, y: int, radius_px: int) -> SnapCandidate | None:
        self.requests.append((x, y, radius_px))
        if self._sequence:
            return self._sequence.pop(0)
        return None


class _UnavailableProbe(FakeProbe):
    """模拟缺少 uiautomation / COM 初始化失败。"""

    def __init__(self) -> None:
        super().__init__(button_candidate())
        self.unavailable = True
        self.initialize_calls = 0

    def initialize(self) -> None:
        self.initialize_calls += 1


class _ThreadRecordingProbe(FakeProbe):
    """记录 initialize() 是在哪个线程里被调用的。"""

    def __init__(self) -> None:
        super().__init__(None)
        self.thread_name: str | None = None

    def initialize(self) -> None:
        self.thread_name = threading.current_thread().name


def _service(probe: FakeProbe, **overrides: object) -> CursorSnapService:
    values: dict[str, object] = {
        "screen_width": 1000,
        "screen_height": 500,
        "enabled": True,
        "radius_px": 30,
        "start_worker": False,
    }
    values.update(overrides)
    return CursorSnapService(probe, **values)  # type: ignore[arg-type]


def test_expired_snapshot_is_dropped_instead_of_snapping() -> None:
    probe = FakeProbe(None)
    service = _service(probe)
    # 手工塞一个已经过期的快照：光标就落在按钮边缘，若快照仍有效会被吸到 (950, 20)。
    service._candidate = replace(
        button_candidate(),
        expires_at=time.monotonic() - 5.0,
    )
    try:
        position = service.transform(0.92, 0.04)
    finally:
        service.shutdown()

    assert position == (0.92, 0.04)
    assert service.target_name is None
    assert service._candidate is None


def test_expiry_does_not_release_the_latch_mid_click(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = {"now": 1000.0}
    monkeypatch.setattr(cursor_snap_module.time, "monotonic", lambda: clock["now"])
    probe = FakeProbe(None)
    service = _service(probe)
    service._candidate = replace(button_candidate(), expires_at=1000.5)
    try:
        first = service.transform(0.92, 0.04)
        assert first != (0.92, 0.04)
        assert service.target_name == "关闭"

        # 快照过期，但光标仍压在按钮上（等价于"按下左键那一瞬间"）。
        clock["now"] = 1002.0
        latched = service.transform(0.92, 0.04)
        still_latched = service.target_name
    finally:
        service.shutdown()

    # 过期只丢弃候选，绝不能强行松锁——否则点击瞬间光标会跳走。
    assert latched != (0.92, 0.04)
    assert service._candidate is None
    assert still_latched == "关闭"


def test_context_change_drops_cache_and_latch() -> None:
    probe = _SequenceProbe([button_candidate(), None])
    window = [111, "记事本"]

    def foreground() -> tuple[int, str]:
        return window[0], window[1]

    service = _service(
        probe,
        screen_signature="DPI=PerMonitorV2 显示器=1",
        foreground_provider=foreground,
    )
    try:
        service.transform(0.92, 0.04)
        service.refresh_once()
        service.transform(0.92, 0.04)
        assert service.target_name == "关闭"
        assert service._candidate is not None

        # 前台窗口句柄变了 → 之前那套像素坐标不再可信
        window[0] = 222
        service.refresh_once()

        assert service._candidate is None
        assert service.target_name is None
    finally:
        service.shutdown()


def test_screen_signature_change_drops_cache() -> None:
    probe = _SequenceProbe([button_candidate(), None])
    service = _service(probe, screen_signature="DPI=System 显示器=1")
    try:
        service.transform(0.92, 0.04)
        service.refresh_once()
        service.transform(0.92, 0.04)
        assert service.target_name == "关闭"

        # DPI 模式 / 显示器数量 / 虚拟屏变化 = 换了坐标系
        service.screen_signature = "DPI=PerMonitorV2 显示器=2"
        service.refresh_once()

        assert service.target_name is None
    finally:
        service.shutdown()


def test_invalidate_clears_cache_without_waiting_for_next_probe() -> None:
    probe = FakeProbe(button_candidate())
    service = _service(probe)
    try:
        service.transform(0.92, 0.04)
        service.refresh_once()
        service.transform(0.92, 0.04)
        assert service.target_name == "关闭"
        assert service._candidate is not None
        assert service._context is not None

        service.invalidate()

        assert service._candidate is None
        assert service._context is None
        assert service.target_name is None
    finally:
        service.shutdown()


def test_unavailable_probe_degradation_is_permanent_and_clickable() -> None:
    probe = _UnavailableProbe()
    service = _service(probe)
    try:
        # 磁吸不可用时点击路径照常工作：transform 只做钳制和整数换算，绝不抛错。
        assert service.transform(0.92, 0.04) == (0.92, 0.04)
        service.refresh_once()
        service.refresh_once()
        service.refresh_once()
    finally:
        service.shutdown()

    # 降级只探测/初始化一次，之后彻底关闭，不每 60ms 重试刷日志。
    assert probe.initialize_calls == 1
    assert probe.requests == []
    assert service.target_name is None


def test_probe_initialization_happens_on_the_worker_thread() -> None:
    probe = _ThreadRecordingProbe()
    service = _service(
        probe,
        start_worker=True,
        poll_interval_seconds=0.02,
    )
    try:
        service.transform(0.5, 0.5)
        deadline = time.monotonic() + 2.0
        while not service._probe_initialized and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        service.shutdown()

    # COM 初始化必须留在 find_target 所在线程里，不能跨线程传递 COM 对象。
    assert probe.thread_name == "cursor-snap-uia"
