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


def test_wide_tab_item_is_snappable_along_its_whole_row() -> None:
    """整行 TabItem 必须整条可吸，否则两端永远够不到。

    旧契约是"TabItem 只吸中心锚点"，那是为浏览器标签页写的。但 WorkBuddy
    左侧栏那种 318×40 的整行条目也报 TabItemControl（2026-10-19 实测），
    按中心算，两端各有一百多像素吸不到，用户反馈"这些条目移不过去"。
    现在改成整条按矩形吸附，标签页里的关闭按钮改由"嵌套控件优先"保证。
    """

    state = MagneticSnapState(radius_px=30)

    x, y, snapped = state.resolve(750, 20, tab_candidate())

    assert (x, y, snapped) == (665, 26, True)
    assert state.target_name == "浏览器标签页"


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


def test_desktop_icon_probe_snaps_to_icon_rect(monkeypatch) -> None:
    """桌面图标走矩形候选源：UIA 在图标上探测会返回任务栏按钮，不能用。"""

    import time as _time

    from handmouse import cursor_snap as module

    monkeypatch.setattr(module, "_desktop_at", lambda x, y: True)
    probe = module.DesktopIconProbe(cache_seconds=999.0)
    probe._icons = [
        module.SnapCandidate(
            x=50,
            y=50,
            left=20,
            top=20,
            right=80,
            bottom=80,
            name="测试图标",
            control_type="DesktopIcon",
        ),
    ]
    probe._loaded_at = _time.monotonic()

    candidate = probe.find_target(95, 50, 55)
    assert candidate is not None
    assert candidate.name == "测试图标"
    assert probe.find_target(600, 600, 55) is None


def test_desktop_icon_probe_ignores_when_desktop_is_covered(monkeypatch) -> None:
    from handmouse import cursor_snap as module

    monkeypatch.setattr(module, "_desktop_at", lambda x, y: False)
    probe = module.DesktopIconProbe()
    assert probe.find_target(50, 50, 80) is None


def test_custom_drawn_button_is_snapped_by_size() -> None:
    """自绘 UI 里没有 Button 节点，靠矩形尺寸兜底。

    2026-10-20 用户反馈"有些软件没有磁性和选中的框框"：那些应用的按钮在
    UIA 里只有 Custom/Pane 节点，整条父链都没有白名单类型，旧实现直接
    返回 None，表现为完全吸不到。
    """

    custom = FakeControl(
        50025,
        "自定义按钮",
        (100, 100, 140, 130),
        (120, 115),
        "CustomControl",
    )
    probe = WindowsUiAutomationProbe()
    probe._auto = FakeAutomation(custom, custom)

    candidate = probe.find_target(120, 115, 30)

    assert candidate is not None
    assert candidate.name == "自定义按钮"


def test_large_custom_panel_is_not_treated_as_button() -> None:
    """整块面板必须被尺寸判据挡掉，否则光标会往窗口中间乱跳。"""

    panel = FakeControl(
        50025,
        "整个面板",
        (0, 0, 1200, 800),
        (600, 400),
        "CustomControl",
    )
    probe = WindowsUiAutomationProbe()
    probe._auto = FakeAutomation(panel, panel)

    assert probe.find_target(600, 400, 30) is None


def test_titlebar_fallback_is_safe_outside_screen() -> None:
    """标题栏兜底只能猜，但绝不能抛异常把主循环带崩。"""

    from handmouse.cursor_snap import _titlebar_fallback

    assert _titlebar_fallback(-9000, -9000, 30) is None


# ---------------------------------------------------------------------------
# 纯图标按钮（2026-10-20 用户反馈：播放器控制条的「−」不吸附，滑块/全屏正常）
# ---------------------------------------------------------------------------


class _ChainedControl(FakeControl):
    """带父链的控件，用来模拟"图标 → 大容器"这种没有 Button 的结构。"""

    def __init__(self, control_type, name, rect, clickable, ctn, parent=None):
        super().__init__(control_type, name, rect, clickable, ctn)
        self._parent = parent

    def GetParentControl(self):
        return self._parent


def test_icon_only_button_is_snap_target() -> None:
    """父链上没有 Button 时，图标自身必须仍能作为落点。

    这是用户报的 bug：播放器控制条的「−」按钮在 UIA 里只有一层 Image，
    父链是个覆盖整条控制条的大 Pane（被 _looks_like_button 的 220px 上限否掉），
    于是这个按钮整个消失，而同一控制条上有 Button/Slider 节点的目标照常能吸。
    """

    big_bar = _ChainedControl(
        50033, "控制条", (0, 400, 800, 440), (400, 420), "PaneControl",
    )
    minus_icon = _ChainedControl(
        50006, "", (100, 405, 120, 425), (110, 415), "ImageControl",
        parent=big_bar,
    )

    probe = WindowsUiAutomationProbe()
    probe._auto = type("Auto", (), {"ControlFromPoint": staticmethod(lambda x, y: minus_icon)})()

    candidate = probe.find_target(110, 415, 22)

    assert candidate is not None, "纯图标按钮不该被整个丢掉"
    assert (candidate.left, candidate.top, candidate.right, candidate.bottom) == (
        100, 405, 120, 425,
    )


def test_real_button_still_wins_over_icon() -> None:
    """父链上有真Button 时仍优先用 Button —— 它的点击区通常比图标大。"""

    button = _ChainedControl(
        50000, "全屏", (300, 400, 340, 440), (320, 420), "ButtonControl",
    )
    icon = _ChainedControl(
        50006, "", (310, 408, 330, 432), (320, 420), "ImageControl",
        parent=button,
    )

    probe = WindowsUiAutomationProbe()
    probe._auto = type("Auto", (), {"ControlFromPoint": staticmethod(lambda x, y: icon)})()

    candidate = probe.find_target(320, 420, 22)

    assert candidate is not None
    assert candidate.control_type == "ButtonControl"
    assert (candidate.left, candidate.right) == (300, 340)


def test_oversized_image_is_not_treated_as_icon() -> None:
    """大块图片/画布不能因为"是 Image"就被吸过来，否则光标会跳到图片中心。"""

    from handmouse.cursor_snap import _looks_like_icon

    big_image = _ChainedControl(
        50006, "", (0, 0, 1200, 800), (600, 400), "ImageControl",
    )
    assert _looks_like_icon(big_image) is False

    ok_icon = _ChainedControl(
        50006, "", (100, 405, 120, 425), (110, 415), "ImageControl",
    )
    assert _looks_like_icon(ok_icon) is True


# ---------------------------------------------------------------------------
# 「进框才吸 + 吸到中心」（2026-10-21 用户要求）
# ---------------------------------------------------------------------------


def _candidate(left, top, right, bottom, px, py):
    """构造一个推荐点击点与矩形中心不同的候选，用于区分两种落点。"""

    return SnapCandidate(
        x=px,
        y=py,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        name="按钮",
        control_type="Button",
    )


def test_inside_only_does_not_pull_from_outside() -> None:
    """光标还在控件外面（哪怕在 radius 内）也不能被拽进去。

    旧行为是从矩形外 radius_px 就开始往里吸，用户表现为"路过就被吸走"。
    现在改成进框才吸，框外必须原样返回。
    """

    button = _candidate(100, 100, 140, 130, 105, 105)
    state = MagneticSnapState(22, inside_only=True)

    snapped_x, snapped_y, snapped = state.resolve(150, 115, button)  # 框右侧 10px

    assert snapped is False
    assert (snapped_x, snapped_y) == (150, 115)


def test_inside_only_snaps_once_cursor_is_inside() -> None:
    button = _candidate(100, 100, 140, 130, 105, 105)
    state = MagneticSnapState(22, inside_only=True)

    snapped_x, snapped_y, snapped = state.resolve(120, 115, button)

    assert snapped is True
    assert (snapped_x, snapped_y) == (105, 105)  # to_center=False 时用点击点


def test_to_center_uses_rect_centre_not_clickable_point() -> None:
    """吸到中心：落点必须是矩形正中，而不是 UIA 的 clickable point。"""

    button = _candidate(100, 100, 140, 130, 105, 105)  # 点击点在左上角附近
    state = MagneticSnapState(22, inside_only=True, to_center=True)

    snapped_x, snapped_y, snapped = state.resolve(120, 115, button)

    assert snapped is True
    assert (snapped_x, snapped_y) == (120, 115)  # (100+140)/2, (100+130)/2


def test_legacy_behaviour_unchanged_when_flags_off() -> None:
    """两个开关都关时保持旧行为：框外 radius 内也吸，落点用点击点。"""

    button = _candidate(100, 100, 140, 130, 105, 105)
    state = MagneticSnapState(22, inside_only=False, to_center=False)

    snapped_x, snapped_y, snapped = state.resolve(150, 115, button)

    assert snapped is True
    assert (snapped_x, snapped_y) == (105, 105)


def test_refresh_once_stays_silent_when_not_engaged() -> None:
    """性能回归（2026-10-21）：未进入预捏合窗口时必须完全不探测。

    后台线程每 ~50ms 调一次 refresh_once，若不做 engaged 闸门，就会以
    ~20 次/秒的频率持续扫桌面 UI Automation（每次 13 个 ControlFromPoint），
    这是光标发顿/系统卡顿的元凶。
    """

    probe = FakeProbe(None)
    service = _service(probe)
    try:
        service._pending = (100, 100)

        service.set_engaged(False)
        service.refresh_once()
        assert probe.requests == [], "未进入窗口却在探测：后台 UIA 扫描没关掉"

        # 进入预捏合窗口后才允许探测。
        service.set_engaged(True)
        service.refresh_once()
        assert len(probe.requests) == 1, "进入窗口后应当正常探测"
    finally:
        service.shutdown()


def test_disengaging_keeps_latch_for_grace_then_drops_stale_candidate() -> None:
    """退出预捏合窗口：**先留一段宽限，再丢弃**候选与锁存。

    2026-10-23 改（原断言是"立刻丢"）。他反馈「夹在两个小图标中间左右
    弹跳」：窗口一关就丢锁存 → 光标从"控件中心"弹回"随手位置"，而手仍在
    漂移，下一帧窗口又开、又锁到旁边那个图标。一开一关就是一左一右。
    所以关窗后先保留 `release_grace_seconds`，等手真的动了（离关窗点超过
    radius_px）或宽限到期才丢。
    """

    probe = FakeProbe(button_candidate())
    service = _service(probe)
    try:
        service.set_engaged(True)
        assert service.snap_window_open(0.5, 0.5) is True  # 顺带把 _pending 刷到位
        service.refresh_once()
        assert service._candidate is not None

        # 关窗：宽限期内仍然保留，光标不会被当场弹回。
        service.set_engaged(False)
        assert service._engaged is False, "探测闸门必须立刻关掉（省掉后台 UIA 扫描）"
        assert service._candidate is not None, "宽限期内不该丢候选，否则光标会弹回"
        assert service.snap_window_open(0.5, 0.5) is True

        # 手明确离开（位移超过 radius_px）→ 宽限立刻作废。
        far_x = 0.5 + (service.radius_px * 4) / (service.screen_width - 1)
        assert service.snap_window_open(far_x, 0.5) is False
        assert service._candidate is None
        assert service.target_name is None
    finally:
        service.shutdown()


def test_disengage_grace_expires() -> None:
    """宽限到期即丢锁存：不能永远留着一个过期目标。"""

    probe = FakeProbe(button_candidate())
    service = _service(probe, release_grace_seconds=0.02)
    try:
        service.set_engaged(True)
        assert service.snap_window_open(0.5, 0.5) is True
        service.refresh_once()
        assert service._candidate is not None

        service.set_engaged(False)
        assert service._candidate is not None
        time.sleep(0.05)
        assert service.snap_window_open(0.5, 0.5) is False
        assert service._candidate is None
    finally:
        service.shutdown()


# ---------------------------------------------------------------------------
# 磁吸"吸谁"：重叠小标要按中心裁决（2026-10-22）
# ---------------------------------------------------------------------------


def test_overlapping_neighbours_pick_the_one_whose_centre_is_closer() -> None:
    """两个小标矩形重叠时，吸**中心更近**的那个，不能被面积小的抢走。

    他原话：「吸在这些小标的左右上下，会离另外一个小标很近，然后有时候
    就会被其他小标吸走，就是要吸在这些小标的中央才行」。
    桌面图标（图标+文字）的矩形又宽又高，相邻两个常常互相重叠——此时
    ``_activation_distance`` 两边都是 0，若再按面积裁决就会选错。
    """

    # A 宽而扁（面积大），中心在 (100,100)；B 小而窄（面积小），中心在 (170,100)。
    # 两者矩形互相重叠；光标在 (120,100)，明显更靠近 A 的中心。
    first = SnapCandidate(
        x=100, y=100, left=40, top=60, right=160, bottom=140,
        name="A", control_type="DesktopIcon", target_id="a",
    )
    second = SnapCandidate(
        x=170, y=100, left=150, top=90, right=190, bottom=110,
        name="B", control_type="DesktopIcon", target_id="b",
    )

    from handmouse.cursor_snap import _best_candidate

    best = _best_candidate(120, 100, [first, second])
    assert best.name == "A", "中心更近的候选没有被优先（会被邻标抢走）"

    # 反过来：光标挪到更靠近 B 的中心时，应当换成 B
    assert _best_candidate(160, 100, [first, second]).name == "B"


# ---------------------------------------------------------------------------
# 标题栏按钮滞回：不许在"最大化 / 关闭"边界上来回跳
# ---------------------------------------------------------------------------


def test_titlebar_button_has_hysteresis_at_the_boundary(monkeypatch) -> None:
    """光标在相邻两个标题栏按钮边界抖动时，不许来回切换目标。

    他原话：「在文件夹这里会夹在放大和叉叉的中间反跳」。
    """

    from handmouse import cursor_snap as module

    maximize = SnapCandidate(
        x=734, y=23, left=711, top=0, right=757, bottom=46,
        name="最大化", control_type="TitleBarButton", target_id="titlebar:最大化",
    )
    close = SnapCandidate(
        x=780, y=23, left=757, top=0, right=803, bottom=46,
        name="关闭", control_type="TitleBarButton", target_id="titlebar:关闭",
    )

    def fake_fallback(x, y, radius_px):
        return maximize if x < 757 else close

    monkeypatch.setattr(module, "_titlebar_fallback", fake_fallback)

    probe = WindowsUiAutomationProbe()
    probe._auto = None
    probe._unavailable = False

    # 先在"最大化"内部选上它
    assert probe._titlebar_pick(730, 23, 30).name == "最大化"
    # 抖到边界右侧 4px（技术上属于"关闭"），仍应保持"最大化"
    assert probe._titlebar_pick(761, 23, 30).name == "最大化"
    # 深入"关闭"里（超出 9px 粘滞带）才切换
    assert probe._titlebar_pick(790, 23, 30).name == "关闭"
