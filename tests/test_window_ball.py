"""悬浮球（窗口底部白色圆球）的纯逻辑测试。

只测不依赖显示器的部分：几何换算、抓放状态机、淡入淡出、回弹曲线。
绘制层（tkinter）在这里不碰——它需要真实窗口，属于实机验证范畴。
"""

from __future__ import annotations

import math

import pytest

from handmouse import window_ball
from handmouse.window_ball import (
    BALL_GAP,
    BALL_RADIUS,
    IDLE_FADE_MS,
    PULL_FULL_PX,
    SPRING_MS,
    WindowBallController,
    ball_center,
    hits_ball,
    resize_for_pull,
    spring_offset,
)

#: 一个 1000x800 的窗口，左上角在 (100, 100)。
RECT = (100, 100, 1100, 900)


# ---- 几何 ----


def test_ball_sits_below_window_bottom_center() -> None:
    """球在窗口底部居中、且**在窗口外面**（不压内容、不挡标题栏）。"""

    cx, cy = ball_center(RECT)
    assert cx == pytest.approx((RECT[0] + RECT[2]) / 2)
    assert cy == pytest.approx(RECT[3] + BALL_GAP + BALL_RADIUS)


def test_hits_ball_tolerates_small_miss() -> None:
    center = ball_center(RECT)
    assert hits_ball(center, *center)
    # 半径内、贴边、以及 4px 容错之内都算命中
    assert hits_ball(center, center[0] + BALL_RADIUS, center[1])
    assert hits_ball(center, center[0] + BALL_RADIUS + 3, center[1])
    # 明显在球外
    assert not hits_ball(center, center[0] + BALL_RADIUS + 20, center[1])


def test_pull_right_extends_right_edge_only() -> None:
    """向右拉 → 只把右边界推出去，左边不动。"""

    left, top, right, bottom = resize_for_pull(RECT, dx=100, dy=0)
    assert left == RECT[0], "向右拉不该动左边界"
    assert right > RECT[2]
    assert (top, bottom) == (RECT[1], RECT[3]), "纯水平拉不该改高度"


def test_pull_left_extends_left_edge_only() -> None:
    """向左拉 → 只把左边界推出去，右边不动（这就是"向左扩展"）。"""

    left, top, right, bottom = resize_for_pull(RECT, dx=-100, dy=0)
    assert left < RECT[0]
    assert right == RECT[2], "向左拉不该动右边界"
    assert (top, bottom) == (RECT[1], RECT[3])


def test_pull_down_enlarges_around_center() -> None:
    """向下拉 → 整体变大，以中心为基准。"""

    left, top, right, bottom = resize_for_pull(RECT, dx=0, dy=150)
    assert (right - left) > (RECT[2] - RECT[0])
    assert (bottom - top) > (RECT[3] - RECT[1])
    assert (left + right) / 2 == pytest.approx((RECT[0] + RECT[2]) / 2, abs=2)
    assert (top + bottom) / 2 == pytest.approx((RECT[1] + RECT[3]) / 2, abs=2)


def test_pull_up_shrinks_around_center() -> None:
    """向上拉 → 整体变小，同样以中心为基准。"""

    left, top, right, bottom = resize_for_pull(RECT, dx=0, dy=-150)
    assert (right - left) < (RECT[2] - RECT[0])
    assert (bottom - top) < (RECT[3] - RECT[1])
    assert (left + right) / 2 == pytest.approx((RECT[0] + RECT[2]) / 2, abs=2)
    assert (top + bottom) / 2 == pytest.approx((RECT[1] + RECT[3]) / 2, abs=2)


def test_pull_is_clamped_so_window_cannot_fly_away() -> None:
    """拉拽超过满量后按满量算，窗口不会一路飞出屏幕。"""

    far = resize_for_pull(RECT, dx=10_000, dy=0)
    full = resize_for_pull(RECT, dx=PULL_FULL_PX, dy=0)
    assert far == full


def test_window_never_shrinks_below_minimum() -> None:
    """缩到极限时仍保留最小可用尺寸，不能缩成一条线。"""

    tiny = (0, 0, 340, 260)
    left, top, right, bottom = resize_for_pull(tiny, dx=0, dy=-PULL_FULL_PX)
    assert right - left >= window_ball.grid_winctl.MIN_WINDOW_W
    assert bottom - top >= window_ball.grid_winctl.MIN_WINDOW_H


def test_diagonal_pull_combines_edge_and_scale() -> None:
    """斜着拉 = 一边扩边一边缩放（他描述的四向行为自然叠加）。"""

    left, top, right, bottom = resize_for_pull(RECT, dx=-120, dy=120)
    assert left < RECT[0], "向左扩展没生效"
    assert (right - left) > (RECT[2] - RECT[0]), "放大没生效"


# ---- 回弹曲线 ----


def test_spring_starts_at_pull_and_decays_to_zero() -> None:
    start = (40.0, -20.0)
    assert spring_offset(0, start) == start
    # 中途必须已经明显衰减
    mid = spring_offset(SPRING_MS / 2, start)
    assert math.hypot(*mid) < math.hypot(*start)
    # 足够久之后归零（不留残影）
    assert spring_offset(SPRING_MS * 3, start) == (0.0, 0.0)


def test_spring_overshoots_at_least_once() -> None:
    """要有"弹回去"的手感：必须出现反向过冲，而不是单调滑回。"""

    start = (40.0, 0.0)
    samples = [spring_offset(t, start)[0] for t in range(0, SPRING_MS, 4)]
    assert any(value < -0.5 for value in samples), "回弹没有过冲，看起来像直接消失"


# ---- 控制器状态机 ----


class _Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> int:
        self.now += int(ms * 1_000_000)
        return self.now


def _patch_window(monkeypatch, rect=RECT, *, maximized=False) -> None:
    monkeypatch.setattr(window_ball.grid_winctl, "foreground_window", lambda **_k: 777)
    monkeypatch.setattr(window_ball.grid_winctl, "window_rect", lambda _h: rect)
    monkeypatch.setattr(window_ball.grid_winctl, "is_maximized", lambda _h: maximized)
    monkeypatch.setattr(window_ball.grid_winctl, "screen_size", lambda: (1920, 1080))


def test_ball_fades_in_for_active_window_then_fades_out_when_idle(monkeypatch) -> None:
    """活跃窗口的球会淡入；静止超过 2 秒后淡出（他明确要求的）。"""

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)

    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha > 0.9, "点了窗口之后球该出现"

    # 静止 1.5 秒还在（未到 2 秒）
    clock.advance_ms(1500)
    controller.update(clock())
    assert controller.track(777).alpha > 0.5

    # 超过 2 秒开始淡出
    clock.advance_ms(IDLE_FADE_MS)
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha < 0.1, "静止 2 秒后必须淡出"


def test_only_the_active_window_shows_a_ball(monkeypatch) -> None:
    """开很多窗口时，只有当前应用有球。"""

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)

    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha > 0.9

    # 前台换成另一个窗口
    monkeypatch.setattr(window_ball.grid_winctl, "foreground_window", lambda **_k: 888)
    for _ in range(80):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha < 0.1, "失去前台的窗口球必须消失"
    assert controller.track(888).alpha > 0.5, "新前台窗口的球要出现"


def test_new_foreground_window_shows_ball_immediately(monkeypatch) -> None:
    """切到某个应用（= 点了它）后球必须**立刻出现**，不能等人去指。

    2026-10-22 实机 bug：球原来只靠"鼠标靠近球"叫醒，可球一开始 alpha=0
    根本看不见，用户不知道该往哪指，于是永远看不到球。他反馈"圆球在哪里"。
    """

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)

    # 一开始什么活动都没有
    controller.update(clock())
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha > 0.9, "切到该应用后球应当自动出现"


def test_ball_fades_after_two_seconds_without_touch(monkeypatch) -> None:
    """静止 2 秒后淡出；再点一下（notice_activity）又回来。"""

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)

    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha > 0.9

    clock.advance_ms(IDLE_FADE_MS + 200)
    for _ in range(60):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha < 0.1, "静止 2 秒后必须淡出"

    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.track(777).alpha > 0.9, "点一下该应用球要回来"


def test_grab_requires_alpha_and_miss_returns_none(monkeypatch) -> None:
    """没点中球、或球还没显示出来时，不能进入拉球模式。"""

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)
    controller.notice_activity()
    controller.update(clock())

    center = ball_center(RECT)
    assert controller.try_grab((center[0] + 300, center[1])) is None, "没点中却抓住了"


def test_grab_swallows_press_and_resizes_window(monkeypatch) -> None:
    """点中球 → 抓住；拖 → 出矩形；松手 → 结束并回弹。"""

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)
    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())

    center = ball_center(RECT)
    anchor = controller.try_grab(center)
    assert anchor is not None and anchor.hwnd == 777
    assert controller.grabbing

    rect = controller.drag_to((center[0] - 120, center[1]))
    assert rect is not None and rect[0] < RECT[0], "向左拉没有扩展左边界"

    hwnd = controller.release()
    assert hwnd == 777
    assert not controller.grabbing
    assert controller.track(777).spring_from_ns > 0, "松手后应当进入回弹动画"


def test_maximized_window_cannot_be_grabbed(monkeypatch) -> None:
    """最大化窗口不给拉——拉了也改不动，会变成假操作。"""

    clock = _Clock()
    _patch_window(monkeypatch, maximized=True)
    controller = WindowBallController(now_ns=clock)
    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.try_grab(ball_center(RECT)) is None


def test_cancel_leaves_window_untouched(monkeypatch) -> None:
    """暂停 / 手丢失时中止拉球，但**不能**顺手把窗口改掉。"""

    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)
    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    controller.try_grab(ball_center(RECT))
    controller.drag_to((ball_center(RECT)[0] - 50, ball_center(RECT)[1]))
    controller.cancel()
    assert not controller.grabbing
    assert controller.drag_to((0, 0)) is None, "中止后不该再产生窗口矩形"


def test_disabled_controller_never_shows_ball(monkeypatch) -> None:
    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock, enabled=False)
    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    assert controller.visible_tracks() == []
    assert controller.try_grab(ball_center(RECT)) is None


# ---- 管道接线：抓住球的那次按下必须被吞掉 --------------------------------


def _worker_with_ball(controller):
    """构造一个只带悬浮球路由所需属性的 ControlWorker（不启线程）。"""

    from handmouse.pipeline import ControlWorker

    worker = ControlWorker.__new__(ControlWorker)
    worker.ball = controller
    worker._cursor_px = None
    return worker


def _active_ball(monkeypatch):
    clock = _Clock()
    _patch_window(monkeypatch)
    controller = WindowBallController(now_ns=clock)
    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    return controller


def test_pipeline_swallows_press_on_ball_and_swallows_release(monkeypatch) -> None:
    """点中球 → 按下**不发给操作系统**；松手也被吞；窗口按拖拽改了尺寸。"""

    from handmouse.contracts import MouseCommand, MouseCommandType

    controller = _active_ball(monkeypatch)
    worker = _worker_with_ball(controller)
    monkeypatch.setattr(window_ball.grid_winctl, "move_resize", lambda *_a: True)

    cx, cy = ball_center(RECT)
    screen = (1920, 1080)

    # 先把光标移到球上（MOVE 用归一化坐标，管道会换算成像素）
    mover = MouseCommand(MouseCommandType.MOVE, 0, x=cx / screen[0], y=cy / screen[1])
    worker._route_to_ball([mover])
    assert worker._cursor_px == pytest.approx((cx, cy))

    press = MouseCommand(MouseCommandType.LEFT_DOWN, 0)
    kept = worker._route_to_ball([press])
    assert kept == [], "点中球的按下必须被吞掉，否则会穿透点到球下面的东西"
    assert controller.grabbing

    # 往左拖 → 左边界外扩
    left_drag = MouseCommand(
        MouseCommandType.MOVE, 0, x=(cx - 120) / screen[0], y=cy / screen[1],
    )
    worker._route_to_ball([left_drag])
    assert controller.track(777).stretch[0] < 0, "球没有被往左扯"

    release = MouseCommand(MouseCommandType.LEFT_UP, 0)
    assert worker._route_to_ball([release]) == [], "松手也要一起吞掉"
    assert not controller.grabbing


def test_pipeline_keeps_press_when_not_on_ball(monkeypatch) -> None:
    """没点中球时，按下照常派发（不能把正常点击吃掉）。"""

    from handmouse.contracts import MouseCommand, MouseCommandType

    controller = _active_ball(monkeypatch)
    worker = _worker_with_ball(controller)
    cx, cy = ball_center(RECT)

    worker._route_to_ball(
        [MouseCommand(MouseCommandType.MOVE, 0, x=(cx + 400) / 1920, y=cy / 1080)],
    )
    press = MouseCommand(MouseCommandType.LEFT_DOWN, 0)
    assert worker._route_to_ball([press]) == [press]
    assert not controller.grabbing


def test_worker_picks_up_ball_injected_after_construction(monkeypatch) -> None:
    """球是后注入的也必须生效（控制线程先构造、叠加层后建好的顺序）。

    2026-10-22 踩过的坑：`GesturePipeline.start()` 里构造控制线程时把
    `pipeline.ball`（当时还是 None）存成了成员，之后 app 再赋值它完全看不到
    ——现象就是"球画出来了但拉不动"。
    """

    from handmouse.contracts import MouseCommand, MouseCommandType

    _patch_window(monkeypatch)
    clock = _Clock()
    holder = {"ball": None}

    from handmouse.pipeline import ControlWorker

    worker = ControlWorker.__new__(ControlWorker)
    worker.ball = None
    worker._cursor_px = None
    worker._ball_provider = lambda: holder["ball"]

    controller = WindowBallController(now_ns=clock)
    controller.notice_activity()
    for _ in range(40):
        clock.advance_ms(16)
        controller.update(clock())
    holder["ball"] = controller
    monkeypatch.setattr(window_ball.grid_winctl, "move_resize", lambda *_a: True)

    cx, cy = ball_center(RECT)
    worker._route_to_ball(
        [MouseCommand(MouseCommandType.MOVE, 0, x=cx / 1920, y=cy / 1080)],
    )
    assert worker._route_to_ball([MouseCommand(MouseCommandType.LEFT_DOWN, 0)]) == []
    assert controller.grabbing, "后注入的球没被认出来"
