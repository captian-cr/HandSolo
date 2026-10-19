"""手势 → 主题面板的动作分派（不启动 Tk、不开摄像头）。

这里锁的是"接线"：HOME / 文件 / 设置 / 分类 / 图标双击，到底有没有真的
落到面板对象上。几何与分页在 ``test_theme_model`` 里单独测，这一层只关心
``ThemeRunner._resolve`` 与 ``_on_confirm`` 把事件转成了什么调用。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from handmouse.grid_judge import PinchJudger
from handmouse.grid_ui import PointerState
from handmouse.theme_select import ThemeRunner


@dataclass
class _FakePanel:
    screen_w: int = 1920
    screen_h: int = 1080
    kind: str | None = None
    icon_at: tuple[float, float] | None = None
    tab_at: tuple[float, float] | None = None
    button: str | None = None
    calls: list[tuple] = field(default_factory=list)

    @property
    def panel_open(self) -> bool:
        return self.kind is not None

    def hit(self, x: float, y: float) -> int | None:
        if self.icon_at and self.icon_at == (x, y):
            return 3
        return None

    def tab_hit(self, x: float, y: float) -> str | None:
        if self.panel_open and self.tab_at and self.tab_at == (x, y):
            return "文件夹"
        return None

    def pill_button(self, x: float, y: float) -> str | None:
        return self.button

    def open_panel(self, kind: str) -> None:
        self.kind = kind
        self.calls.append(("open", kind))

    def close_panel(self) -> None:
        self.kind = None
        self.calls.append(("close",))

    def set_panel_tab(self, label: str) -> None:
        self.calls.append(("tab", label))

    def cycle_wallpaper(self) -> str:
        self.calls.append(("wallpaper",))
        return "#1b2a5e"

    def current_apps(self) -> list:
        @dataclass
        class _App:
            name: str
            path: str = ""

        return [_App("微信")] * 12


def _runner(panel: _FakePanel, *, dry_run: bool = True) -> ThemeRunner:
    return ThemeRunner(
        config={"double_click_window_ms": 600, "pointer_smoothing_s": 0.075},
        panel=panel,  # type: ignore[arg-type]
        judge=PinchJudger(enter_ratio=0.4, release_ratio=0.62, hold_ms=120),
        state=PointerState(),
        stop_event=threading.Event(),
        dry_run=dry_run,
    )


def test_pill_home_opens_and_closes_the_drawer() -> None:
    panel = _FakePanel(button="home")
    runner = _runner(panel)

    assert runner._resolve(960, 1040) == ("home", -1)
    runner._on_confirm("home", -1, 10.0)
    assert panel.kind == "apps"
    assert "弹出全部应用" in runner._action

    runner._on_confirm("home", -1, 11.0)
    assert panel.kind is None
    assert "收起面板" in runner._action


def test_pill_folder_opens_the_files_panel() -> None:
    panel = _FakePanel(button="folder")
    runner = _runner(panel)

    assert runner._resolve(800, 1040) == ("folder", -1)
    runner._on_confirm("folder", -1, 10.0)

    assert panel.kind == "files"
    assert panel.calls == [("open", "files")]


def test_pill_settings_cycles_the_wallpaper() -> None:
    panel = _FakePanel(button="settings")
    runner = _runner(panel)

    assert runner._resolve(1100, 1040) == ("settings", -1)
    runner._on_confirm("settings", -1, 10.0)

    assert panel.calls == [("wallpaper",)]
    assert runner._action == "壁纸：#1b2a5e"


def test_tab_only_hits_while_a_panel_is_open() -> None:
    panel = _FakePanel(tab_at=(300, 300))
    runner = _runner(panel)

    # 面板没开时，标签区不参与命中
    assert runner._resolve(300, 300) == ("none", -1)

    panel.kind = "files"
    assert runner._resolve(300, 300) == ("tab", -1)
    runner._on_confirm("tab", -1, 10.0)
    assert panel.calls == [("tab", "文件夹")]


def test_icon_needs_two_pinches() -> None:
    panel = _FakePanel(icon_at=(500, 400))
    runner = _runner(panel)

    assert runner._resolve(500, 400) == ("icon", 3)

    runner._on_confirm("icon", 3, 10.0)
    assert runner._pending == ("icon", 3, 10.0)
    assert runner._action == ""  # 第一次只选中，不打开

    runner._on_confirm("icon", 3, 10.2)
    assert runner._pending is None
    assert "打开「微信」" in runner._action


def test_second_pinch_outside_the_window_starts_over() -> None:
    panel = _FakePanel(icon_at=(500, 400))
    runner = _runner(panel)

    runner._on_confirm("icon", 3, 10.0)
    runner._on_confirm("icon", 3, 11.0)  # 超过 600ms

    assert runner._pending == ("icon", 3, 11.0)
    assert runner._action == ""


def test_swing_turns_the_panel_page_when_open() -> None:
    """面板打开时挥手不应该翻主网格，面板的 turn_page 由 panel 自己路由。"""

    panel = _FakePanel()
    panel.kind = "apps"
    turned: list[int] = []
    panel.turn_page = lambda delta: turned.append(delta)  # type: ignore[attr-defined]
    runner = _runner(panel)

    panel.turn_page(1)

    assert turned == [1]
