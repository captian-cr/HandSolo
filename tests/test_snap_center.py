"""磁吸落点与"冻结期间也要吸附"（2026-10-23）。

用户原话：「每次都会被吸到图标的不同位置，很难被吸到小标的中央」。
查下来是两件事叠在一起：

1. **落点算错了**：桌面图标的候选矩形是"图案 + 文字"整格，格子中心落在
   文字上（单行偏低 11px、双行偏低 21px），所以不同图标落点高低不一。
2. **磁吸在最该生效的时候根本不工作**：`snap_engaged` 为真的时刻（停住满
   一秒、捏合前锚定）恰好都被 `cursor_frozen` 拦在门外，而它要落点中心
   就必须发 MOVE。于是磁吸只算不做。

所以这里既测"落点选的是图案中心"，也测"冻结期间 MOVE 必须被放行给磁吸"。
"""

from __future__ import annotations

import threading
import time

from handmouse.config import AppConfig
from handmouse.contracts import GestureType
from handmouse.cursor_snap import (
    MagneticSnapState,
    SnapCandidate,
)
from handmouse.grid_desktop import DesktopIcon
from handmouse.latest_slot import LatestSlot
from handmouse.mouse import MouseController, NullMouseBackend
from handmouse.pipeline import ControlWorker
from handmouse.screen_map import ScreenMap

BASE = 1_000_000_000


def _candidate(**overrides: object) -> SnapCandidate:
    values = {
        "x": 100,
        "y": 100,
        "left": 50,
        "top": 40,
        "right": 150,
        "bottom": 140,
    }
    values.update(overrides)
    return SnapCandidate(**values)  # type: ignore[arg-type]


# ---- 1. 落点 ----

def test_explicit_snap_point_wins_over_rect_center() -> None:
    """桌面图标：命中判定用整格，落点用图案中心。"""

    state = MagneticSnapState(radius_px=34, to_center=True)
    candidate = _candidate(snap_x=100, snap_y=60)

    x, y, snapped = state.resolve(100, 90, candidate)

    assert snapped is True
    assert (x, y) == (100, 60), "应该落在图案中心，而不是整格中心 (100,90)"


def test_explicit_snap_point_also_wins_when_not_to_center() -> None:
    """``to_center`` 关掉时也是显式落点优先——否则图标又会落回点击点。"""

    state = MagneticSnapState(radius_px=34, to_center=False)
    candidate = _candidate(snap_x=100, snap_y=60)

    x, y, _ = state.resolve(100, 90, candidate)

    assert (x, y) == (100, 60)


def test_without_explicit_snap_point_uses_rect_center() -> None:
    state = MagneticSnapState(radius_px=34, to_center=True)

    x, y, _ = state.resolve(100, 90, _candidate())

    assert (x, y) == (100, 90)


# ---- 2. 图标图案中心 ----

def test_icon_center_uses_glyph_area_when_available() -> None:
    """实测数据：格子 y 5—92，图案区 y 5—69，图案中心 37（格子中心 48）。"""

    icon = DesktopIcon(
        index=0, name="回收站", left=0, top=5, right=94, bottom=92,
        icon_top=5, icon_bottom=69,
    )

    assert icon.icon_center_y == 37
    assert icon.center[1] == 48


def test_icon_center_falls_back_to_cell_when_glyph_unreadable() -> None:
    icon = DesktopIcon(
        index=0, name="回收站", left=0, top=5, right=94, bottom=92,
    )

    assert icon.icon_center_y == 48


def test_two_line_icon_keeps_the_same_glyph_centre() -> None:
    """双行名字的整格更高（107 vs 87），图案中心必须仍对齐。"""

    single = DesktopIcon(
        index=0, name="A", left=0, top=5, right=94, bottom=92,
        icon_top=5, icon_bottom=69,
    )
    double = DesktopIcon(
        index=1, name="B", left=95, top=5, right=189, bottom=112,
        icon_top=5, icon_bottom=69,
    )

    assert single.icon_center_y == double.icon_center_y == 37
    assert single.center[1] != double.center[1], "整格中心本来就不一样，这正是旧行为的问题"


# ---- 3. 冻结期间磁吸要能动 ----

class _StubEngine:
    """只提供 ControlWorker 需要的几个字段，避免伪造整个状态机。"""

    def __init__(
        self,
        *,
        frozen: bool,
        engaged: bool,
        config: AppConfig | None = None,
        position: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.cursor_frozen = frozen
        self.snap_engaged = engaged
        self.paused = False
        self.current_gesture = GestureType.POINTER
        self.cursor_position = position
        # ControlWorker 通过 engine.config 读 cursor_snap_windowed。
        self.config = config or AppConfig()


def _worker(config: AppConfig) -> tuple[ControlWorker, NullMouseBackend]:
    backend = NullMouseBackend(ScreenMap(dpi_aware=False))
    worker = ControlWorker(
        LatestSlot(),
        _StubEngine(frozen=True, engaged=True, config=config),  # type: ignore[arg-type]
        MouseController(backend),
        threading.Event(),
        lambda _gesture, _paused: None,
        interpolation_ms=42,
    )
    return worker, backend


def test_frozen_cursor_still_gets_snapped() -> None:
    """冻结 + 磁吸窗口打开 → 必须把冻结位置重新送出去，磁吸才有机会落中心。"""

    worker, backend = _worker(AppConfig())

    worker._advance_motion(BASE)

    moves = [item for item in backend.commands if item[0] == "move"]
    assert len(moves) == 1, "冻结期间磁吸被完全拦住了"
    assert moves[0][1] == (0.25, 0.75), "重发的应该是引擎记的冻结位置，而不是别处"


def test_frozen_without_snap_window_stays_put() -> None:
    """不在磁吸窗口内时，冻结仍然是"一格都不许动"。"""

    worker, backend = _worker(AppConfig())
    worker.engine.snap_engaged = False  # type: ignore[attr-defined]

    worker._advance_motion(BASE)

    assert [item for item in backend.commands if item[0] == "move"] == []


def test_frozen_snap_can_be_disabled_for_old_behaviour() -> None:
    """``cursor_snap_windowed=False``（旧行为）下，冻结期间照样要重发。"""

    config = AppConfig()
    config.cursor_snap_windowed = False
    worker, backend = _worker(config)
    worker.engine.snap_engaged = False  # type: ignore[attr-defined]

    worker._advance_motion(BASE)

    assert len([item for item in backend.commands if item[0] == "move"]) == 1
