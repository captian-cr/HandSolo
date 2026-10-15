import threading
import time

from handmouse.config import AppConfig
from handmouse.contracts import GestureType, MouseCommand, MouseCommandType
from handmouse.gestures import GestureEngine
from handmouse.latest_slot import LatestSlot
from handmouse.mouse import MouseController, NullMouseBackend
from handmouse.pipeline import ControlWorker

from conftest import left_pinch_points, observation

BASE = 1_000_000_000


def test_pinch_lock_cancels_pending_cursor_interpolation() -> None:
    backend = NullMouseBackend()
    engine = GestureEngine(AppConfig())
    worker = ControlWorker(
        LatestSlot(),
        engine,
        MouseController(backend),
        threading.Event(),
        lambda _gesture, _paused: None,
        interpolation_ms=42,
    )
    now = time.monotonic_ns()
    worker._execute_commands(
        [MouseCommand(MouseCommandType.MOVE, now, x=0.20, y=0.30)],
    )
    worker._execute_commands(
        [MouseCommand(MouseCommandType.MOVE, now + 1, x=0.80, y=0.70)],
    )
    moves_before_lock = [item for item in backend.commands if item[0] == "move"]

    engine.current_gesture = GestureType.PINCH_ARMED
    worker._advance_motion(now + 12_000_000)
    moves_after_lock = [item for item in backend.commands if item[0] == "move"]

    assert moves_after_lock == moves_before_lock


def test_tick_state_transition_is_reported() -> None:
    """无观测帧时 engine.tick() 的状态迁移必须回调 on_state。

    on_state 既是 [状态] 日志的唯一出口，也是快照手势的唯一写入口。r2 盖镜头
    后 58 秒无手，engine 内部早已转 lost，但日志与快照始终停在 left_pinch
    ——硬线 H1 因此根本测不出来。
    """

    reported: list[tuple[GestureType, bool]] = []
    config = AppConfig()
    engine = GestureEngine(config)
    worker = ControlWorker(
        LatestSlot(),
        engine,
        MouseController(NullMouseBackend()),
        threading.Event(),
        lambda gesture, paused: reported.append((gesture, paused)),
        interpolation_ms=42,
    )
    engine.process(observation(left_pinch_points(), BASE))
    assert engine.current_gesture is GestureType.LEFT_PINCH
    reported.clear()

    worker._handle_tick(BASE + (config.hand_lost_ms + 50) * 1_000_000)

    assert engine.current_gesture is GestureType.LOST
    assert reported, "手部丢失迁移没有回调 on_state：日志与快照会停在旧手势"
