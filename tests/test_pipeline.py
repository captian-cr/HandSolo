import threading
import time

from handmouse.config import AppConfig
from handmouse.contracts import GestureType, MouseCommand, MouseCommandType
from handmouse.gestures import GestureEngine
from handmouse.latest_slot import LatestSlot
from handmouse.mouse import MouseController, NullMouseBackend
from handmouse.pipeline import ControlWorker


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
