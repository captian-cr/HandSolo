import cv2
import numpy as np

from handmouse.partial_tracking import PartialFingerTracker


def _frame() -> np.ndarray:
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    for x, color in ((30, (255, 255, 255)), (90, (180, 255, 120))):
        cv2.rectangle(frame, (x - 7, 33), (x + 7, 47), color, 2)
        cv2.line(frame, (x - 6, 34), (x + 6, 46), color, 2)
        cv2.line(frame, (x + 6, 34), (x - 6, 46), color, 2)
    return frame


def _shift(frame: np.ndarray, dx: int, dy: int = 0) -> np.ndarray:
    transform = np.float32(((1, 0, dx), (0, 1, dy)))
    return cv2.warpAffine(frame, transform, (frame.shape[1], frame.shape[0]))


def test_optical_flow_preserves_index_and_middle_identity() -> None:
    tracker = PartialFingerTracker(max_frames=8)
    frame = _frame()
    tracker.seed(
        frame,
        {
            "Index": (30 / 120, 40 / 80),
            "Middle": (90 / 120, 40 / 80),
        },
    )

    tracked = tracker.track(_shift(frame, 4, 3))
    by_name = {finger.name: finger for finger in tracked}

    assert set(by_name) == {"Index", "Middle"}
    assert abs(by_name["Index"].x - 34 / 120) < 0.02
    assert abs(by_name["Index"].y - 43 / 80) < 0.02
    assert abs(by_name["Middle"].x - 94 / 120) < 0.02


def test_optical_flow_drops_finger_that_leaves_frame() -> None:
    tracker = PartialFingerTracker(max_frames=8)
    frame = _frame()
    tracker.seed(
        frame,
        {
            "Index": (30 / 120, 40 / 80),
            "Middle": (90 / 120, 40 / 80),
        },
    )

    tracked = ()
    for shift in (-8, -16, -24, -32):
        tracked = tracker.track(_shift(frame, shift))

    assert [finger.name for finger in tracked] == ["Middle"]


def test_quality_based_tracking_keeps_identity_beyond_old_eight_frame_limit() -> None:
    tracker = PartialFingerTracker(max_frames=0)
    frame = _frame()
    tracker.seed(
        frame,
        {
            "Index": (30 / 120, 40 / 80),
        },
    )

    tracked = ()
    for shift in range(1, 31):
        tracked = tracker.track(_shift(frame, shift))

    assert len(tracked) == 1
    assert tracked[0].name == "Index"
    assert abs(tracked[0].x - 60 / 120) < 0.03


def test_explicit_frame_limit_remains_available_for_diagnostics() -> None:
    tracker = PartialFingerTracker(max_frames=2)
    frame = _frame()
    tracker.seed(frame, {"Index": (30 / 120, 40 / 80)})

    assert tracker.track(_shift(frame, 1))
    assert tracker.track(_shift(frame, 2))
    assert tracker.track(_shift(frame, 3)) == ()


def test_implausible_single_frame_jump_invalidates_identity(monkeypatch) -> None:
    tracker = PartialFingerTracker(max_frames=0, max_step_ratio=0.05)
    frame = _frame()
    tracker.seed(frame, {"Index": (30 / 120, 40 / 80)})
    calls = 0

    def fake_flow(_previous, _current, points, *_args, **_kwargs):
        nonlocal calls
        calls += 1
        delta = 40.0 if calls == 1 else -40.0
        moved = points.copy()
        moved[:, :, 0] += delta
        status = np.ones((len(points), 1), dtype=np.uint8)
        error = np.zeros((len(points), 1), dtype=np.float32)
        return moved, status, error

    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", fake_flow)

    assert tracker.track(frame) == ()
