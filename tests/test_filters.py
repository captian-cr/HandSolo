from handmouse.filters import (
    AdaptiveEmaFilter,
    CursorMotionInterpolator,
    OneEuroPointerFilter,
    map_activity_region,
)
from statistics import mean, pstdev


def test_activity_region_maps_edges_and_clamps() -> None:
    assert map_activity_region(0.1, 0.2, 0.1, 0.2) == (0.0, 0.0)
    assert map_activity_region(0.9, 0.8, 0.1, 0.2) == (1.0, 1.0)
    assert map_activity_region(-1.0, 2.0, 0.1, 0.2) == (0.0, 1.0)


def test_ema_keeps_tiny_movements_inside_deadzone() -> None:
    filter_ = AdaptiveEmaFilter(deadzone=0.01)
    assert filter_.update(0.5, 0.5) == (0.5, 0.5)
    assert filter_.update(0.505, 0.504) == (0.5, 0.5)


def test_ema_uses_faster_response_for_large_movements() -> None:
    filter_ = AdaptiveEmaFilter(
        base_alpha=0.2,
        fast_alpha=0.8,
        speed_threshold=0.1,
    )
    filter_.update(0.0, 0.0)
    x, y = filter_.update(1.0, 1.0)
    assert x == 0.8
    assert y == 0.8


def test_one_euro_filter_reduces_static_jitter_and_follows_fast_motion() -> None:
    filter_ = OneEuroPointerFilter(min_cutoff=1.7, beta=0.9)
    timestamp = 1_000_000_000
    filter_.update(0.5, 0.5, timestamp)

    stationary = []
    for index, offset in enumerate((0.002, -0.002, 0.0015, -0.001, 0.0), start=1):
        stationary.append(
            filter_.update(
                0.5 + offset,
                0.5 - offset,
                timestamp + index * 16_000_000,
            ),
        )

    assert max(abs(x - 0.5) for x, _ in stationary) < 0.0015
    fast_x, _ = filter_.update(0.8, 0.5, timestamp + 96_000_000)
    assert fast_x > 0.65


def test_cursor_interpolator_fills_gaps_between_detection_frames() -> None:
    interpolator = CursorMotionInterpolator(
        min_duration_ms=16,
        max_duration_ms=48,
    )
    assert interpolator.set_target(0.1, 0.1, 0) == (0.1, 0.1)
    interpolator.set_target(0.16, 0.16, 10_000_000)

    positions = [
        interpolator.sample(timestamp)
        for timestamp in (18_000_000, 26_000_000, 34_000_000, 50_000_000, 70_000_000)
    ]
    values = [position[0] for position in positions if position is not None]

    assert len(values) >= 4
    assert values == sorted(values)
    assert abs(values[-1] - 0.16) < 0.005


def test_cursor_interpolator_does_not_stop_between_steady_visual_frames() -> None:
    """匀速输入不应被插值器改造成每帧起步、刹停的脉冲运动。"""

    interpolator = CursorMotionInterpolator(
        min_duration_ms=12,
        max_duration_ms=42,
    )
    interpolator.set_target(0.10, 0.50, 0)
    interpolator.set_target(0.14, 0.50, 0)
    targets = iter(
        (
            (45_000_000, 0.18),
            (90_000_000, 0.22),
            (135_000_000, 0.26),
            (180_000_000, 0.30),
            (225_000_000, 0.34),
        ),
    )
    next_time, next_x = next(targets)
    positions: list[tuple[int, float]] = []

    for timestamp_ns in range(0, 265_000_001, 8_000_000):
        while timestamp_ns >= next_time:
            interpolator.set_target(next_x, 0.50, next_time)
            try:
                next_time, next_x = next(targets)
            except StopIteration:
                next_time = 10**30
        position = interpolator.sample(timestamp_ns)
        if position is not None:
            positions.append((timestamp_ns, position[0]))

    velocities = [
        (positions[index][1] - positions[index - 1][1]) / 0.008
        for index in range(1, len(positions))
        if 48_000_000 <= positions[index][0] <= 240_000_000
    ]
    coefficient_of_variation = pstdev(velocities) / mean(velocities)

    assert min(velocities) > 0.02
    assert coefficient_of_variation < 0.50
