"""DTW 轨迹匹配测试。

覆盖两层：
1. 数学性质——弧长重采样、归一化、DTW 距离本身是否可靠；
2. 模板回归——W/C/E/X/CIRCLE/LINE 六个内置形状在缩放、平移、噪声下
   是否仍能命中最优模板，以及不相似的笔画是否会被拒。
"""

from __future__ import annotations

import math

import pytest

from handmouse.config import AppConfig
from handmouse.dtw import (
    build_default_matcher,
    dedupe,
    dtw_distance,
    normalize,
    path_length,
    prepare,
    resample,
)

# 与 dtw.build_default_matcher 中相同的设计稿折线（0..1 坐标系）。
DESIGN: dict[str, list[tuple[float, float]]] = {
    "W": [(0.12, 0.12), (0.31, 0.88), (0.50, 0.42), (0.69, 0.88), (0.88, 0.12)],
    "C": [
        (0.80, 0.18), (0.52, 0.10), (0.24, 0.30), (0.18, 0.50),
        (0.24, 0.70), (0.52, 0.90), (0.80, 0.82),
    ],
    "E": [
        (0.20, 0.90), (0.20, 0.10), (0.82, 0.10), (0.22, 0.10), (0.22, 0.50),
        (0.74, 0.50), (0.22, 0.50), (0.22, 0.90), (0.82, 0.90),
    ],
    "X": [(0.16, 0.12), (0.84, 0.88), (0.20, 0.82), (0.80, 0.18)],
    "CIRCLE": [
        (
            0.5 + 0.38 * math.cos(index * 6.28318 / 24),
            0.5 + 0.38 * math.sin(index * 6.28318 / 24),
        )
        for index in range(25)
    ],
    "LINE": [(0.12, 0.50), (0.32, 0.50), (0.52, 0.50), (0.72, 0.50), (0.90, 0.50)],
}


def _densify(
    points: list[tuple[float, float]],
    steps: int = 12,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    noise: float = 0.0,
) -> list[tuple[float, float]]:
    """把折线按固定步长加密成"手绘轨迹"。

    摄像头不会只采到折线顶点，而是每帧一个点，中间点落在段内。加密后再加
    缩放/平移/噪声，用来验证匹配对这些扰动不敏感。
    """

    dense: list[tuple[float, float]] = []
    for start, end in zip(points, points[1:]):
        for step in range(steps):
            ratio = step / steps
            dense.append(
                (
                    start[0] + (end[0] - start[0]) * ratio,
                    start[1] + (end[1] - start[1]) * ratio,
                ),
            )
    dense.append(points[-1])
    if noise:
        dense = [
            (x + noise * math.sin(index * 1.7), y + noise * math.cos(index * 2.3))
            for index, (x, y) in enumerate(dense)
        ]
    return [
        (
            0.5 + (x - 0.5) * scale + offset[0],
            0.5 + (y - 0.5) * scale + offset[1],
        )
        for x, y in dense
    ]


# ---------------- 数学性质 ----------------
def test_resample_returns_exact_count_and_keeps_endpoints() -> None:
    raw = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    sampled = resample(raw, 16)

    assert len(sampled) == 16
    assert sampled[0] == pytest.approx(raw[0])
    assert sampled[-1] == pytest.approx(raw[-1])


def test_resample_spaces_points_evenly_along_arc_length() -> None:
    # 折线总长 2.0；采样 9 个点，步长应为 0.25。
    sampled = resample([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], 9)
    steps = [path_length(pair) for pair in zip(sampled, sampled[1:])]

    assert max(steps) == pytest.approx(min(steps), abs=1e-6)
    assert steps[0] == pytest.approx(2.0 / 8)


def test_resample_does_not_overshoot_short_inputs() -> None:
    # 只有两个点（一段）时也必须给出请求的点数，而不是按顶点数截断。
    sampled = resample([(0.2, 0.3), (0.6, 0.7)], 32)
    assert len(sampled) == 32


def test_dedupe_drops_repeated_points() -> None:
    raw = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.5, 0.0), (0.5, 0.0)]

    assert dedupe(raw) == [(0.0, 0.0), (0.5, 0.0)]


def test_path_length_sums_segments() -> None:
    assert path_length([(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)]) == pytest.approx(7.0)
    assert path_length([(1.0, 1.0)]) == 0.0


@pytest.mark.parametrize(
    "scale,offset",
    [(0.35, (0.0, 0.0)), (1.0, (0.05, -0.03)), (2.2, (-0.10, 0.10))],
)
def test_normalize_is_invariant_to_scale_and_translation(
    scale: float,
    offset: tuple[float, float],
) -> None:
    raw = DESIGN["W"]
    moved = [
        (
            0.5 + (x - 0.5) * scale + offset[0],
            0.5 + (y - 0.5) * scale + offset[1],
        )
        for x, y in raw
    ]

    assert dtw_distance(prepare(raw, 32), prepare(moved, 32)) == pytest.approx(0.0, abs=1e-9)


def test_normalize_centers_trajectory_on_origin() -> None:
    normalized = normalize([(2.0, 5.0), (4.0, 5.0), (3.0, 6.0)])

    assert sum(point[0] for point in normalized) / 3 == pytest.approx(0.0, abs=1e-9)
    assert sum(point[1] for point in normalized) / 3 == pytest.approx(0.0, abs=1e-9)
    assert max(max(abs(x), abs(y)) for x, y in normalized) == pytest.approx(1.0)


def test_dtw_distance_is_zero_for_identical_trajectory_and_symmetric() -> None:
    first = prepare(DESIGN["C"], 32)
    second = prepare(_densify(DESIGN["X"]), 32)

    assert dtw_distance(first, first) == pytest.approx(0.0, abs=1e-9)
    assert dtw_distance(first, second) == pytest.approx(dtw_distance(second, first), abs=1e-9)


def test_dtw_distance_is_infinite_for_empty_input() -> None:
    assert dtw_distance([], [(0.0, 0.0)]) == math.inf


# ---------------- 模板回归 ----------------
@pytest.mark.parametrize("name", sorted(DESIGN))
def test_shape_matches_its_own_template(name: str) -> None:
    matcher = build_default_matcher(32)
    candidates = matcher.match(_densify(DESIGN[name]), 6)

    assert candidates, f"{name} 没有产生任何候选"
    assert candidates[0].name == name
    assert candidates[0].distance < AppConfig().motion_accept_distance
    # 第二名必须明显落后，否则说明模板之间不可区分。
    assert candidates[1].distance > candidates[0].distance * 2


@pytest.mark.parametrize("name", sorted(DESIGN))
@pytest.mark.parametrize("noise", [0.01, 0.02])
def test_shape_survives_hand_jitter(name: str, noise: float) -> None:
    matcher = build_default_matcher(32)
    candidates = matcher.match(_densify(DESIGN[name], noise=noise), 3)

    assert candidates[0].name == name
    assert candidates[0].distance < AppConfig().motion_accept_distance


def test_vertical_stroke_is_not_accepted() -> None:
    """竖线不是内置形状，匹配距离必须高于接受阈值。"""

    matcher = build_default_matcher(32)
    vertical = _densify([(0.50, 0.10), (0.50, 0.30), (0.50, 0.50), (0.50, 0.70), (0.50, 0.90)])
    candidates = matcher.match(vertical, 3)

    assert candidates[0].distance > AppConfig().motion_accept_distance


def test_line_and_circle_are_distinguishable() -> None:
    matcher = build_default_matcher(32)
    line = matcher.match(_densify(DESIGN["LINE"]), 6)
    circle = matcher.match(_densify(DESIGN["CIRCLE"]), 6)

    assert line[0].name == "LINE" and line[0].distance == pytest.approx(0.0, abs=1e-6)
    assert circle[0].name == "CIRCLE" and circle[0].distance == pytest.approx(0.0, abs=1e-6)
    # 两者的交叉距离必须远大于各自的自匹配距离，否则横线和圈无法区分。
    cross = dict((candidate.name, candidate.distance) for candidate in line)
    assert cross["CIRCLE"] > AppConfig().motion_accept_distance * 2


def test_matcher_returns_nothing_without_templates() -> None:
    from handmouse.dtw import TrajectoryMatcher

    assert TrajectoryMatcher(point_count=32).match(_densify(DESIGN["W"])) == []


def test_candidate_count_is_capped_and_positive() -> None:
    matcher = build_default_matcher(32)

    assert len(matcher.match(_densify(DESIGN["W"]), 3)) == 3
    assert len(matcher.match(_densify(DESIGN["W"]), 0)) == 1
