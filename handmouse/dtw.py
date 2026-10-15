"""轨迹重采样与 DTW 匹配。

用途：把"在空中画一个字母/形状"的手指轨迹与内置模板比对，给出候选列表。
这不是 OCR，也不需要训练——模板是程序生成的规范折线，靠形状相似度排序。

为什么是"重采样 + DTW"而不是直接比对
------------------------------------
手画轨迹的采样点数、速度、起点全都不可控：同样一个 W，快手 12 个点、
慢手 60 个点，中间还可能停顿。所以先按**弧长**重采样成固定点数（消除速度
影响），再统一平移缩放到单位框（消除位置和大小影响），最后用 DTW 做弹性
对齐（允许局部快慢不一致）。

刻意不做旋转归一化：横线和竖线必须能区分，字母方向也必须有效，所以这里
只提供 ``indicative_angle`` 供调试，不参与匹配。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import atan2, cos, degrees, hypot, sin
from typing import Iterable, Sequence

Point = tuple[float, float]


def dedupe(points: Sequence[Point], epsilon: float = 1e-4) -> list[Point]:
    """去掉相邻重复点；重复点会让弧长重采样的间隔分配失衡。"""

    result: list[Point] = []
    for point in points:
        if not result:
            result.append((float(point[0]), float(point[1])))
            continue
        last = result[-1]
        if hypot(point[0] - last[0], point[1] - last[1]) > epsilon:
            result.append((float(point[0]), float(point[1])))
    return result


def path_length(points: Sequence[Point]) -> float:
    return sum(
        hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(points, points[1:])
    )


def resample(points: Sequence[Point], count: int) -> list[Point]:
    """按弧长等距重采样成 ``count`` 个点。

    用累积弧长做参数化插值，而不是逐段"边插边累加"，这样既不会因为浮点
    累积误差少给一个点，也不需要修改输入序列。退化输入（单点/零长度）会
    复制首点填满，调用方应先做路径长度校验。
    """

    count = max(2, int(count))
    cleaned = dedupe(points)
    if not cleaned:
        return [(0.0, 0.0)] * count
    if len(cleaned) == 1:
        return [cleaned[0]] * count

    cumulative = [0.0]
    for a, b in zip(cleaned, cleaned[1:]):
        cumulative.append(cumulative[-1] + hypot(b[0] - a[0], b[1] - a[1]))
    total = cumulative[-1]
    if total <= 1e-9:
        return [cleaned[0]] * count

    targets = [total * index / (count - 1) for index in range(count)]
    output: list[Point] = []
    segment = 0
    for target in targets:
        while segment < len(cumulative) - 2 and cumulative[segment + 1] < target:
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        if span <= 1e-12:
            output.append(cleaned[segment])
            continue
        ratio = (target - cumulative[segment]) / span
        ax, ay = cleaned[segment]
        bx, by = cleaned[segment + 1]
        output.append((ax + (bx - ax) * ratio, ay + (by - ay) * ratio))
    return output


def normalize(points: Sequence[Point]) -> list[Point]:
    """平移到质心、按最大跨度等比缩放，得到与位置/大小无关的轨迹。"""

    if not points:
        return []
    count = len(points)
    center_x = sum(point[0] for point in points) / count
    center_y = sum(point[1] for point in points) / count
    translated = [(point[0] - center_x, point[1] - center_y) for point in points]
    extent = max(
        max(abs(point[0]) for point in translated),
        max(abs(point[1]) for point in translated),
        1e-9,
    )
    return [(point[0] / extent, point[1] / extent) for point in translated]


def indicative_angle(points: Sequence[Point]) -> float:
    """轨迹主轴角度（度），仅供调试观察，不参与归一化。"""

    if len(points) < 2:
        return 0.0
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    return degrees(atan2(center_y - points[0][1], points[0][0] - center_x))


def prepare(points: Iterable[Point], count: int) -> list[Point]:
    """完整预处理链：去重 → 弧长重采样 → 平移缩放归一化。"""

    return normalize(resample(list(points), count))


def dtw_distance(first: Sequence[Point], second: Sequence[Point]) -> float:
    """经典 DTW，返回按路径长度归一化的平均欧氏代价。

    归一化很关键：不归一化时点数多的模板天然吃亏，候选排序会全都偏向
    点数少的形状。
    """

    rows = len(first)
    columns = len(second)
    if rows == 0 or columns == 0:
        return float("inf")

    # 滚动两行即可，轨迹点数很小（默认 32），不需要更花哨的优化。
    previous = [float("inf")] * (columns + 1)
    previous[0] = 0.0
    for row in range(1, rows + 1):
        current = [float("inf")] * (columns + 1)
        point_a = first[row - 1]
        for column in range(1, columns + 1):
            point_b = second[column - 1]
            cost = hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
            current[column] = cost + min(
                previous[column],
                current[column - 1],
                previous[column - 1],
            )
        previous = current
    return previous[columns] / float(rows + columns)


@dataclass(frozen=True, slots=True)
class TrajectoryTemplate:
    """一条已归一化的规范轨迹。"""

    name: str
    points: tuple[Point, ...]


@dataclass(slots=True)
class MatchCandidate:
    """一个候选结果。"""

    name: str
    distance: float

    def __str__(self) -> str:
        return f"{self.name}({self.distance:.3f})"


@dataclass(slots=True)
class TrajectoryMatcher:
    """模板集合 + 匹配入口。"""

    point_count: int = 32
    templates: dict[str, TrajectoryTemplate] = field(default_factory=dict)

    def add_template(self, name: str, raw: Sequence[Point]) -> None:
        prepared = tuple(prepare(raw, self.point_count))
        self.templates[name.upper()] = TrajectoryTemplate(name.upper(), prepared)

    def match(
        self,
        raw: Sequence[Point],
        max_candidates: int = 3,
    ) -> list[MatchCandidate]:
        if len(templates := self.templates) == 0:
            return []
        prepared = prepare(raw, self.point_count)
        scored = [
            MatchCandidate(template.name, dtw_distance(prepared, template.points))
            for template in templates.values()
        ]
        scored.sort(key=lambda candidate: candidate.distance)
        return scored[: max(1, int(max_candidates))]


def _scale_polyline(points: Sequence[Point]) -> list[Point]:
    """把 0..1 设计稿坐标缩放进 ±1 的归一化框，并放到画布中心。"""

    return [(point[0] - 0.5, point[1] - 0.5) for point in points]


def build_default_matcher(point_count: int = 32) -> TrajectoryMatcher:
    """内置模板：W / C / E / X / 圈 / 横线。

    模板是"单笔连续"折线，即假设用户抬笔也会被摄像头连续跟踪。X 因此按
    "左上→右下→左下→右上"的顺序画（先写 \\ 再写 /），这是空中画 X 最自然
    的连续动作。
    """

    matcher = TrajectoryMatcher(point_count=point_count)

    matcher.add_template(
        "W",
        _scale_polyline(
            [
                (0.12, 0.12),
                (0.31, 0.88),
                (0.50, 0.42),
                (0.69, 0.88),
                (0.88, 0.12),
            ],
        ),
    )
    matcher.add_template(
        "C",
        _scale_polyline(
            [
                (0.80, 0.18),
                (0.52, 0.10),
                (0.24, 0.30),
                (0.18, 0.50),
                (0.24, 0.70),
                (0.52, 0.90),
                (0.80, 0.82),
            ],
        ),
    )
    matcher.add_template(
        "E",
        _scale_polyline(
            [
                (0.20, 0.90),
                (0.20, 0.10),
                (0.82, 0.10),
                (0.22, 0.10),
                (0.22, 0.50),
                (0.74, 0.50),
                (0.22, 0.50),
                (0.22, 0.90),
                (0.82, 0.90),
            ],
        ),
    )
    matcher.add_template(
        "X",
        _scale_polyline(
            [
                (0.16, 0.12),
                (0.84, 0.88),
                (0.20, 0.82),
                (0.80, 0.18),
            ],
        ),
    )
    matcher.add_template(
        "CIRCLE",
        _scale_polyline(
            [
                (
                    0.5 + 0.38 * cos(index * 6.28318 / 24),
                    0.5 + 0.38 * sin(index * 6.28318 / 24),
                )
                for index in range(25)
            ],
        ),
    )
    matcher.add_template(
        "LINE",
        _scale_polyline(
            [
                (0.12, 0.50),
                (0.32, 0.50),
                (0.52, 0.50),
                (0.72, 0.50),
                (0.90, 0.50),
            ],
        ),
    )
    return matcher
