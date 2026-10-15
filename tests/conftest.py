"""测试用合成手部关键点。"""

from __future__ import annotations

from handmouse.contracts import HandObservation, Landmark


def pointer_points() -> tuple[Landmark, ...]:
    """食指伸出、其余手指弯曲的右手关键点。"""

    values = [
        (0.50, 0.80),  # 0 wrist
        (0.42, 0.70), (0.36, 0.62), (0.30, 0.55), (0.25, 0.48),
        (0.42, 0.60), (0.42, 0.45), (0.42, 0.33), (0.42, 0.20),
        (0.50, 0.60), (0.50, 0.48), (0.51, 0.53), (0.52, 0.58),
        (0.58, 0.61), (0.58, 0.50), (0.59, 0.55), (0.60, 0.60),
        (0.66, 0.64), (0.66, 0.54), (0.67, 0.59), (0.68, 0.64),
    ]
    return tuple(Landmark(x, y) for x, y in values)


def left_pinch_points() -> tuple[Landmark, ...]:
    points = list(pointer_points())
    points[4] = points[8]
    return tuple(points)


def fist_like_left_pinch_points() -> tuple[Landmark, ...]:
    """录像中的侧向左键捏合：其余手指收拢，二维投影近似握拳。"""

    points = list(fist_points())
    points[4] = points[8]
    return tuple(points)


def right_pinch_points() -> tuple[Landmark, ...]:
    points = list(pointer_points())
    points[4] = points[12]
    return tuple(points)


def right_pinch_from_open_points() -> tuple[Landmark, ...]:
    """张掌后主动伸出中指并与拇指接触的右键捏合。"""

    points = list(open_palm_points())
    points[12] = Landmark(0.50, 0.32)
    points[4] = points[12]
    return tuple(points)


def middle_tucked_pointer_points() -> tuple[Landmark, ...]:
    """食指独立伸出时，拇指自然贴近已收拢的中指。

    这是用户录屏中最常见的移动姿势；只看拇指与中指尖距离会将它
    误判成右键捏合。
    """

    points = list(pointer_points())
    points[4] = Landmark(points[12].x + 0.01, points[12].y)
    return tuple(points)


def ambiguous_right_pinch_points() -> tuple[Landmark, ...]:
    """右键捏合过程中食指也短暂靠近拇指的过渡帧。"""

    points = list(right_pinch_points())
    points[8] = Landmark(points[4].x + 0.01, points[4].y)
    # 保留一根自然伸展的无关手指，避免合成数据退化成真正的握拳。
    points[16] = Landmark(0.58, 0.27)
    return tuple(points)


def rotated_scroll_points() -> tuple[Landmark, ...]:
    """张掌向镜头方向翻腕后的深度关键点。

    掌心中心不平移，只改变腕部到中指掌指关节的相对深度，用于验证
    原地翻腕也能产生滚轮事件。
    """

    points = list(open_palm_points())
    points[9] = Landmark(points[9].x, points[9].y, 0.10)
    return tuple(points)


def rotated_projected_closed_points() -> tuple[Landmark, ...]:
    """翻腕后二维投影近似握拳，但掌面深度已明显变化的过渡帧。"""

    points = list(fist_points())
    points[9] = Landmark(points[9].x, points[9].y, 0.10)
    return tuple(points)


def fist_points() -> tuple[Landmark, ...]:
    points = list(pointer_points())
    for tip, x, y in (
        (8, 0.43, 0.56),
        (12, 0.51, 0.58),
        (16, 0.59, 0.60),
        (20, 0.67, 0.63),
    ):
        points[tip] = Landmark(x, y)
    return tuple(points)


def open_palm_points() -> tuple[Landmark, ...]:
    """四指自然伸展的张掌关键点。"""

    points = list(pointer_points())
    for tip, x, y in (
        (12, 0.50, 0.24),
        (16, 0.58, 0.27),
        (20, 0.66, 0.31),
    ):
        points[tip] = Landmark(x, y)
    return tuple(points)


def translate_points(
    points: tuple[Landmark, ...],
    dx: float = 0.0,
    dy: float = 0.0,
) -> tuple[Landmark, ...]:
    return tuple(Landmark(point.x + dx, point.y + dy, point.z) for point in points)


def observation(
    points: tuple[Landmark, ...],
    timestamp_ns: int,
) -> HandObservation:
    return HandObservation(
        landmarks=points,
        handedness="Right",
        timestamp_ns=timestamp_ns,
        inference_ms=12.0,
        handedness_score=0.98,
    )
