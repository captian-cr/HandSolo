"""测试用合成手部关键点。"""

from __future__ import annotations

import pytest

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


def entry_palm_points() -> tuple[Landmark, ...]:
    """进入门姿势：五指张开 + 掌心朝镜（镜像自拍坐标系下的右手）。

    坐标系约定：摄像头帧先做 ``cv2.flip`` 再送 MediaPipe，而 MediaPipe 的
    handedness 假定输入是"已镜像的自拍画面"，所以标签就是真实左右手。
    自拍镜像下的右手掌心朝镜，拇指落在画面右侧（解剖学姿势里拇指外侧朝人
    体右侧，正对镜头时在画面左，镜像后翻到右）。此夹具据此构造，掌轴
    (5→17) 与指向拇指的矢量 (5→4) 的二维叉积为正。
    """

    points = open_palm_points()
    center_x = sum(points[index].x for index in (0, 5, 9, 13, 17)) / 5
    return tuple(Landmark(2 * center_x - point.x, point.y, point.z) for point in points)


def entry_back_points() -> tuple[Landmark, ...]:
    """误进防护：同样五指张开，但手背朝镜（拇指落在掌轴另一侧）。"""

    return open_palm_points()


def side_palm_points() -> tuple[Landmark, ...]:
    """误进防护：侧掌。掌面几乎平行于视线，二维掌宽被压扁。"""

    points = list(entry_palm_points())
    axis = points[9].x
    return tuple(
        Landmark(axis + (point.x - axis) * 0.12, point.y, point.z)
        for point in points
    )


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


class ArrivalClock:
    """可控的"到达墙钟"，用于替代 ``handmouse.gestures`` 里的 ``monotonic_ns``。"""

    def __init__(self, start_ns: int = 0) -> None:
        self.now_ns = start_ns

    def __call__(self) -> int:
        return self.now_ns

    def set_ns(self, value: int) -> None:
        self.now_ns = value

    def advance_ms(self, ms: float) -> int:
        self.now_ns += int(ms * 1_000_000)
        return self.now_ns


@pytest.fixture(autouse=True)
def arrival_clock(monkeypatch) -> ArrivalClock:
    """把 GestureEngine 的到达墙钟换成可控时钟，默认冻结在 0。

    ``last_hand_ns`` 记的是"观测到达控制线程的墙钟时刻"，测试里必须与 ``tick()``
    传入的墙钟参数同源，否则 elapsed 会落成负值或跨域。冻结在 0 时 ``tick(BASE + Δ)``
    的 elapsed 恰为 Δ，既有用例的语义不变；需要区分"采集时刻 vs 到达时刻"的用例
    再显式驱动这个时钟。
    """

    from handmouse import gestures as gestures_module

    clock = ArrivalClock()
    monkeypatch.setattr(gestures_module, "monotonic_ns", clock)
    return clock

