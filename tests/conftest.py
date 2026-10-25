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


def swipe_points() -> tuple[Landmark, ...]:
    """双指滑动姿势：食指+中指都竖直伸开，无名指与小指收拢。

    在 ``pointer_points`` 基础上把中指伸直（MCP→PIP→DIP→TIP 竖直向上），
    与"只伸食指"的唯一区别就是中指这一根——正好用来测模型/几何判据能否
    把这两类分开。
    """

    points = list(pointer_points())
    points[10] = Landmark(0.50, 0.45)
    points[11] = Landmark(0.50, 0.33)
    points[12] = Landmark(0.50, 0.20)
    return tuple(points)


def left_pinch_points() -> tuple[Landmark, ...]:
    points = list(pointer_points())
    points[4] = points[8]
    return tuple(points)


def fist_like_left_pinch_points() -> tuple[Landmark, ...]:
    """录像中的侧向左键捏合：其余手指收拢，二维投影近似握拳。"""

    points = list(fist_points())
    points[4] = points[8]
    return tuple(points)


def frontal_two_finger_points() -> tuple[Landmark, ...]:
    """正对镜头（掌面平行于成像面）的双指姿势——用户复报的失灵场景。

    与 ``swipe_points``（侧手投影）的关键区别是**掌宽没有再被压扁**：
    本夹具 掌宽(5→17) ÷ 手长(0→9) = 1.043，接近正手的真实比例 ≈1.0；
    而侧手夹具只有 0.55 左右。

    这正是旧判据 ``|MCP→tip| ÷ 掌宽 >= 1.3`` 的致命处：它的分母里含着一个
    会被透视压缩的掌宽，侧手时掌宽只剩一半、比值虚高到 1.3 以上；正手时
    掌宽是满宽，同一根伸直的手指算出来只有

    * 食指 1.150、中指 1.227 —— 双双卡在 1.3 以下

    于是"手指明明伸直了，双指却永远不触发"。用户说的"手移到某个高度以下
    就失灵"，本质是手腕角度变了、掌宽投影变宽，而不是位置本身。
    """

    values = [
        (0.50, 0.85),  # 0 wrist
        (0.44, 0.80), (0.38, 0.74), (0.33, 0.70), (0.30, 0.66),   # thumb
        (0.40, 0.62), (0.40, 0.50), (0.40, 0.40), (0.40, 0.32),   # index
        (0.50, 0.60), (0.50, 0.47), (0.50, 0.36), (0.50, 0.28),   # middle
        (0.56, 0.61), (0.57, 0.53), (0.565, 0.60), (0.56, 0.64),  # ring（收拢）
        (0.66, 0.64), (0.67, 0.56), (0.665, 0.62), (0.66, 0.66),  # pinky（收拢）
    ]
    return tuple(Landmark(x, y) for x, y in values)


def right_pinch_points() -> tuple[Landmark, ...]:
    points = list(pointer_points())
    # 中指主动伸直（MCP→PIP→DIP→tip 一条直线向上），再让拇指贴上去。
    # 只伸食指时中指是弯的，那种情况不应触发右键捏合。
    points[10] = Landmark(0.50, 0.45)
    points[11] = Landmark(0.50, 0.33)
    points[12] = Landmark(0.50, 0.20)
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


def thumb_open_pointer_points() -> tuple[Landmark, ...]:
    """停住姿势：只伸食指、其余三指握拳，但**拇指主动张开**远离食指。

    与 ``pointer_points`` 的食指状态完全一致，唯一区别是拇指尖(4)与食指掌指
    关节(5)的距离——即 ``_thumb_spread_value``：

    * ``pointer_points``             → 约 0.86（默认合成手本来就偏外展）
    * 本夹具                         → 约 1.09（对应用户实测「停住」1.1~1.4）

    注意它**不是**五指张开：中指/无名指/小指仍然弯曲，因此不得进入滚轮模式。
    """

    points = list(pointer_points())
    points[4] = Landmark(0.20, 0.45)
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
    """进入门姿势：五指张开 + 掌心朝镜（模型坐标里的真实左手掌）。

    坐标系约定按 2026-10-16 sign-ref 三轮回算修正：摄像头帧先 ``cv2.flip``
    再送 MediaPipe，MediaPipe 按输入像素的**表观外观**判 handedness，不做镜像
    补偿。本模块 ``observation()`` 固定标签 ``"Right"``，在 ``mirror=True`` 下
    它对应**真实左手**。

    真实左手掌心（模型坐标，实测中位）掌轴 0→5 指向画面右上、掌宽轴 5→17 指向
    左下。三点 0-5-17 的二维有向面积 ``ax*by-ay*bx`` 因此为负，与
    ``mirror=True`` + 标签 ``Right`` 推出的期望符号(-1) 一致，才通过朝向判定。
    实测四姿势（sign-ref 三轮回算）：右手掌心 nz>0 / 右手背 nz<0 /
    左手掌心 nz<0 / 左手背 nz>0。

    旧夹具（直接复用右手取向的 ``open_palm_points``）在旧观测量 5→17 × 5→4 下
    恰好符号正确，但换成 0-5-17 三角朝向后符号相反、会被判成"手背"——这是观测量
    更换后必须同步修正的夹具，不是放宽判定。
    """

    points: list[Landmark] = [Landmark(0.5, 0.5, 0.0)] * 21
    coords = {
        0: (0.50, 0.90),                                             # wrist
        1: (0.55, 0.78), 2: (0.62, 0.72), 3: (0.68, 0.66), 4: (0.72, 0.60),   # thumb
        5: (0.62, 0.62), 6: (0.63, 0.50), 7: (0.64, 0.40), 8: (0.65, 0.30),   # index
        9: (0.54, 0.60), 10: (0.54, 0.48), 11: (0.54, 0.37), 12: (0.54, 0.26),  # middle
        13: (0.46, 0.61), 14: (0.46, 0.49), 15: (0.46, 0.38), 16: (0.46, 0.29),  # ring
        17: (0.38, 0.64), 18: (0.38, 0.52), 19: (0.38, 0.42), 20: (0.38, 0.34),  # pinky
    }
    for idx, (x, y) in coords.items():
        points[idx] = Landmark(x, y, 0.0)
    # 0→5=(+0.12,-0.28)、5→17=(-0.24,+0.02) ⇒ nz = -0.0648 < 0，与期望符号一致。
    return tuple(points)


def entry_back_points() -> tuple[Landmark, ...]:
    """误进防护：同一只手手背朝镜（沿画面水平轴镜像 ⇒ 0-5-17 三角朝向翻号）。

    水平镜像把 nz 的符号取反（0→5 的 x 分量与 5→17 的 x 分量同时变号），
    因此手背的 nz>0，与期望符号(-1) 相反、被拒。
    """

    points = entry_palm_points()
    center_x = sum(points[index].x for index in (0, 5, 9, 13, 17)) / 5
    return tuple(Landmark(2 * center_x - point.x, point.y, point.z) for point in points)


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
    world: tuple[Landmark, ...] = (),
    aspect: float = 4.0 / 3.0,
) -> HandObservation:
    return HandObservation(
        landmarks=points,
        handedness="Right",
        timestamp_ns=timestamp_ns,
        inference_ms=12.0,
        handedness_score=0.98,
        world_landmarks=world,
        image_aspect=aspect,
    )


#: 米制 3D 手（原点在手部几何中心，单位米）。三个骨长用真实比例：
#: 近节 0.045 / 中节 0.028 / 末节 0.022 —— 完全伸直时「弦长 ÷ 骨长和」= 1.0。
_WORLD_KNUCKLES_Y = -0.020
_BONE_PROX, _BONE_MID, _BONE_DIST = 0.045, 0.028, 0.022


def world_hand(
    *,
    index_extended: bool = True,
    middle_extended: bool = True,
    ring_extended: bool = False,
    pinky_extended: bool = False,
) -> tuple[Landmark, ...]:
    """构造米制 3D 手部关键点，可逐指指定"伸直 / 蜷起"。

    ``world_landmarks`` 是 MediaPipe 的**米制 3D 骨架**（原点在手心），
    判据在它上面量「这根手指直不直」才是量手指本体：没有画面宽高比失真，
    也没有透视压缩。伸直时 ``|MCP→TIP| ÷ 骨长和`` ≈ 1.0，蜷起时约 0.5。

    布局：y 轴大致沿手指方向（掌心 → 指尖为正），x 轴沿掌指关节排，
    四指分别落在 x = -0.024 / -0.008 / 0.008 / 0.024 上。
    """

    points: list[Landmark] = [Landmark(0.0, 0.0, 0.0)] * 21
    points[0] = Landmark(0.0, -0.085, 0.0)  # wrist
    xs = {"index": -0.024, "middle": -0.008, "ring": 0.008, "pinky": 0.024}
    y = _WORLD_KNUCKLES_Y
    for name, (mcp, pip, dip, tip) in (
        ("index", (5, 6, 7, 8)),
        ("middle", (9, 10, 11, 12)),
        ("ring", (13, 14, 15, 16)),
        ("pinky", (17, 18, 19, 20)),
    ):
        x = xs[name]
        extended = {
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended,
        }[name]
        points[mcp] = Landmark(x, y, 0.0)
        if extended:
            points[pip] = Landmark(x, y + _BONE_PROX, 0.0)
            points[dip] = Landmark(x, y + _BONE_PROX + _BONE_MID, 0.0)
            points[tip] = Landmark(x, y + _BONE_PROX + _BONE_MID + _BONE_DIST, 0.0)
        else:
            # 蜷起：中节抬起、末节折回掌心，弦长被压到骨长和的一半左右。
            points[pip] = Landmark(x, y + _BONE_PROX, 0.0)
            points[dip] = Landmark(x, y + _BONE_PROX + 0.012, 0.0)
            points[tip] = Landmark(x, y + _BONE_PROX - 0.004, 0.0)
    # 拇指：只求几何合理，判据不用它。
    points[1] = Landmark(-0.030, -0.070, 0.0)
    points[2] = Landmark(-0.042, -0.050, 0.0)
    points[3] = Landmark(-0.052, -0.030, 0.0)
    points[4] = Landmark(-0.058, -0.012, 0.0)
    return tuple(points)


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

