"""手部姿态分类：从 21 个关键点判断手指伸直/弯曲。

为什么需要这个
--------------
之前只要画面里出现手就会接管光标，于是抬手、放下、路过的动作全成了误触。
加一道姿态门槛之后，只有**食指单独伸出**时才进入操控，误触会少一个数量级；
同时也把"翻页"和"操控"拆成了两个互斥动作，不必再用热键切换模式。

三种姿态
--------
* ``point``：只伸食指 → 操控光标
* ``open``：五指张开 → 左右挥动翻页
* ``fist`` / ``other``：空闲，不做任何事

判断方法
--------
不用手指关节角度（容易受手掌旋转影响），改比**"指尖到指根的距离"是否显著
大于"指节到指根的距离"**。手指伸直时这段距离接近指长，弯曲时会缩到指节的
长度附近，比值稳定在 1.3 以上/以下，对掌面朝向不敏感。

拇指是例外：它折叠方向与其余四指近乎垂直，用同一判据会一直判成"伸着"。
所以拇指改用"指尖到小指根的距离"——收拢时这个距离会明显缩短。
"""

from __future__ import annotations

from math import hypot

#: 每根手指的 (指根, 指节, 指尖) 索引。
FINGER_JOINTS: dict[str, tuple[int, int, int]] = {
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}

#: 食指以外、参与"握拳"判断的三根手指。
OTHER_FINGERS = ("middle", "ring", "pinky")

#: 伸直判据：指尖到指根 / 指节到指根 超过这个比值即认为伸直。
#: 1.20 是实测定下来的：1.35 太严，手指自然放松时食指略弯就会被判成
#: "没伸"，用户感觉"识别不到了"。
EXTEND_RATIO = 1.20

#: 中指容差。食指伸的时候中指常会跟着抬一点，若严格要求它完全弯下，
#: 大量正常指向动作会被判成 other 而失效。低于这个比值仍按弯曲处理。
MIDDLE_TOLERANCE = 1.62

#: 拇指判据：指尖到小指根 相对 拇指根到小指根 的比值。
THUMB_RATIO = 1.15

WRIST = 0


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def finger_ratio(points: list[tuple[float, float]], finger: str) -> float:
    """返回"指尖到指根 / 指节到指根"的比值。

    伸直时接近 2 以上，完全弯曲时接近 0.5。用连续值而不是布尔，是为了让
    ``classify_pose`` 能做"中度弯曲也算弯"这类容差判断。
    """

    mcp, pip, tip = FINGER_JOINTS[finger]
    base = _distance(points[pip], points[mcp])
    if base <= 1e-6:
        return 0.0
    return _distance(points[tip], points[mcp]) / base


def is_extended(points: list[tuple[float, float]], finger: str) -> bool:
    """判断某根手指是否伸直。``points`` 为归一化坐标的 21 个关键点。"""

    if finger == "thumb":
        return _distance(points[4], points[17]) > (
            _distance(points[2], points[17]) * THUMB_RATIO
        )
    return finger_ratio(points, finger) >= EXTEND_RATIO


def classify_pose(points: list[tuple[float, float]]) -> str:
    """返回 ``point`` / ``open`` / ``fist`` / ``other``。

    拇指不参与分类：它收着还是张开，对"是不是在指东西"影响很小，
    但误判会把整只手的状态带偏。
    """

    if len(points) < 21:
        return "other"

    index_ratio = finger_ratio(points, "index")
    middle_ratio = finger_ratio(points, "middle")
    ring_up = is_extended(points, "ring")
    pinky_up = is_extended(points, "pinky")

    index_up = index_ratio >= EXTEND_RATIO
    middle_up = middle_ratio >= EXTEND_RATIO
    both_curled = not ring_up and not pinky_up

    if index_up and both_curled and not middle_up:
        return "point"
    # 中指跟着微抬（常见于自然指向）仍算操控，否则用户会觉得"识别不到"。
    if index_up and both_curled and middle_ratio < MIDDLE_TOLERANCE:
        return "point"
    if index_up and middle_up and ring_up and pinky_up:
        return "open"
    if not index_up and not middle_up and not ring_up and not pinky_up:
        return "fist"
    return "other"


def extended_names(points: list[tuple[float, float]]) -> list[str]:
    """调试用：列出所有判定为伸直的手指。"""

    if len(points) < 21:
        return []
    names = ["thumb", *OTHER_FINGERS, "index"]
    return [name for name in names if is_extended(points, name)]
