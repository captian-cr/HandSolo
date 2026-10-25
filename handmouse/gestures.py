"""基于关键点几何关系的手势状态机。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from enum import Enum
from math import acos, degrees, exp, hypot, sqrt
from time import monotonic_ns

from handmouse.config import AppConfig
from handmouse.contracts import (
    GestureType,
    HandObservation,
    Landmark,
    MouseCommand,
    MouseCommandType,
)
from handmouse.filters import OneEuroPointerFilter, map_activity_region_edges
from handmouse.gesture_data import (
    GestureDataset,
    default_dataset_path,
    default_pending_path,
    normalize,
)
from handmouse.gesture_model import GestureClassifier, suggest_reject_distance


class ControlState(str, Enum):
    """控制状态机内部状态。"""

    IDLE = "idle"
    MOVING = "moving"
    PINCH_PENDING = "pinch_pending"
    DRAGGING = "dragging"
    SCROLLING = "scrolling"
    PAUSED = "paused"
    LOST = "lost"


def _distance(a: Landmark, b: Landmark) -> float:
    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


#: 本文件所有几何阈值的**标定宽高比**。改这个数字等于让所有阈值失准。
_REFERENCE_ASPECT = 4.0 / 3.0


def _aspect_scale(aspect: float) -> float:
    """把「画面宽高比」换算成 x 方向的还原倍率。

    归一化坐标里 x 除的是**画面宽**、y 除的是**画面高**，所以同一个物理长度
    横向量出来会比纵向短 ``aspect`` 倍（16:9 时短 25%）。凡是"两点距离 ÷
    掌宽"这种比值判据，分子分母受横向压缩影响的幅度不同，比值就会随摄像头
    宽高比漂移。

    做法是把 x 乘上 ``aspect ÷ (4/3)``，还原到标定尺度：4:3 时倍率恰好 1.0
    （与历史行为完全一致，所有阈值不动），16:9 时乘 1.333，把缩水的那 25%
    补回来。这是**纯几何补偿**，不含任何拟合参数。
    """

    try:
        value = float(aspect)
    except (TypeError, ValueError):
        return 1.0
    if not (value > 0.0):
        return 1.0
    # 现实中摄像头从 1:1 到 21:9 都有；超范围说明数据坏了，宁可不动。
    return max(0.5, min(2.4, value / _REFERENCE_ASPECT))


def _isotropic(points: tuple[Landmark, ...], k: float) -> tuple[Landmark, ...]:
    """把归一化坐标按 ``k`` 横向还原，得到"各向同性"的度量空间。

    k == 1.0 时**原样返回同一个元组**（不加分配、不改语义），所以 4:3 的
    默认设备上这条路径等价于不存在。
    """

    if not points or k == 1.0:
        return points
    return tuple(Landmark(point.x * k, point.y, point.z) for point in points)


def _distance_2d(a: Landmark, b: Landmark) -> float:
    """手势阈值使用图像平面距离，避开单目深度坐标的帧间噪声。"""

    return hypot(a.x - b.x, a.y - b.y)


def _distance_3d(a: Landmark, b: Landmark) -> float:
    """米制 3D 世界坐标下的距离。

    ⚠️ 只能用于 ``HandObservation.world_landmarks``。归一化图像坐标的 z 是
    网络猜的相对深度，直接开三维会把透视问题变成噪声问题，比二维更糟。
    """

    return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _angle(a: Landmark, vertex: Landmark, c: Landmark) -> float:
    """返回以 ``vertex`` 为顶点的图像平面夹角。

    单目模型的相对深度在侧手和弱光下更容易波动，屈伸分类只依赖二维
    骨架即可，因此不让z噪声把伸直的手指误判为弯曲。
    """

    first = (a.x - vertex.x, a.y - vertex.y)
    second = (c.x - vertex.x, c.y - vertex.y)
    first_norm = sqrt(sum(value * value for value in first))
    second_norm = sqrt(sum(value * value for value in second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return 0.0
    cosine = sum(a_value * b_value for a_value, b_value in zip(first, second))
    cosine /= first_norm * second_norm
    cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(cosine))


def _palm_width(points: tuple[Landmark, ...]) -> float:
    return max(_distance_2d(points[5], points[17]), 1e-4)


def _palm_center(points: tuple[Landmark, ...]) -> Landmark:
    indices = (0, 5, 9, 13, 17)
    return Landmark(
        x=sum(points[index].x for index in indices) / len(indices),
        y=sum(points[index].y for index in indices) / len(indices),
        z=sum(points[index].z for index in indices) / len(indices),
    )


def _palm_depth_tilt(points: tuple[Landmark, ...]) -> float:
    """返回腕部到中指掌指关节的归一化深度差。

    掌心前后翻转时，掌心在画面中的中心可以几乎不动，但9号点相对
    手腕的深度会稳定改变。以掌宽归一化后可兼容不同距离和手型。
    """

    return (points[9].z - points[0].z) / _palm_width(points)


def _finger_extensions(points: tuple[Landmark, ...]) -> tuple[bool, bool, bool, bool]:
    fingers = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    return tuple(_angle(points[mcp], points[pip], points[tip]) >= 135.0 for mcp, pip, tip in fingers)  # type: ignore[return-value]


def _is_pointer(points: tuple[Landmark, ...]) -> bool:
    # 张掌和捏合已在状态机中优先处理。此处只要求食指伸展，允许其余
    # 手指自然半弯，避免用户必须维持僵硬的“标准手势”。
    index, _middle, _ring, _pinky = _finger_extensions(points)
    return index


def _is_open_palm(points: tuple[Landmark, ...]) -> bool:
    """张掌：食指、中指、无名指都伸直，小指可自然弯曲。

    纯角度判据（``>=135°``）在**手掌朝下/侧向**时会失效：手指虽然蜷着，
    但投影到画面上两个指节几乎共线，角度仍然很大——用户实测"只伸食指"却被
    判成张掌，就是这个原因。滚轮模式因此整体移除，本函数现在只用于右键
    预备（``right_click_enabled``），不再触发任何滚轮/翻腕行为。

    所以在角度之外再要求**中指指尖离指根足够远**（距离比），双判据兜底。
    """

    index, middle, ring, _pinky = _finger_extensions(points)
    if not (index and middle and ring):
        return False
    base = _distance_2d(points[10], points[9])
    if base <= 1e-6:
        return False
    return _distance_2d(points[12], points[9]) / base >= 1.30


def _is_fist(points: tuple[Landmark, ...]) -> bool:
    if any(_finger_extensions(points)):
        return False
    center = _palm_center(points)
    width = _palm_width(points)
    tips = (8, 12, 16, 20)
    mean_distance = sum(_distance(points[index], center) for index in tips) / len(tips)
    return mean_distance / width < 1.25


# ---- 双指滑动姿势判据（标度不变）------------------------------------------
#
# 🔴🔴 2026-10-22 第五次实机复报后**整体换掉判据维度**，原因写在这里：
#
# 旧判据是 ``|MCP→tip| ÷ 掌宽 >= 1.3``。它只在**侧手**成立：
# 掌宽（5 号点到 17 号点）正对镜头时是满宽（约 8cm），伸直的食指比例只有
# ~1.0；把手侧过来，掌宽被透视压扁到 4~5cm，同一个手势的比例才涨到 1.9。
# 也就是说这条判据**本质是"侧手专用"**——用户正对镜头持双指时永远不触发。
# 用户反复报"手指到这个位置以下双指就失灵"，根因就在这里。
#
# 换尺子（掌宽 → 手长）**也救不了**，因为他 3378 帧真实样本里双指类内
# ``|MCP→tip| ÷ 手长`` 的 p5~p95 是 0.41~1.51，跨度 3.7 倍（``scripts/
# audit_swipe_geom.py``）。**任何"绝对长度 ÷ 某个基准"的阈值都只能覆盖一部分姿态。**
#
# 所以彻底换维度：不比"多长"，只比"谁比谁长"。以四指指尖到掌心距离
# ``t_i/t_m/t_r/t_p`` 为基础，除以本帧最大值 ``peak = max(...)``：
#   * 只伸食指（move）    ``t_m/peak ≈ 0.35``
#   * 五指张开（open）    四指都 ≈ 0.9~1.0
#   * 双指（swipe）       ``t_i/peak``、``t_m/peak`` ≈ 1，而无名指/小指明显小
# 分子分母共享**同一个**投影缩放系数，姿态变化直接约掉——这才是旋转/透视
# 不变的量。``peak`` 自己再除以手长，用来排除"手是握着的"（所有指尖都缩在
# 掌心附近时，上面那组比值退化成噪声：握拳类 p50 只有 0.45）。
#
# 实测（他 3378 帧真实样本 + 按 ``captured_ns`` 还原的原始帧序，时序仿真，
# 见 ``scripts/audit_swipe_final.py``，判据直接 import 自本模块、不会漂移）：
#   旧判据 ÷掌宽：双指可用度 **36.3%**（单帧口径）／只伸食指误报 1.4%
#   现行判据：双指可用度 **84.9%**（确认 150ms + 持有 450ms）
#             ／只伸食指 0.0%／停住 0.0%／捏合 0.0%／五指张开 0.0%／握拳 0.0%
# 剩下 15% 里约 10.4% 是**真实的姿态模糊**（他的"双指"帧里有一批食指最长、
# 中指折着，几何上就是"只伸食指"），拒掉是对的；其余交给持有窗口吞掉。
# **不再压门限**去换召回——那会把"只伸食指"重新拖进误报，正是光标乱飞的来源。

#: 进入双指姿势：最长指尖到掌心 ÷ 手长。低于此值说明手是握着的，
#: 上面那组相对比值没有意义（握拳类 p50=0.45，双指 p25=1.16）。
_SWIPE_TIP_FLOOR = 0.80
#: 进入双指姿势：食指、中指尖到掌心 ÷ 本帧最长。两者都要接近 1。
_SWIPE_TIP_MAIN = 0.70
#: 进入双指姿势：无名指、小指尖到掌心 ÷ 本帧最长，必须明显收拢。
#:
#: 2026-10-22 第六次修订：0.65 → **0.80**。原因是把"最差的那几次录制"逐帧拆开
#: 看过（``collect-20261022-154435`` 等）：他失败帧的无名指比值中位是 **0.871**、
#: 小指 **0.853**——**只是自然半弯，并不是收拢**。0.65 把这类真实姿势全拒了，
#: 它们占了非"移动姿势"失败帧的 60~69%，是当时最大的单一口径损失。
#:
#: 实测（3378 帧 + 时序仿真，确认 150ms、持有 450ms）：
#:   0.65 → 可用度 80.7%；0.70 → 81.2%；0.75 → 82.2%；**0.80 → 84.9%**
#:   而"只伸食指 / 停住 / 捏合 / 五指张开 / 握拳"五个类的误报**全部仍为 0.0%**
#: 放宽到 0.85 才开始出事（五指张开误报 10.5% 静态、0.5~1.4% 动态），所以
#: 0.80 是这条前沿上"最后一段免费的距离"。
#:
#: ⚠️ 试过改成相对判据 ``max(r,p) <= K * min(i,m)``（想用"两根主指比收拢指长"
#: 来摆脱绝对阈值），**实测更差**：K=0.90 时五指张开误报 16.1%，因为张开时
#: 四指长度本来就不齐。绝对阈值在这里反而更稳，别再改回去。
#: ⚠️ 0.80 之后剩下的缺口是真实姿态模糊：他"双指"里约 10.4% 的帧其实是指尖
#: 最长、中指折着（``i/peak≈1.0 & m/peak<0.6``）——那几何上就是"只伸食指"，
#: 拒掉是对的，否则光标会乱飞。这部分只能靠持有窗口吞，不能靠压门限。
_SWIPE_TIP_CURL = 0.80
#: 持有（滚动中判据抖动）用的放宽门限：**只要手还是半张着**就按住不松手。
#:
#: 🔴 这里**不能再套进入判据的任何一条**（2026-10-22 实测踩过）：
#: 抖动之所以发生，恰恰是因为那几条同时不成立——它们高度相关。给持有也
#: 加 `tm>=0.55` 之类的形状闸，持有就永远不生效（实测双指可用度只有 65.0%，
#: 和不持有几乎没差别）。持有唯一需要的语义是"他还在做双指这件事"，用
#: ``peak``（四指最长指尖到掌心 ÷ 手长）一个粗判据就够：低于 0.45 说明手
#: 已经握起来了。放行时长由 ``swipe_hold_ms`` 兜底。
_SWIPE_HOLD_FLOOR = 0.45
#: 模型可放宽的"收拢"上限（见下）。必须严格大于 ``_SWIPE_TIP_CURL``，
#: 否则这条放宽等于不存在。0.90 对应"无名指基本伸直"，仍由核心闸兜底。
_SWIPE_MODEL_CURL = 0.90
#: 中指关节角下限（度）。**与上面那组长度比是正交维度**，不能省：
#: 长度比值量的是"指尖离掌心多远"，而**折成 90° 的中指指尖照样很远**
#: （2026-10-22 14:17 事故：只伸食指、中指 PIP 折 90° 往侧面翘，
#: ``tm/peak`` 高达 0.889，新判据曾把它放行 → 光标冻结 + 幻影滚动）。
#: 弯曲是真实的三维特征，投影压不扁它，所以用关节夹角兜这一刀。
#: 取 110° 而不是旧的 135°：他真实双指帧的关节角 p10 只有 78.7°、
#: p25 是 143.6°（单目在侧手/运动模糊下角度抖动很大），135° 会削掉
#: 3.8% 的双指可用度，110° 只削 0.5%（``scripts/audit_swipe_final.py``），
#: 同时仍能拒掉真正折起来的中指。
_SWIPE_MIDDLE_ANGLE_MIN = 110.0
#: 单帧通过就进入，会把孤立的单帧误报变成一次真的滚动；而纯按"连续 N 帧"
#: 计数在这里会死锁（第一帧未确认 → 走持有失败 → 重置计数 → 第一帧）。
_SWIPE_CONFIRM_MS = 150.0
#: 捏合锚定需要连续成立的帧数（见 `_pinch_approach_detected`）。
#: 单帧触发会让"指向前方"被随机冻住 1 秒（实测移动时 30.6% 时间被冻）。
_PINCH_ANCHOR_FRAMES = 3


def _two_finger_diag(points: tuple[Landmark, ...]) -> tuple[float, float, float, float, float, float]:
    """返回双指判据的 6 个诊断量：``(peak, i, m, r, p, a_m)``。

    ``peak`` 是本帧四指指尖到掌心距离的最大值 ÷ 手长（腕 0 号点 → 中指
    掌指关节 9 号点），``i/m/r/p`` 是各指尖到掌心距离 ÷ ``peak``，
    ``a_m`` 是中指 MCP-PIP-TIP 的关节夹角（度）。
    """

    center = _palm_center(points)
    length = _distance_2d(points[0], points[9])
    if length < 1e-4:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    tips = [_distance_2d(points[index], center) / length for index in (8, 12, 16, 20)]
    peak = max(tips)
    if peak < 1e-4:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    angle = _angle(points[9], points[10], points[12])
    return (peak, tips[0] / peak, tips[1] / peak, tips[2] / peak, tips[3] / peak, angle)


def _two_finger_main_ok(diag: tuple[float, ...]) -> bool:
    """核心闸：食指、中指都"量出来够长"**且中指没折起来**——模型也不能绕。"""

    peak, index, middle, _ring, _pinky, angle = diag
    return (
        peak >= _SWIPE_TIP_FLOOR
        and index >= _SWIPE_TIP_MAIN
        and middle >= _SWIPE_TIP_MAIN
        and angle >= _SWIPE_MIDDLE_ANGLE_MIN
    )


# ---- 并联的第二个尺子：逐指"伸出来了吗"（2026-10-22 用户提出）-------------
#
# 用户原话：「直接上视觉看我的手指，只要是中指和食指伸出来，那就是要划动，
# 不管怎么变角度……只要能识别出中指和食指就行了」。
#
# 动机完全正确：上面 ``_is_two_finger`` 里有一个 ``÷ peak`` 的**跨手指**比例，
# 换个手腕角度各指的投影缩放就不一样，阈值永远调不完。而下面这套两个量
# **只在单指内部归一化**，不含任何跨手指比较：
#
#   * ``s``（伸直度）= 〚MCP→TIP〛 ÷ (〚MCP→PIP〛+〚PIP→DIP〛+〚DIP→TIP〛)
#     只量 PIP/DIP 有没有弯，伸直 ≈ 1.0，对均匀投影缩放**严格不变**。
#   * ``a``（指根角）= 掌指关节排中心→MCP 的方向 与 MCP→PIP 的夹角。
#     补上 ``s`` 抓不到的"MCP 关节把整根手指折向掌心"——那种情况三节自身
#     是直的（``s``≈1）却趴在掌心里。
#
# 🔴 **但它不能单独当判据**（实测，``scripts/audit_finger_extended.py``）：
# 3378 帧零误报下只有 **73.8%**，而现行跨指指尖距离判据是 84.9%；
# 逐会话均值 70.5% vs 83.6%，**最差的一次录制掉到 8.3%**（现行 65.1%）。
# 原因在 ``collect-20261022-161748``：那里他失败帧的中指伸直度是 **0.991**
# （很直）但指根角只有 57°——整根手指趴在掌心里，"看它直不直"恰好判错，
# 而"指尖离掌心多远"恰好判对。**两个尺子的盲区正好互补。**
#
# 所以只把它作为**并联的补充路径**（``_is_two_finger`` ∪ ``_finger_extended``）：
# 实测 84.9% → **85.9%**，五个非双指类误报仍全 0.0%，
# 且它单独能救回现行判据漏掉的会话（如 ``collect-20261022-161133``
# 71.2% → 87.5%）。**不要用它替换主判据。**
_FINGER_STRAIGHT_MIN = 0.90
#: 指根角下限（度）。取 75 而不是最优点 55：留出现测过的安全边距
#: （55~95 之间可用度只差 1%，但 55 太靠近"趴向掌心"的那一侧）。
_FINGER_ROOT_MIN = 75.0
#: 无名指/小指的"没伸出来"上限（用伸直度）。五指张开时四指都 ≈1.00，
#: 取 0.90 与它们拉开距离；放宽到 0.95 会让"停住"类开始误报 7.9%。
_FINGER_CURL_STRAIGHT_MAX = 0.90
#: 并联路径里中指尖到掌心 ÷ 本帧最长的**下限**。挡的是"手指指向镜头被透视
#: 压短"这种情况：那种姿势三节仍然共线（伸直度 ≈0.98），只靠伸直度会把它
#: 当成"伸出来了"，可它的指尖其实就在掌心附近。实测代价只有 0.1 个百分点
#: （85.9% → 85.8%），是白捡的正交保险。
_FINGER_MIN_REACH = 0.50

#: 各指的关键点下标 ``(MCP, PIP, DIP, TIP)``。
_FINGER_CHAINS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _finger_feats(points: tuple[Landmark, ...]) -> dict[str, tuple[float, float]]:
    """返回 ``{finger: (伸直度 s, 指根角 a)}``，两个量都只用到该指自己的点。"""

    knuckles = (5, 9, 13, 17)
    cx = sum(points[i].x for i in knuckles) / len(knuckles)
    cy = sum(points[i].y for i in knuckles) / len(knuckles)

    out: dict[str, tuple[float, float]] = {}
    for name, (mcp, pip, dip, tip) in _FINGER_CHAINS.items():
        path = (
            _distance_2d(points[mcp], points[pip])
            + _distance_2d(points[pip], points[dip])
            + _distance_2d(points[dip], points[tip])
        )
        straight = _distance_2d(points[mcp], points[tip]) / path if path > 1e-6 else 0.0
        # 指根角要一个"手掌方向"作参照：用四个掌指关节的中心，比用腕点稳
        # （腕点会随腕部旋转整体摆动）。
        u1 = (cx - points[mcp].x, cy - points[mcp].y)
        u2 = (points[pip].x - points[mcp].x, points[pip].y - points[mcp].y)
        n1 = (u1[0] ** 2 + u1[1] ** 2) ** 0.5 or 1e-9
        n2 = (u2[0] ** 2 + u2[1] ** 2) ** 0.5 or 1e-9
        cosine = (u1[0] * u2[0] + u1[1] * u2[1]) / (n1 * n2)
        out[name] = (straight, _degrees_of(cosine))
    return out


def _degrees_of(cosine: float) -> float:
    """``acos`` 的薄包装，单独拎出来便于测试。"""

    import math

    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _finger_extended(feat: tuple[float, float]) -> bool:
    straight, root = feat
    return straight >= _FINGER_STRAIGHT_MIN and root >= _FINGER_ROOT_MIN


def _two_finger_by_fingers(points: tuple[Landmark, ...]) -> bool:
    """并联路径：食指伸出来 且 中指伸出来 且（无名指 或 小指 没伸出来）。

    语义就是用户的原话——"只要能识别出中指和食指伸出来"。

    ⚠️ 无名指/小指的"没伸出来"**只看伸直度，不看指根角**：指根角用的参照是
    "掌指关节排中心 → MCP"，而外侧两指本来就呈扇形外张，这个方向天然与手指
    方向成大角。拿它判"伸没伸出"会把真正张开的五指判成"没伸出"——
    实测五指张开待机被误吃 **44.4%**（`scripts/audit_finger_extended.py`）。
    伸直度没有这个偏向（五指张开时四指都 ≈1.00），所以外侧两指只用伸直度。
    """

    feats = _finger_feats(points)
    if not (_finger_extended(feats["index"]) and _finger_extended(feats["middle"])):
        return False
    # 反过来挡一次"指向镜头的短指"：三节共线但指尖还在掌心附近。
    if _two_finger_diag(points)[2] < _FINGER_MIN_REACH:
        return False
    ring_straight = feats["ring"][0]
    pinky_straight = feats["pinky"][0]
    return (
        ring_straight <= _FINGER_CURL_STRAIGHT_MAX
        or pinky_straight <= _FINGER_CURL_STRAIGHT_MAX
    )


def _is_two_finger(diag: tuple[float, ...]) -> bool:
    """主判据：核心闸 + 无名指/小指收拢。"""

    _peak, _index, _middle, ring, pinky, _angle = diag
    return _two_finger_main_ok(diag) and ring <= _SWIPE_TIP_CURL and pinky <= _SWIPE_TIP_CURL


def _two_finger_hit(
    points: tuple[Landmark, ...],
    diag: tuple[float, ...],
    world: tuple[Landmark, ...] = (),
) -> bool:
    """双指进入判据的**唯一入口**。

    🔴 有 3D 就**只听 3D**：``world_landmarks`` 齐全时它说了算，2D 判据
    完全不参与。理由见 ``_FINGER3D_STRAIGHT_MIN`` 上方——他自己的数据上，
    2D 距离判据的"只伸食指"误触是 13.2%、"五指张开"误触 12.1%，而 3D 是
    3.3% 与 0%。把 2D 并进来（OR）会把这些误触原封不动带回来（实测最差
    19.6%），所以**不是并联，是降级兜底**。

    没有 3D（回放、测试、局部跟踪降级、旧模型）时才退回 2D 的两条路。

    引擎和 ``scripts/audit_*.py`` 都必须调这个函数，否则两边口径会漂移。
    """

    if len(world) == 21:
        return _two_finger_by_world(world)
    return _is_two_finger(diag) or _two_finger_by_fingers(points)


# ---- 3D 判据：用 MediaPipe 的米制世界坐标（2026-10-22 用户要求）-----------
#
# 用户原话：「你直接用摄像头识别我的中指食指手指本体，不要靠 2D 手势架」。
#
# ``hand_world_landmarks`` 是模型输出的**米制 3D 骨架**（原点在手部几何
# 中心）。在那里量"这根手指直不直"，才是在量手指本体，而不是量它在画面上
# 的投影——不受画面宽高比、透视压缩、手腕朝向影响。
#
# 判据与 2D 逐指版同构，但量在真三维里：
#   ``straight3d`` = 〚MCP→TIP〛 ÷ (三节骨长之和)
#
# 🔴 **3D 有权威性，不是"并联补充"**（2026-10-22 第二次修订，被实机纠正）。
# 第一版把它当补充、阈值还取了估的 0.97，结果他实机截图里手明明伸着两根，
# HUD 却显示 `指伸3D 0000`——因为他真实的伸直度只有 0.93~0.96。
# 现在 3D 齐全时**由 3D 说了算**，2D 只在没有 3D 时兜底。
# 依据（他自己 701 帧带 3D 的采集，`scripts/audit_2d_vs_3d_real.py`）：
#   2D 距离判据：双指 81.0%，但**只伸食指误触 13.2%、五指张开误触 12.1%**
#               （他实机报的"单食指被判成划动"就是这个）
#   3D 伸直判据：双指 76.0%，误触 只伸食指 3.3% / 其余 0%
#   → 3D 少 5 个点的召回，换来移动误触 -10 点、张开误触 -12 点。方向明确。
#
# ⚠️ 仍要知道的边界：世界坐标在**离镜头 0.5m 以内**会明显退化（文献一致
# 结论），他坐桌前正在边缘。所以 3D 缺失/退化时必须有 2D 兜底。
#
# 阈值是**在他自己数据上标的**，不是估的：
#   他真实双指帧 3D 伸直度 食指中位 0.962、中指 0.926 → 取 0.97 会全灭。
_FINGER3D_STRAIGHT_MIN = 0.86
#: 无名指/小指"确实蜷起来了"的上限。0.86 是关键档：放宽到 0.90 会让
#: **五指张开待机被误吃 6.5%**、到 0.93 时"停住"也被误吃 12.7%。
_FINGER3D_CURL_MAX = 0.86


def _finger3d_feats(world: tuple[Landmark, ...]) -> dict[str, float]:
    """从米制 3D 关键点算各指的伸直度（0~1，1 = 完全伸直）。"""

    out: dict[str, float] = {}
    for name, (mcp, pip, dip, tip) in _FINGER_CHAINS.items():
        path = (
            _distance_3d(world[mcp], world[pip])
            + _distance_3d(world[pip], world[dip])
            + _distance_3d(world[dip], world[tip])
        )
        out[name] = _distance_3d(world[mcp], world[tip]) / path if path > 1e-6 else 0.0
    return out


def _two_finger_by_world(world: tuple[Landmark, ...]) -> bool:
    """3D 判据：食指伸出来 且 中指伸出来 且（无名指或小指确实蜷着）。

    只在 ``world_landmarks`` 齐全（21 点）时可用；否则一律返回 False，
    由 2D 的几条路兜底。
    """

    if len(world) != 21:
        return False
    feats = _finger3d_feats(world)
    if not (
        feats["index"] >= _FINGER3D_STRAIGHT_MIN
        and feats["middle"] >= _FINGER3D_STRAIGHT_MIN
    ):
        return False
    return (
        feats["ring"] <= _FINGER3D_CURL_MAX
        or feats["pinky"] <= _FINGER3D_CURL_MAX
    )


def _is_two_finger_hold(diag: tuple[float, ...]) -> bool:
    """持有判据：**只要求手还是半张着的**，不重复进入判据的任何一条。

    抖动时进入判据的各条往往同时不成立（相关性极强），所以持有必须换一个
    正交的粗判据，否则等于没有持有（实测：套形状闸只有 65% 可用度，
    只判 ``peak`` 有 78.9%）。
    """

    return diag[0] >= _SWIPE_HOLD_FLOOR


#: 进入门阈值：拇指外展程度 = 拇指尖到小指掌指关节的距离 / 掌宽。
_ENTRY_THUMB_SPREAD_MIN = 0.9
#: 进入门阈值：掌面正对镜头程度 = 掌宽 / 腕到中指指尖的距离，用于排除侧掌。
_ENTRY_PALM_FRONTAL_MIN = 0.25

#: 五指张开的**投影长度**门槛：四指 |掌指关节→指尖| ÷ 掌宽 全部 ≥ 此值。
#:
#: 🔴 为什么必须有这一维（2026-10-22 用户实机截图 + 他的 5700 帧真实样本回放）：
#: 纯角度判据在侧手投影下会把**收拢的四指**判成伸直——MCP/PIP/TIP 三个投影点
#: 近似共线，夹角照样 ≥135°。用户自己样本上的假阳率
#: （``scripts/audit_gesture_conflicts.py``，判据 import 自本模块）：
#:   * move（只伸食指）   **9.8%**  → HUD 报「五指张开（待机）」、光标冻结
#:   * stop（食指+拇指张开）**40.3%** → 想做"停住"，反被抢进待机
#:   * swipe（双指）      13.7%  → 划到一半被判待机，光标"乱飞"
#: 前两条正是他截图里那行字的来源。
#:
#: 投影长度是**与角度正交的维度**：手指弯曲时 |MCP→tip| 必然缩短
#: （move 中指中位 0.91 vs open 1.90，差一倍），而掌宽的透视压扁同时作用于
#: 分子分母、不改变这个比值。复测（``scripts/audit_five_open.py``）：
#:   T=1.10 → move 误抓 7.9% / stop **0%** / swipe 5.4% / open 漏判 3.2%
#:   T=1.30 → move 3.3%，但 open 漏判 **36.3%**（小指投影天然短，p10 仅 1.21）→ 太狠
#: 所以取 1.10。残余 7.9% 是"手侧到极限、四指投影恰好都够长"的帧，几何上已
#: 无法再分——交给 ``_confirm_five_open`` 的时序确认兜底，**不再压阈值**。
_FIVE_OPEN_REACH_MIN = 1.10

#: 五指张开要**连续保持**多久才切进待机（进严出松）。
#: 单帧误报就切待机会把光标瞬冻。取 300ms 的依据（他 2026-10-21 采集会话
#: 中 9 个"只伸食指却被判张开"的帧，最长连续段 4 帧）：
#:   * 实机 FPS **15**（他截图里 15/15，每帧约 66ms）→ 4 帧 = **265ms**
#:   * 30fps 采集时同段仅 125ms
#: 所以必须按"最慢帧率"算：300ms 才能完整吃掉那段误报，200ms 会漏。
#: 与 ``entry_hold_ms``（默认 300）取同值，语义统一——"要做这个姿势就得
#: 保持 300ms"。反过来退出待机**不加延迟**，下一帧就恢复移动。
_FIVE_OPEN_CONFIRM_MS = 300.0


def _is_five_open(
    points: tuple[Landmark, ...],
    world: tuple[Landmark, ...] = (),
) -> bool:
    """五指张开：四指伸直，并且拇指明显外展。

    ``_is_open_palm`` 只要 3/4 指伸直（照顾日常滚轮姿势），进不了进入门。
    进入门是全局闸门，判据必须更严，否则"半握"也能把闸门顶开。

    🔴 2026-10-22：**有 3D 就只听 3D**。他 701 帧带 3D 的采集上实测：

    =====================  ======  =====  =====  =====  =====  =====
    判据                    swipe   move   stop   pinch  open   fist
    =====================  ======  =====  =====  =====  =====  =====
    2D 全判据（现行）           7.0%   1.7%   2.4%   0.0%  82.2%   0.9%
    **3D≥0.86 + 拇指闸**      0.0%   0.0%   3.2%   0.0%  **92.5%**
    =====================  ======  =====  =====  =====  =====  =====

    两个方向同时变好：召回 82.2% → 92.5%，而"双指被误判成待机"从 7.0% 归零。
    后者正是他"划着划着就跳去待机、光标冻住"的来源之一。

    没有 3D 时退回 2D，且仍然**两族判据同时成立**（缺一不可）：
      * 关节夹角 ≥135°（``_finger_extensions``）—— 判断"关节没弯"；
      * 四指 |MCP→tip| ÷ 掌宽 ≥ ``_FIVE_OPEN_REACH_MIN`` —— 判断"真的伸出来了"。
    只留角度：侧手投影把收拢的手判成张开（见上面实测）。只留长度：半弯但
    指节翘起的姿势会骗过它（swipe 判据里同一教训的注释，2026-10-22 事故）。

    ⚠️ 拇指闸（4→17 ÷ 掌宽 ≥0.9）**不具备区分力**，但两个版本都保留它：
    实测"食指+拇指张开"的 stop 姿势该值中位 **2.02**，比真五指张开（1.66）
    还大——它挡不住任何收拢姿势，真正拦住 stop 的是手指那一族判据。
    保留它只是不让历史行为发生无谓变化。
    """

    if len(world) == 21:
        feats = _finger3d_feats(world)
        if not all(
            feats[name] >= _FINGER3D_STRAIGHT_MIN
            for name in ("index", "middle", "ring", "pinky")
        ):
            return False
    else:
        if not all(_finger_extensions(points)):
            return False
        width = _palm_width(points)
        if width <= 1e-9:
            return False
        for mcp, tip in ((5, 8), (9, 12), (13, 16), (17, 20)):
            if _distance_2d(points[mcp], points[tip]) / width < _FIVE_OPEN_REACH_MIN:
                return False
    width = _palm_width(points)
    if width <= 1e-9:
        return False
    return (
        _distance_2d(points[4], points[17]) / width
        >= _ENTRY_THUMB_SPREAD_MIN
    )


#: 拇指张开度实测分布（用户 2026-10-19 提供 8 张截图）：
#:   收起（只伸食指、拇指贴手）: 0.60 / 0.71
#:   张开（拇指与食指分开）    : 0.86 / 0.90 / 1.12 / 1.22 / 1.24 / 1.44
#: 用户实际做"张开"时幅度不大，最低只到 0.86，所以进入阈值取 0.85（原 0.95
#: 太高，他的动作够不到）。退出取 0.72，仍高于"收起"的上限 0.71。
#:
#: 🔴 2026-10-21 实测结论：**这套阈值在"手指朝下"时完全失效，该功能已按用户
#: 要求关闭（config.json ``thumb_stop_enabled=false``）**。
#:
#: 实测 628 帧、下半部 623 帧（`scripts/probe_thumb_stop.py`）：
#:   手指朝下时 spread 最小 0.34 / **中位 1.16** / 最大 1.87
#:   各阈值下的"被判停住"比例：
#:     0.76 → 98.6%    0.85 → 97.3%    0.95 → 90.2%
#:     1.05 → 60.0%    1.15 → 52.3%
#: 用户症状：「向下动不了，一直卡在画面，要么就颤动」——正是 98.6% 的帧
#: 把 MOVE 全部丢弃，加上在 0.76/0.72 之间来回穿越造成的冻结/恢复交替。
#:
#: 为什么调阈值没用：朝下时的 spread（中位 1.16）与"拇指张开"姿势
#: （1.1~1.4）**区间完全重叠**，任何单一阈值都分不开两者——
#: 即使调到 1.15 仍冻结 52.3%。
#:
#: 根因与滚轮模式同一个：`spread` 用 **掌宽(5→17)** 归一化，而掌宽在用户
#: 侧手投影下会被压扁约 40~50%（他 889 帧实测掌宽/手长压缩比中位 0.597），
#: 分母塌陷 + 手腕转动抬高分子 → 比值爆炸。**二维比值判据对侧手不可靠。**
#: 换成手长归一化也只是按 ~0.6 等比缩放，分布仍然重叠，救不回来。
#:
#: 若要恢复此功能，必须换判据（例如拇指相对食指的**夹角**或
#: 相对掌面的三维关系），并重新标定——不要再在距离比上打补丁。
_THUMB_SPREAD_STOP = 0.85
_THUMB_SPREAD_RESUME = 0.72


def _thumb_spread_value(points: tuple[Landmark, ...]) -> float:
    """拇指张开度 = dist(拇指尖, 食指掌指关节) / 掌宽。

    区分用户的两种单手姿势：
    * 「操控」只伸食指、其余三指握拳、**拇指收起**靠近食指 —— 实测 0.6~0.7
    * 「停住」同样只伸食指，但**拇指主动张开**远离食指   —— 实测 1.1~1.4

    两者食指状态一样，只能靠拇指的张开程度分开。

    注意：**不能用捏合比（拇指尖到食指尖）来判**，用户实测该值在两种姿势间
    交叉（张开时因为手掌转动反而更小）。
    """

    width = _palm_width(points)
    if width <= 1e-6:
        return 0.0
    return _distance_2d(points[4], points[5]) / width


def _palm_facing_unit(points: tuple[Landmark, ...]) -> float:
    """掌面 0-5-17 三角形 3D 法向的 z 分量（单位化），即掌面朝向镜头量。

    符号来自**二维投影朝向**：0→5→17 三角形在图像平面的有向面积
    ``ax*by - ay*bx``（恰为 3D 叉积的 z 分量）。掌心与掌背在图像里互为镜像，
    绕前臂轴旋前时该三角形朝向整体翻转，故符号随掌/背翻转。

    为什么不用旧的 5→17 × 5→4：拇指尖偏离掌面，旋前时 5→4 的沿轴分量占主导，
    叉积的可变部分被压进噪声区（"符号逻辑对、观测量错"）。换成 0-5-17 的
    三角朝向，观测量才对得上符号逻辑。

    三点 z 只贡献分母（3D 模长），用于**幅度归一化**——不参与符号判定；写死
    在此，防止将来有人看到"3D 法向"就把 z 当符号来源、把归一化改回旧路。

    |返回值| 是掌面朝向镜头的置信度：→1 正对/背对镜头，→0 侧掌（0/5/17 三点
    近乎共线，退化帧返回 0.0）。**高 |值| 并不保证符号逐帧正确**：右手背的原始
    符号统计曾显示约四成帧不匹配，后经特征化确认那部分是翻手/换手的 lag 块
    （成块翻转、伴 |nz| 谷），不是逐帧噪声。过渡块由符号锁存压掉，见
    ``GestureEngine._facing_committed``；不靠调幅度门槛解决。
    """

    ax = points[5].x - points[0].x
    ay = points[5].y - points[0].y
    az = points[5].z - points[0].z
    bx = points[17].x - points[5].x
    by = points[17].y - points[5].y
    bz = points[17].z - points[5].z
    nz = ax * by - ay * bx
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    norm = sqrt(nx * nx + ny * ny + nz * nz)
    if norm < 1e-9:
        return 0.0
    return nz / norm


def _log_sign_ref_facing(
    points: tuple[Landmark, ...],
    handedness: str,
    hn_valid: bool,
    frontal_ratio: float,
    mirror: bool,
    min_nz: float,
    handedness_score: float = 0.0,
) -> None:
    """【仅参照模式】无条件输出掌面朝向量与幅度，供掌/背四姿势采符号参照。

    ``HANDMOUSE_SIGN_REF=1`` 时用它替代常规 ``nz_unit/(skip)`` 行：即使
    ``frontal_ratio`` 不足也照算照打，免得手背/侧掌那组拿不到符号。
    **纯日志**——不返回、不写状态；判定链（``entry_palm_presented`` /
    ``gate_eligible`` / gate / 注入）在上面早已算完，本函数一行都不碰。
    整块独立，删除即干净 revert。

    ``hn_score``（MediaPipe handedness 分类置信度）同样只观测不参与判定：
    ④ 调查的**分叉判据**——若追踪链错锚首帧系统性低分，score 门槛可成为
    最小修法；若错锚帧照样高置信，则标签不可救、改走几何法独立判手别。
    结论未出前本字段只供统计，任何判定不得读它。

    观测量口径随本笔由旧 ``cross``（5→17 × 5→4）换成 ``nz_unit``（0-5-17
    三角朝向）：旧参照日志里的 ``cross`` 行与新日志的 ``nz_unit`` 行不可混读。
    """

    if not hn_valid:
        _DIAG_LOGGER.info(
            "  [sign-ref] nz_unit=N/A hn=%s hn_score=%.4f frontal=%.4f（无有效手别）",
            handedness or '""', handedness_score, frontal_ratio,
        )
        return
    facing = _palm_facing_unit(points)
    _DIAG_LOGGER.info(
        "  [sign-ref] nz_unit=%.6f |nz|=%.4f(min=%.2f) hn=%s hn_score=%.4f "
        "frontal=%.4f mirror=%s（仅参照模式，不影响判定）",
        facing, abs(facing), min_nz, handedness, handedness_score,
        frontal_ratio, mirror,
    )


def _expected_palm_sign(handedness: str, mirror: bool) -> float:
    """由手别 + mirror 推出"掌心朝镜"应有的 nz 符号（+1/-1；手别缺失返回 0.0）。

    期望符号的由来（**关键，勿凭直觉改**）：
      MediaPipe 的 handedness 按**输入像素的表观外观**判手别，不做镜像补
      偿。当 ``mirror=True``（像素在推理前已被水平翻转），真实右手在模型
      帧里长得像左手、报 ``Left``，真实左手报 ``Right``。因此期望符号必须
      随 mirror 反号：``expected = (+1 if label=="Right" else -1)``，
      ``mirror`` 为真时再取 ``-expected``。写成"基准符号 × mirror 修正"而非
      两套分支，避免将来 mirror 语义再变时漏改一处。

    实测（2026-10-16 sign-ref 三轮回算，模型坐标 nz_unit 符号）：
      * 右手掌心 label=Left  nz>0 → 期望 +1；
      * 右手背   label=Left  nz<0 → 期望 +1（手背应与期望相反才叫"拒绝"）；
      * 左手掌心 label=Right nz<0 → 期望 -1；
      * 左手背   label=Right nz>0 → 期望 -1。

    handedness 缺失返回 0.0，由 ``GestureEngine._palm_presented`` 保守判 False
    （宁可进不去控制态，也不能让手背/侧掌把手误开闸门）。
    """

    if handedness not in ("Left", "Right"):
        return 0.0
    expected = 1.0 if handedness == "Right" else -1.0
    return -expected if mirror else expected


# 进入门诊断开关：环境变量控制，跨模块生效。
VERBOSE_GATE = os.environ.get("HANDMOUSE_VERBOSE_GATE", "").strip().lower() in (
    "1", "true", "yes", "on",
)
# 参照模式开关（HANDMOUSE_SIGN_REF）：**默认关、环境变量控制、不进 config**，
# 避免被持久化进 config.json 或误进生产。唯一作用：诊断日志在 frontal 不足时
# 也照打 nz_unit（供掌/背四姿势采符号参照）。**纯日志开关**，不改任何判定。
SIGN_REF = os.environ.get("HANDMOUSE_SIGN_REF", "").strip().lower() in (
    "1", "true", "yes", "on",
)
_DIAG_LOGGER = logging.getLogger("handmouse.gate-diag")
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GestureDebug:
    """最近一次识别的几何量，便于后续校准。"""

    left_pinch_ratio: float = 99.0
    right_pinch_ratio: float = 99.0
    palm_width: float = 0.0


class GestureEngine:
    """处理时序冲突并生成标准鼠标指令。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.filter = self._build_pointer_filter(config)
        # 最近一次 _smooth 输出（屏幕归一化）。正常拖拽必先经过若干 pointer
        # 帧，初值实际不会被用到；保险起见给屏幕中心，避免极端直入路径把
        # 拖拽锚点锁在左上角。
        self._last_smoothed: tuple[float, float] = (0.5, 0.5)
        #: 最近一帧的捏合比，供 HUD / 使用记录排查"捏了但没识别"。
        self._last_pinch_ratio: float = 9.9
        #: ``pointer_anchor="palm"`` 用的偏移补偿：``(食指尖 - 掌心)`` 的慢速 EMA。
        #: 见 ``_pointer_point``。``None`` = 还没初始化。
        self._anchor_offset: tuple[float, float] | None = None
        self._anchor_offset_ns: int | None = None
        self.state = ControlState.IDLE
        self.current_gesture = GestureType.NONE
        self.paused = False
        # 最近一帧有效观测"到达控制线程"的墙钟时刻。tick() 的失联判定用的也是
        # 墙钟，两者必须同源；若记采集时刻戳，elapsed 会混入取包延迟而假性失联。
        self.last_hand_ns: int | None = None
        self.debug = GestureDebug()
        # 进入门：entry_gate_enabled 关闭时恒为 True，与旧行为完全等价。
        self._gate_active = not config.entry_gate_enabled
        #: 拇指张开（停住）的迟滞锁存，避免在阈值边界反复跳。
        self._thumb_stop_latched = False
        #: 停住→捏合 的锚定起点；非 None 表示光标正在被钉住。
        self._pinch_anchor_ns: int | None = None
        #: 进入停住那一刻的捏合比，用来判断手指是否开始收拢。
        self._thumb_stop_pinch_base: float | None = None
        # 个人手势样本库（可选）。没有采集样本时为 None，行为与以前完全一致。
        self._gesture_classifier: GestureClassifier | None = None
        self._gesture_model_note = ""
        #: 本帧样本库的诊断串（"open@0.42" / "point@0.88 拒识"），HUD 用。
        self._model_debug = ""
        #: 本帧样本库给出的姿势标签，供 _allow_cursor_move 复用，避免重复算。
        self._current_model_label: str | None = None
        #: 本帧样本库最近邻距离（与 _current_model_label 同步更新）。
        #: 双指判据用它做**分级信任**：距离深（≤ gesture_model_deep_distance）
        # 说明模型深信这就是用户本人的双指姿势（实测真双指 0.53/0.67，
        # 单指误报 0.91/0.94——距离本身就是可分特征），此时直接采纳；
        # 犹豫区才落回几何可行性闸。见 gestures.py 双指块注释。
        self._model_distance: float | None = None
        self._load_gesture_model()
        self._entry_started_ns: int | None = None
        # 符号锁存：进入门朝向判定的迟滞状态（仅 _gate_active=False 时维护，
        # 开闸/退出门/暂停/失联时清零，见 _reset_facing_latch）。冷启动首帧免 K，
        # 之后只有 |nz|≥commit 的帧连续 K 帧同号才允许改变已锁定符号。
        self._locked_facing: float | None = None
        self._candidate_facing: float | None = None
        self._candidate_count: int = 0
        # 诊断用：进入门几何判定层的帧计数和去重缓存
        self._gate_diag_frame = 0
        self._gate_diag_last_sig: tuple | None = None
        # 诊断用：gate_eligible 的逐帧翻转计数（False→True）。用于把"lost 路径
        # 清零"与"姿势判定在门槛附近翻动导致清零"分开归因。
        self._gate_diag_prev_eligible = False
        self._gate_diag_eligible_flips = 0
        # 诊断用：最近一次观测被 process() 消费的墙钟时刻（到达时刻戳）。
        # 笔1 之后 last_hand_ns 已改用同一到达时刻打点，本字段因此恒等于
        # last_hand_ns（差值为 0），tick() 的 elapsed 不再含"取包延迟"段；
        # 取包延迟改用 pipeline 的 [采样] latency_ms 观察。仅在 VERBOSE_GATE 下写入。
        self._last_hand_arrival_ns: int | None = None

        self._left_active = False
        self._left_started_ns: int | None = None
        self._right_active = False
        self._right_started_ns: int | None = None
        self._right_ready = False
        self._right_open_ratio = 0.0
        self._right_requires_middle_extension = False
        self._fist_started_ns: int | None = None
        self._fist_latched = False
        self._open_anchor_x: float | None = None
        self._scroll_anchor_y: float | None = None
        #: 本帧「五指张开」判据连续成立的起始时刻（时序确认用，见
        #: ``_confirm_five_open``）。None = 当前不成立。
        self._five_open_since_ns: int | None = None
        self._pointer_tilt_anchor: float | None = None
        self._pointer_rotation_active = False
        self._page_swipe_latched = False
        #: 双指滑动锚点（掌心画面坐标）与计时
        self._swipe_anchor: tuple[float, float] | None = None
        self._swipe_started_ns: int | None = None
        self._swipe_latched_until_ns: int = 0
        #: ---- 滑轮式滚动（2026-10-22 改：连续、按距离×速度给量）----
        #: 上一帧掌心位置，用来算"这一帧走了多少"（滚动量与之成正比）。
        self._swipe_prev: tuple[float, float] | None = None
        #: 当前这一划的起点（方向反转就重新起算）。
        self._swipe_stroke_start: tuple[float, float] | None = None
        #: 锁定的滚动方向：-1=手指向上(看下方/负 delta)，+1=手指向下(看上方/正 delta)，0=未锁。
        #: 连续上划时手指必然要向下"回手"，靠这个锁避免回手被算成反向滚动。
        self._swipe_dir: int = 0
        #: 最近一次有效划动时刻，超时也解锁方向（唯一的解锁途径，见 _swipe_update）。
        self._swipe_last_stroke_ns: int | None = None
        #: 逐帧累积但未发完的滚动量（滚轮单位），保证小数不被吞掉。
        self._swipe_scroll_acc: float = 0.0
        #: 最近一次"确认是双指"的到达时刻，供滞回持有窗口计时。
        self._swipe_last_seen_ns: int | None = None
        #: 最近两次"双指判据成立"的到达时刻，用于 150ms 时间窗内的两次确认。
        #: ⚠️ 这两个字段**故意不被 `_reset_swipe_tracking` 清空**：
        #: 若清空，第一帧未确认 → 走持有失败 → 重置 → 下一帧又变回"第一帧"，
        #: 永远凑不满两次。让时间戳自然过期才是对的（见 `_SWIPE_CONFIRM_MS`）。
        self._swipe_hit_ns: int | None = None
        self._swipe_hit_prev_ns: int | None = None
        #: 本帧的 3D 逐指伸直度（``{}`` 表示该后端不提供 3D）。仅供诊断读取。
        self._last_finger3d: dict[str, float] = {}
        #: 瞄准死区的锚点 = 上一次真正发出去的 MOVE 位置。手指在死区内抖动
        #: 时 MOVE 被吞掉，光标因此钉住（见 `_suppress_micro_jitter`）。
        self._micro_anchor_xy: tuple[float, float] | None = None
        self._micro_lock: bool = False
        #: 捏合比连续低于锚定阈值的帧数（见 `_PINCH_ANCHOR_FRAMES`）。
        self._pinch_low_streak: int = 0
        #: 上一次「立即落地」点击的 ``(时刻, x, y)``（归一化坐标）。
        #: 第二次捏合据此把光标送回同一落点再发双击——捏合本身会把食指尖
        #: 带偏（实测中位漂移 56px），不修就永远凑不成双击。
        self._last_click: tuple[int, float, float] | None = None
        #: 暂停状态下"五指张开"从什么时候开始保持（见 `wake_by_open_palm`）。
        self._wake_started_ns: int | None = None
        #: 预捏合停留检测：光标原地停留超阈值 = "我要捏合了"。
        #: 仅作**软信号**（开启磁吸 + 提示），不硬冻结光标，
        #: 否则移动途中每次停顿都会卡住。
        #: ``_dwell_slow`` 是低通后的位置（激活判据用它，抖不掉）；
        #: ``_dwell_last`` 是低通位置的锚点；
        #: ``_dwell_trail`` 是最近若干原始帧，供"短窗解除"判据用。
        self._dwell_slow: tuple[float, float] | None = None
        self._dwell_slow_ns: int | None = None
        self._dwell_last: tuple[float, float] | None = None
        self._dwell_trail: list[tuple[int, float, float]] = []
        self._dwell_started_ns: int | None = None
        self._dwell_active: bool = False
        self._lost_reported = False
        self._left_button_down = False
        # 拖拽锚定式光标：进入拖拽帧锁掌心屏幕坐标与当时光标，拖拽中
        # MOVE = 锚定光标 + drag_gain × (当前掌心 - 锚定掌心)。
        self._drag_anchor_palm: tuple[float, float] | None = None
        self._drag_anchor_cursor: tuple[float, float] = (0.5, 0.5)
        # 双击（捏合两下）子状态，不占顶层 ControlState：
        # 第一次短捏合释放后不立即提交单击，挂起一个双击窗口；窗口内第二次
        # 同位置捏合（中间必须完全释放过）升级为 LEFT_DOUBLE_CLICK，窗口
        # 到期则由下一帧/tick() 补提交单击。
        self._click_pending = False
        self._click_pending_ns: int | None = None
        self._click_pending_x: float | None = None
        self._click_pending_y: float | None = None
        # 第二次捏合已确认、正在等待其释放（释放瞬间发双击）。
        self._double_armed = False
        # 本次捏合发生时的食指尖位置（归一化画面坐标），双击容差判定用。
        self._left_pinch_x: float | None = None
        self._left_pinch_y: float | None = None
        # 因双击窗口到期而提前生成的单击，随当前帧的命令一起交付。
        self._due_commands: list[MouseCommand] = []
        # KI-002 观测状态：(label, wrist_x, wrist_y, arrival_ns)，仅 VERBOSE_GATE/
        # SIGN_REF 下维护，用于 [handedness-flip] 日志；不碰任何判定路径。
        self._last_obs_handedness: tuple[str, float, float, int] | None = None
        # KI-002 轨迹纠正状态：(adopted_label, wrist_x, wrist_y, arrival_ns)。
        # 与观测日志独立——默认关闭时仍维护，保证开闸即用。
        self._trajectory_state: tuple[str, float, float, int] | None = None
        # 最近一次 handedness 翻转事件（供 HUD debug 层读取）：
        # (from_label, to_label, wrist_dist, seconds_ago, timestamp_ns)
        self._last_flip_info: tuple[str, str, float, float, int] | None = None
        # ---- 摇杆式光标状态（cursor_joystick_enabled 时生效） ----
        # 内部维护的光标位置（屏幕归一化坐标），靠帧间速度累加推进。
        self._joystick_x: float = 0.5
        self._joystick_y: float = 0.5
        # 上一帧摇杆更新的墙钟时刻，用于计算 dt；None 表示尚未初始化。
        self._last_cursor_ns: int | None = None
        #: 本帧几何度量的横向还原倍率（= 画面宽高比 ÷ 4/3）。
        #: 4:3 设备恒为 1.0，所有历史阈值原封不动；换摄像头才生效。
        #: 见 `_aspect_scale` / `_isotropic`。
        self._aspect_k: float = 1.0

    def update_config(self, config: AppConfig) -> None:
        self.config = config
        self.filter = self._build_pointer_filter(config)
        if not config.entry_gate_enabled:
            # 关掉闸门时立刻回到"始终可控"，不再要求重新做进入姿势。
            self._gate_active = True
            self._entry_started_ns = None
            self._reset_facing_latch()

    def _exit_gate(self, timestamp_ns: int) -> list[MouseCommand]:
        """握拳退出控制态：闸门回到未开。

        退出后手必须重新做进入姿势（掌心朝镜+五指张开 300ms）才能再控制，
        这样一次幻影握拳最多只是把闸门关上，不会连锁触发下游动作。
        """

        commands = self._flush_click_pending(timestamp_ns)
        self._gate_active = False
        self._entry_started_ns = None
        self._reset_transient()
        self.filter.reset()
        self.state = ControlState.IDLE
        self.current_gesture = GestureType.GATE_IDLE
        return commands

    @staticmethod
    def _build_pointer_filter(config: AppConfig) -> OneEuroPointerFilter:
        min_cutoff = 3.0 - 2.2 * config.pointer_stability
        beta = 0.35 + 1.55 * config.pointer_response
        return OneEuroPointerFilter(min_cutoff=min_cutoff, beta=beta)

    def _reset_facing_latch(self) -> None:
        """清零符号锁存。仅在进入门未开时有意义；开闸/退出门/暂停/失联时调用。"""

        self._locked_facing = None
        self._candidate_facing = None
        self._candidate_count = 0

    def _facing_committed(
        self,
        nz_unit: float,
        commit_min: float,
        k: int,
        in_hold: bool = False,
    ) -> float | None:
        """符号锁存：返回当前已锁定的朝向符号(+1/-1)；尚无锁定时返回 None。

        机制 = |nz| 参与门 + **两段式**确认：
          * 冷启动：首个 |nz| ≥ entry_palm_nz_min 的帧**立即采纳**其符号、免 K。
            否则"进入门 300ms + K 帧 ≈ 1.2s"会拖慢开闸，等于引入新 bug；
          * **hold 期**（``in_hold=True``，进入计时正在累积）：过 commit 门的
            错号帧**立即翻号、免 K**。hold 期的防抖由 entry_hold_ms 本身提供
            ——presented 一旦翻 False，进入计时立即清零重计，错号不可能靠
            "锁存翻号慢于 300ms"把闸门顶开（H4 竞速：K=8≈0.94s > hold=300ms，
            冷启动锁错过渡帧后闸门会先于锁存打开）；
          * **非 hold 期**（闸门未开、进入计时未在累积）：维持原 K 帧迟滞，
            只有 |nz| ≥ commit_min 的帧连续 K 帧同号才改符号，压换手/翻手的
            过渡块；|nz| < commit_min 的帧冻结候选（清零）。

        ``in_hold`` 取求值前的 ``_entry_started_ns is not None`` 且本帧五指张开：
        进入计时在 presented 求值之后才更新，故进入 attempt 的第一帧（计时此刻才
        置位）仍走 K 分支；冷启动 locked=None 首帧本就免 K，行为不变。五指不张开
        的 pose-break 帧不给快路径，只让进入计时重置、不翻锁存。

        迟滞的意义：|nz| 落在 [min, commit) 的帧"看得清（进得去）但不够格改变
        朝向判定"。实测右手背换手过渡块 |nz|≈0.46~0.56 正卡在此区间，被 commit
        门吃掉大部分——这是"压过渡块"而非"修逐帧符号噪声"。
        """

        sign = 1.0 if nz_unit > 0 else -1.0

        if self._locked_facing is None:
            if abs(nz_unit) >= self.config.entry_palm_nz_min:
                self._locked_facing = sign
            return self._locked_facing

        if abs(nz_unit) < commit_min:
            self._candidate_facing = None
            self._candidate_count = 0
            return self._locked_facing

        if sign == self._locked_facing:
            self._candidate_facing = None
            self._candidate_count = 0
            return self._locked_facing

        # hold 期立即跟随：防抖由 entry_hold_ms 承担，不能让 K 帧确认（≈0.94s）
        # 与 300ms 进入计时竞速——冷启动锁错过渡帧时，竞速必输（H4 实机漏检）。
        if in_hold:
            self._locked_facing = sign
            self._candidate_facing = None
            self._candidate_count = 0
            return self._locked_facing

        if sign == self._candidate_facing:
            self._candidate_count += 1
        else:
            self._candidate_facing = sign
            self._candidate_count = 1

        if self._candidate_count >= k:
            self._locked_facing = sign
            self._candidate_facing = None
            self._candidate_count = 0
        return self._locked_facing

    def _stabilize_handedness(
        self,
        handedness: str,
        points: tuple[Landmark, ...],
        arrival_ns: int,
    ) -> str:
        """KI-002 轨迹纠正：同一 wrist 轨迹下标签翻转 → 继承上一帧标签。

        只在 wrist 位移 < ``trajectory_correction_wrist_dist`` 且时间差
        ≤ ``trajectory_correction_window_ms`` 时认为属于同一追踪链；
        大跳跃或超时视为换手/追踪断裂，采纳 MediaPipe 新标签。

        默认关闭（``enable_trajectory_correction=False`` 时直接透传）。
        开关闭时仍维护 ``_trajectory_state``，保证开闸即用、不丢锚点。
        """

        if not self.config.enable_trajectory_correction:
            if points and handedness:
                self._trajectory_state = (handedness, points[0].x, points[0].y, arrival_ns)
            else:
                self._trajectory_state = None
            return handedness

        if not points or not handedness:
            self._trajectory_state = None
            return handedness

        wrist = points[0]
        prev = self._trajectory_state
        if prev is None:
            self._trajectory_state = (handedness, wrist.x, wrist.y, arrival_ns)
            return handedness

        prev_label, prev_x, prev_y, prev_ns = prev
        dt_ms = (arrival_ns - prev_ns) / 1_000_000
        dist = hypot(wrist.x - prev_x, wrist.y - prev_y)

        # 时间窗口外：追踪链断裂，采纳新标签。
        if dt_ms > self.config.trajectory_correction_window_ms:
            self._trajectory_state = (handedness, wrist.x, wrist.y, arrival_ns)
            return handedness

        # 标签未变：正常更新锚点。
        if handedness == prev_label:
            self._trajectory_state = (handedness, wrist.x, wrist.y, arrival_ns)
            return handedness

        # 标签变了且 wrist 距离在阈值内：同一追踪链的翻转，纠正为上一帧标签。
        # 锚点坐标仍推进到当前帧，避免手移动后 dist 从旧坐标算假性超标。
        if dist < self.config.trajectory_correction_wrist_dist:
            self._trajectory_state = (prev_label, wrist.x, wrist.y, arrival_ns)
            return prev_label

        # 大跳跃：换手，采纳新标签。
        self._trajectory_state = (handedness, wrist.x, wrist.y, arrival_ns)
        return handedness

    def _palm_presented(
        self,
        points: tuple[Landmark, ...],
        expected_sign: float,
        in_hold: bool = False,
    ) -> bool:
        """掌心朝镜且掌面基本正对镜头（排除侧掌与手背朝镜）。

        判据 = 非侧掌 + **锁存符号**等于期望符号 + |nz| 过进入门：
          1) 侧掌排除：掌宽/腕-中指指尖 < _ENTRY_PALM_FRONTAL_MIN；
          2) 符号判定走符号锁存 ``_facing_committed``（**先更新锁存**）。锁存分
             两段：``in_hold``（进入计时累积中）时过 commit 门即立即翻号，防抖
             交给 entry_hold_ms（presented 翻 False 计时清零）；非 hold 期才要求
             连续 K 帧同号，压掉换手/翻手的过渡块（lag 块）；
          3) |nz| < entry_palm_nz_min 的几何退化帧直接判 False。锁存**先于**
             本门更新，故退化帧会走 ``<commit`` 分支清零候选——"连续 K 帧"因此
             是严格的，不会被退化帧隔断后跨谷累加。

        实测（2026-10-16 sign-ref 三轮回算，模型坐标 nz_unit 符号）：
          右手掌心 label=Left nz>0 → 应判 True；右手背 label=Left nz<0 → False；
          左手掌心 label=Right nz<0 → 应判 True；左手背 label=Right nz>0 → False。

        expected_sign==0（handedness 缺失）时返回 False（保守关闭闸门）。
        """

        if expected_sign == 0.0:
            return False
        span = _distance_2d(points[0], points[12])
        if span < 1e-6 or _palm_width(points) / span < _ENTRY_PALM_FRONTAL_MIN:
            return False
        nz = _palm_facing_unit(points)
        # 先更新锁存、再过 min 门：退化帧（|nz| < min ≤ commit）走 _facing_committed
        # 的 ``<commit`` 分支清零候选，锁存不变。这样"连续 K 帧"严格，退化帧打断
        # 候选而不是被无视（否则 -0.9/-0.25 交替也能把 -0.9 累加到 K）。
        locked = self._facing_committed(
            nz,
            commit_min=self.config.entry_palm_nz_commit,
            k=self.config.entry_facing_commit_frames,
            in_hold=in_hold,
        )
        if abs(nz) < self.config.entry_palm_nz_min:
            return False
        if locked is None:
            return False
        return (locked > 0) == (expected_sign > 0)

    def _command(
        self,
        kind: MouseCommandType,
        timestamp_ns: int,
        **kwargs: float | int | bool | None,
    ) -> MouseCommand:
        return MouseCommand(kind=kind, timestamp_ns=timestamp_ns, **kwargs)

    def _map(self, point: Landmark) -> tuple[float, float]:
        """landmark 画面归一化坐标 → 屏幕归一化坐标（无状态纯映射）。

        四边内缩走 `AppConfig.effective_margins()`：实机标定证实该用户
        **上面够不到、下面有余量**，对称内缩会让屏幕上方一大块永远点不到。
        """

        left, top, right, bottom = self.config.effective_margins()
        return map_activity_region_edges(point.x, point.y, left, top, right, bottom)

    def _smooth(self, x: float, y: float, timestamp_ns: int) -> tuple[float, float]:
        """屏幕归一化坐标过 OneEuro，并维护最近平滑输出。

        ``_last_smoothed`` 是拖拽锚定式 MOVE 的光标基准；正常路径与
        拖拽路径共享同一个滤波器，退出拖拽时不需要重新收敛。
        """

        filtered_x, filtered_y = self.filter.update(x, y, timestamp_ns)
        if self.config.pointer_edge_override:
            # 目标到达安全活动区边界时直接完成最后一小段屏幕行程。
            # 这样不需要把手移出摄像头等待滤波器慢慢收敛到0或1。
            # 拖拽路径的 raw 同为屏幕归一化坐标，边界语义一致。
            if x <= 0.0 or x >= 1.0:
                filtered_x = x
            if y <= 0.0 or y >= 1.0:
                filtered_y = y
        self._last_smoothed = (filtered_x, filtered_y)
        return filtered_x, filtered_y

    def _cursor(self, point: Landmark, timestamp_ns: int) -> tuple[float, float]:
        return self._smooth(*self._map(point), timestamp_ns)

    def _pointer_point(
        self,
        points: tuple[Landmark, ...],
        now: int,
    ) -> Landmark:
        """光标锚点：默认食指尖，可选"掌心 + 补偿偏移"。

        动机（2026-10-23 离线实测 :mod:`scripts.audit_pointer_anchor`，701 帧）：
        掌心的**形状抖动只有食指尖的 17%~36%**（捏合段 17%）。所以锚在掌心能同时
        消掉两个老问题——"磁吸释放瞬间光标跳回随手位置"与"捏合时 1.42→0.90 这段
        光标跟着食指走"。

        🔴 但不能**直接**换成掌心：掌心在画面里比食指尖偏下偏低，直接映射等于
        把整个光标区域平移，屏幕顶部就够不到了。所以这里维护
        ``(食指尖 - 掌心)`` 的慢速 EMA 并加回去：

            锚点 = 掌心 + EMA(食指尖 - 掌心)

        平均位置与"用食指尖"几乎一致（覆盖范围不变），但**快抖被滤掉了**——
        食指尖的高频成分只剩它在 EMA 窗口里的那一点点。

        ⚠️ 偏移必须是**相对量**（两个点同一帧相减），所以换姿势（伸指/收指）时
        它自己就跟着变；闲置超过 500ms（丢手、切走再回来）直接重新初始化，
        避免拿旧姿势的偏移套新姿势。
        """

        if str(getattr(self.config, "pointer_anchor", "index_tip")) != "palm":
            return points[8]

        tip = points[8]
        palm = _palm_center(points)
        raw = (tip.x - palm.x, tip.y - palm.y)
        if self._anchor_offset is None or self._anchor_offset_ns is None or (
            now - self._anchor_offset_ns > 500_000_000
        ):
            self._anchor_offset = raw
            self._anchor_offset_ns = now
            return Landmark(palm.x + raw[0], palm.y + raw[1], palm.z)

        tau = max(0.1, float(getattr(self.config, "pointer_anchor_offset_tau_s", 1.0)))
        dt = max(0.0, min(0.5, (now - self._anchor_offset_ns) / 1_000_000_000))
        self._anchor_offset_ns = now
        alpha = 1.0 - exp(-dt / tau) if dt > 0 else 0.0
        offset_x = self._anchor_offset[0] + alpha * (raw[0] - self._anchor_offset[0])
        offset_y = self._anchor_offset[1] + alpha * (raw[1] - self._anchor_offset[1])
        self._anchor_offset = (offset_x, offset_y)
        return Landmark(palm.x + offset_x, palm.y + offset_y, palm.z)

    def _joystick_update(self, point: Landmark, arrival_ns: int) -> tuple[float, float]:
        """摇杆式光标：食指尖偏离画面中心 → 速度 → 位置累加。

        与绝对映射（``_cursor``）的本质区别：输入是食指尖相对画面中心
        的偏移方向与幅度，输出是速度，位置靠帧间 dt 累加推进。手臂可
        保持稳定，只动手指/手腕即可控制光标。

        - 死区内光标不动（过滤生理手抖）。
        - 死区外偏移越大速度越快（非线性，微动精、大动快）。
        - dt 上限 200ms，防止卡顿时光标飞出去。
        - 首次调用时把摇杆位置初始化为当前光标位置，避免跳变。
        """

        if self._last_cursor_ns is None or (arrival_ns - self._last_cursor_ns) > 500_000_000:
            # 首次调用或摇杆已闲置超过 500ms：重新锚定到当前光标位置，
            # 避免从 DRAG/绝对映射/手势切换切回摇杆时光标跳变。
            self._last_cursor_ns = arrival_ns
            self._joystick_x, self._joystick_y = self._last_smoothed
            return self._joystick_x, self._joystick_y

        dt_s = min((arrival_ns - self._last_cursor_ns) / 1_000_000_000, 0.2)
        self._last_cursor_ns = arrival_ns

        # 食指尖偏离画面中心的偏移（画面归一化坐标，x/y 范围 [0,1]）。
        dx = point.x - 0.5
        dy = point.y - 0.5
        dist = hypot(dx, dy)

        deadzone = self.config.cursor_deadzone_radius
        if dist < deadzone:
            return self._joystick_x, self._joystick_y

        # 死区外偏移映射到速度因子 [0, 1]：以 0.24 为满速参考半径
        # （画面半宽 0.5 的一半），超出钳为 1。1.5 次方让微动更精确、
        # 大动更快速，类似摇杆的非线性响应曲线。
        raw_factor = min((dist - deadzone) / 0.24, 1.0)
        speed_factor = raw_factor ** 1.5

        speed = (
            speed_factor
            * self.config.cursor_max_speed
            * self.config.cursor_speed_multiplier
        )
        move_dist = speed * dt_s

        nx = dx / dist
        ny = dy / dist
        self._joystick_x = max(0.0, min(1.0, self._joystick_x + nx * move_dist))
        self._joystick_y = max(0.0, min(1.0, self._joystick_y + ny * move_dist))
        return self._joystick_x, self._joystick_y

    def _release_left(self, timestamp_ns: int) -> list[MouseCommand]:
        if not self._left_button_down:
            return []
        self._left_button_down = False
        self._drag_anchor_palm = None
        return [self._command(MouseCommandType.LEFT_UP, timestamp_ns)]

    def _expire_click_pending(self, now_ns: int) -> None:
        """双击窗口到期：把挂起的单击补提交（进入 ``_due_commands``）。

        延迟提交单击是双击判定的必要代价：想双击时第一次点击绝不能提前
        落地（否则图标先被选中/启动一次）。
        """

        if (
            self._click_pending
            and self._click_pending_ns is not None
            and now_ns - self._click_pending_ns
            >= self.config.double_click_window_ms * 1_000_000
        ):
            self._due_commands.append(
                self._command(MouseCommandType.LEFT_CLICK, now_ns),
            )
            self._click_pending = False
            self._click_pending_ns = None
            self._click_pending_x = None
            self._click_pending_y = None

    def _flush_click_pending(self, timestamp_ns: int) -> list[MouseCommand]:
        """终止时刻（暂停/退出门/失联）立即兑现挂起的单击。

        双击已不可能再完成：被暂缓的第一次点击（含第二次捏合已经发生的
        情形，等价真实鼠标"点一下后改按住拖动"）按单击补提交。
        """

        commands: list[MouseCommand] = []
        if self._click_pending or self._double_armed:
            commands.append(self._command(MouseCommandType.LEFT_CLICK, timestamp_ns))
        self._click_pending = False
        self._click_pending_ns = None
        self._click_pending_x = None
        self._click_pending_y = None
        self._double_armed = False
        return commands

    def _reset_transient(self) -> None:
        # 进入门锁存也属"跨帧瞬时状态"：退出门/暂停/失联时一并清零，
        # 避免上一轮的朝向残值污染下次进入。
        self._reset_facing_latch()
        self._left_active = False
        self._left_started_ns = None
        self._right_active = False
        self._right_started_ns = None
        self._right_ready = False
        self._right_open_ratio = 0.0
        self._right_requires_middle_extension = False
        self._reset_orientation_tracking()
        self._reset_dwell()

    def _confirm_five_open(self, raw: bool, now: int) -> bool:
        """把"本帧几何/模型说五指张开"变成"确认的五指张开"（进严出松）。

        为什么需要这一层：``_is_five_open`` 加了长度闸后，在用户的真实样本上
        仍有 ~8% 的"只伸食指"帧全部四指投影都够长（手侧到极限，收拢指被拍成
        伸直）。单帧误报就切待机会让光标瞬冻，而**待机是没有代价补偿的状态**
        ——冻结一帧也是用户眼里的"卡了"。所以要求连续保持
        ``_FIVE_OPEN_CONFIRM_MS`` 才承认；已在待机时只要判据还成立就继续待机，
        不重复计时（否则停留期间会每隔 200ms 掉出待机、光标突然暴走一帧）。

        反面（退出待机）**不加任何延迟**：raw 为假立刻返回假，下一帧就能恢复
        移动。进严出松是刻意的——错进待机是"卡死"，错出待机只是"待机没生效"。
        """

        if not raw:
            self._five_open_since_ns = None
            return False
        if self._five_open_since_ns is None:
            self._five_open_since_ns = now
        if self.current_gesture == GestureType.OPEN_PALM:
            return True
        held_ms = (now - self._five_open_since_ns) / 1_000_000
        return held_ms >= _FIVE_OPEN_CONFIRM_MS

    def _reset_orientation_tracking(self) -> None:
        self._reset_scroll_tracking()
        self._reset_swipe_tracking()
        self._reset_pointer_rotation()

    def _reset_scroll_tracking(self) -> None:
        self._open_anchor_x = None
        self._scroll_anchor_y = None
        self._page_swipe_latched = False

    def _swipe_update(
        self,
        center: Landmark,
        now: int,
    ) -> list[MouseCommand]:
        """双指 = 滑轮：左右划翻页，上下"连续抽拉"滚动。

        方向遵循手指推着内容走：
          手指向上 → 内容上移 → **看到下面的内容** → 滚轮向下（负 delta）
          手指向下 → 看到上面的内容 → 滚轮向上
          手指向左 → 内容左移 → **看到右边的内容** → 下一页
          手指向右 → 看到左边的内容 → 上一页

        2026-10-22 重做（用户原话：翻文档/评论区太慢）：
        旧实现是"整段过阈值 → 发一次固定 3 格滚轮 → 冷却 350ms"，
        看长文档要不停抽手、每抽一次只走一点点。改成：

        1. **逐帧连续给量**：滚动量与这一帧掌心走过的位移成正比，
           划得越快/越长 → 每帧给得越多、帧数也越多 → 内容滚得越多；
           手慢下来 → 立刻变慢（"我放慢速率，滑动也要跟着放慢"）。
        2. **方向锁定**：第一次达到阈值的一划决定方向。之后同向持续累加，
           **反向的回手动作不算滚动**——连续向上划时手指必然要向下复位，
           否则每抽一次就会被"反抽"抵消掉。
        3. **解锁只看停手**：超过 ``swipe_window_ms`` 没再划 → 解锁方向，
           停下来再换方向划即可反向。
           ⚠️ 2026-10-22 用户实测后**移除了"掌心回到画面中央=静止点"的解锁**：
           人手向下划时本来就会穿过画面中央，旧逻辑把"路过中央"误判成
           "回中解锁"，正在滚动的方向被半路清零 → 滚动断掉/乱跳。
           现在滚动方向只由"停手超时"解锁，不再依赖任何屏幕位置，
           所以向下、向上划都不受手的落点影响。
        """

        prev = self._swipe_prev
        # 任何情况下都要先把"这一帧位置"记下来，否则逐帧位移会算错。
        self._swipe_prev = (center.x, center.y)
        if prev is None:
            self._swipe_stroke_start = (center.x, center.y)
            return []

        step_x = center.x - prev[0]
        step_y = center.y - prev[1]
        threshold = max(0.02, float(self.config.swipe_threshold))
        min_step = max(0.0, float(getattr(self.config, "swipe_step_deadzone", 0.002)))

        # 太久没划 → 解锁，避免方向锁死（失败还有兜底）。
        # 这是**唯一**的方向解锁途径：不再看掌心是否路过画面中央（见 docstring 3.）。
        window_ns = max(120, int(self.config.swipe_window_ms)) * 1_000_000
        if self._swipe_dir != 0 and self._swipe_last_stroke_ns is not None:
            if now - self._swipe_last_stroke_ns > window_ns:
                self._swipe_dir = 0
                self._swipe_stroke_start = (center.x, center.y)
                self._swipe_scroll_acc = 0.0

        if abs(step_x) < min_step and abs(step_y) < min_step:
            return []

        move_dir = -1 if step_y < 0 else 1

        # ---- 方向反转 = 回手/复位：重新起算这一划，且不滚动 ----
        if self._swipe_dir != 0 and move_dir != self._swipe_dir and abs(step_y) >= min_step:
            self._swipe_stroke_start = (center.x, center.y)
            return []

        if self._swipe_stroke_start is None:
            self._swipe_stroke_start = (center.x, center.y)
        start = self._swipe_stroke_start
        total_x = center.x - start[0]
        total_y = center.y - start[1]

        # ---- 横向：翻页（按整段累计判定方向，节流防连发）----
        if (
            abs(total_x) >= threshold
            and abs(total_x) > abs(total_y)
            and abs(step_x) >= min_step
        ):
            if now < self._swipe_latched_until_ns:
                return []
            kind = (
                MouseCommandType.PAGE_NEXT
                if total_x < 0
                else MouseCommandType.PAGE_PREVIOUS
            )
            self._swipe_latched_until_ns = (
                now + max(0, int(self.config.swipe_latch_ms)) * 1_000_000
            )
            self._swipe_last_stroke_ns = now
            self._swipe_stroke_start = (center.x, center.y)
            return [self._command(kind, now)]

        # ---- 纵向：首次达阈值锁方向，之后逐帧按比例给滚轮 ----
        if self._swipe_dir == 0:
            if abs(total_y) < threshold or abs(step_y) < min_step:
                return []
            self._swipe_dir = -1 if total_y < 0 else 1
            self._swipe_stroke_start = (center.x, center.y)

        if move_dir != self._swipe_dir:
            return []

        gain = max(0.0, float(getattr(self.config, "swipe_scroll_gain", 14.0)))
        self._swipe_scroll_acc += abs(step_y) * gain * 120.0
        self._swipe_last_stroke_ns = now
        whole = int(self._swipe_scroll_acc)
        min_delta = max(1, int(getattr(self.config, "swipe_min_delta", 40)))
        if whole < min_delta:
            return []
        self._swipe_scroll_acc -= whole
        delta = whole * self._swipe_dir
        return [self._command(MouseCommandType.WHEEL, now, delta=delta)]

    def _reset_swipe_tracking(self) -> None:
        self._swipe_anchor = None
        self._swipe_started_ns = None
        self._swipe_prev = None
        self._swipe_stroke_start = None
        self._swipe_dir = 0
        self._swipe_last_stroke_ns = None
        self._swipe_scroll_acc = 0.0
        self._swipe_last_seen_ns = None

    def _reset_dwell(self) -> None:
        """清空"原地停留"软信号（见 ``_update_dwell``）。"""
        self._dwell_slow = None
        self._dwell_slow_ns = None
        self._dwell_last = None
        self._dwell_trail = []
        self._dwell_started_ns = None
        self._dwell_active = False

    def _update_dwell(self, pos: tuple[float, float], now: int) -> None:
        """光标原地停留检测：连续停在阈值内超过 ``pinch_pre_dwell_ms`` = 预捏合软信号。

        用户 2026-10-21 提出：停留满 1 秒说明"我要捏合了"，把它并入捏合趋向，
        就可以把磁吸从"永远开着"降级成"只在捏合前那一小段开"，平时光标完全跟手。

        🔴 2026-10-22 起它**也参与冻结光标**（见 `cursor_frozen`）。
        理由：捏合比这个信号有个先天缺陷——
        他指向前方时拇指本来就贴着食指（他 701 帧里 move 类捏合比 p5 只有
        0.47），所以"手指合拢"这件事靠距离判永远半真半假。而"我停住不动了"
        是**用户主动做出来的**动作，与手的几何完全无关，语义也更干净。

        🔴 2026-10-23 重写判据形状（原实现**一次都没亮过**）。
        原实现拿**原始光标位置**比"从锚点的累计位移"，阈值 0.005（≈9.6px）。
        用他自己 460 秒真实日志（14270 帧）回放：整段里最安静的 1 秒，
        锚点漂移还有 24.8px，静止时逐帧抖动就有 2~4px —— 累计位移几十毫秒
        就越界、计时永远清零，激活 0 次、占比 0.0%。磁吸窗口因此从未打开。

        现在把"激活"与"解除"拆成两个不同时间尺度（**非对称是刻意的**）：

        * **激活**：先对位置做 ``pinch_pre_dwell_tau_s`` 的指数低通，再比
          ``pinch_pre_dwell_epsilon``。低通滤掉逐帧生理抖动、只留真实漂移，
          半径才可能达到（回放实测 2.3 次/分、最长 1.75s）。
        * **解除**：看**原始位置**在最近两个窗口内的位移——
          ``pinch_pre_dwell_release_*``（100ms，短促动作，一动就松）与
          ``pinch_pre_dwell_drift_*``（500ms，缓慢漂移）。若解除也用激活那套
          慢滤波，光标被钉住之后要等半秒以上才认得出他在动，就成了"黏住"。

        回放工具：``scripts/audit_dwell.py``（改参数先跑它，不许凭直觉调）。
        """
        eps = max(0.0005, float(self.config.pinch_pre_dwell_epsilon))
        tau = max(0.02, float(getattr(self.config, "pinch_pre_dwell_tau_s", 0.55)))
        release_ms = max(0, int(getattr(self.config, "pinch_pre_dwell_release_ms", 100)))
        release_eps = max(
            0.001,
            float(getattr(self.config, "pinch_pre_dwell_release_epsilon", 0.025)),
        )
        drift_ms = max(release_ms, int(getattr(self.config, "pinch_pre_dwell_drift_ms", 500)))
        drift_eps = max(
            0.001,
            float(getattr(self.config, "pinch_pre_dwell_drift_epsilon", 0.06)),
        )

        if self._dwell_slow is None or self._dwell_slow_ns is None:
            self._dwell_slow = pos
            self._dwell_slow_ns = now
            self._dwell_last = pos
            self._dwell_trail = [(now, pos[0], pos[1])]
            self._dwell_started_ns = now
            self._dwell_active = False
            return

        # 1) 位置低通：指数滑动平均。dt 越大系数越大，与 OneEuro 的语义一致。
        dt = max(0.0, (now - self._dwell_slow_ns) / 1_000_000_000)
        self._dwell_slow_ns = now
        alpha = 1.0 - exp(-min(dt, 0.5) / tau) if dt > 0 else 0.0
        slow_x = self._dwell_slow[0] + alpha * (pos[0] - self._dwell_slow[0])
        slow_y = self._dwell_slow[1] + alpha * (pos[1] - self._dwell_slow[1])
        self._dwell_slow = (slow_x, slow_y)

        # 2) 原始轨迹：只保留长窗那一段，长度有上限。
        window_ns = drift_ms * 1_000_000
        self._dwell_trail.append((now, pos[0], pos[1]))
        while len(self._dwell_trail) > 2 and now - self._dwell_trail[1][0] >= window_ns:
            self._dwell_trail.pop(0)
        trail = self._dwell_trail

        def _moved_since(milliseconds: int, limit: float) -> bool:
            """最近 ``milliseconds`` 内，原始位置是否移动超过 ``limit``。

            取"刚好够旧"的那一帧当参考（逆序遍历，第一个满足的就是最新的
            够旧帧）。窗口还没攒够时返回 False——不拿更短的时间当参考，
            否则会把一帧抖动放大成"他在动"。
            """

            need = milliseconds * 1_000_000
            for entry in reversed(trail):
                if now - entry[0] >= need:
                    return hypot(pos[0] - entry[1], pos[1] - entry[2]) > limit
            return False

        # 3) 解除：短促动作（短窗）或缓慢漂移（长窗）→ 立刻清零计时。
        moved_slow = self._dwell_last is not None and (
            hypot(slow_x - self._dwell_last[0], slow_y - self._dwell_last[1]) > eps
        )
        if (
            moved_slow
            or _moved_since(release_ms, release_eps)
            or _moved_since(drift_ms, drift_eps)
        ):
            self._dwell_last = (slow_x, slow_y)
            self._dwell_started_ns = now
            self._dwell_active = False
            return

        # 4) 激活。
        if (
            self._dwell_started_ns is not None
            and (now - self._dwell_started_ns) / 1_000_000
            >= self.config.pinch_pre_dwell_ms
        ):
            self._dwell_active = True

    def _still_ms(self, now: int) -> float:
        """"已经停稳多久"（毫秒）。dwell 未激活时也有值——它是计时的年龄。"""

        if self._dwell_started_ns is None:
            return 0.0
        return max(0.0, (now - self._dwell_started_ns) / 1_000_000)

    @property
    def dwell_active(self) -> bool:
        """给 HUD/诊断：原地停留是否已满阈值。"""

        return self._dwell_active

    @property
    def _dwell_freeze_active(self) -> bool:
        """停留是否应当**冻结光标**（与"开磁吸窗口"分开）。

        🔴 停留其实同时触发两件事，语义不同（2026-10-23 拆分）：
        * 开磁吸窗口 —— 想吸到控件中心就必须有它，走 `snap_engaged`；
        * 冻结光标 —— 防止"要点击了手还在漂"，走这里。

        后者是 2026-10-22 才加的行为，而 dwell 判据修好之前它**从没真的触发过**，
        所以用户第一次体验到它时很可能觉得"我只是停了一下，光标怎么就冻了"。
        `pinch_pre_dwell_freeze=False` 只关冻结，磁吸照旧按停留打开。
        """

        if not self._dwell_active:
            return False
        return bool(getattr(self.config, "pinch_pre_dwell_freeze", True))

    @property
    def still_ms(self) -> float:
        """给 HUD/诊断：当前已经停稳多少毫秒（未激活时也会增长）。"""

        return self._still_ms(monotonic_ns())

    @property
    def snap_engaged(self) -> bool:
        """磁吸是否应当开启：只在"预捏合窗口"内开。

        正常运行（自由移动光标）时磁吸关闭、光标完全跟手；一旦检测到
        原地停留满阈值、或已触发捏合趋向锚定，才允许磁吸把光标轻轻拉向
        附近控件——点按钮更准，又不会平时"到处乱吸"。

        🔴 2026-10-23：捏合趋向锚定这一支**必须再叠一道"已停稳"**。
        捏合比在他正常移动时就会频繁掉到 0.90 以下（实测 move 类帧里约
        10% 命中），单靠它开窗口 → 光标被反复拽到某个控件上、一秒后又松回
        随手位置，正是他反馈的"夹在两个小图标中间左右弹跳"。
        现在移动中的捏合只**冻结**光标（防漂移作用不变），不开磁吸窗口。
        """
        if self.current_gesture != GestureType.POINTER:
            return False
        if self._dwell_active:
            return True
        if self._pinch_anchor_ns is None:
            return False
        require_ms = max(0, int(getattr(self.config, "pinch_snap_still_ms", 200)))
        return self._still_ms(monotonic_ns()) >= require_ms

    def _reset_pointer_rotation(self) -> None:
        self._pointer_tilt_anchor = None
        self._pointer_rotation_active = False

    # ---- 个人手势样本库（可选增强）----
    #
    # 几何判据（夹角/距离比）假设所有人的手在镜头里投影规律一致，实测不成立：
    # 用户收拢的四指在侧手角度下投影近似共线，"只伸食指"被判成五指张开，
    # 滚轮模式修三轮修不好就是这个原因（2026-10-19 已整体移除）。
    # 样本库换成"用用户自己的手当标准"；没采集时整段跳过，行为与以前一致。

    def _load_gesture_model(self) -> None:
        """装载个人手势样本库。任何失败都只降级，绝不影响主程序启动。"""

        if not getattr(self.config, "gesture_model_enabled", False):
            return
        raw_path = getattr(self.config, "gesture_model_path", "")
        path = Path(raw_path) if raw_path else default_dataset_path()
        try:
            dataset = GestureDataset.load(path)
        except OSError as exc:
            LOGGER.warning("个人手势库读取失败，回退几何判据：%s", exc)
            self._gesture_model_note = "读取失败"
            return
        if not dataset.is_ready():
            self._gesture_model_note = f"样本不足({len(dataset)}帧)"
            LOGGER.warning(
                "个人手势库样本不足，回退几何判据：%s（每种至少 20 帧）",
                dataset.counts(),
            )
            return
        threshold = getattr(self.config, "gesture_model_reject_distance", 0.0)
        if threshold <= 0:
            threshold = suggest_reject_distance(dataset)
        self._gesture_classifier = GestureClassifier(
            dataset,
            reject_distance=threshold,
            pending_path=default_pending_path(),
            margin_ratio=getattr(self.config, "gesture_model_margin_ratio", 1.8),
        )
        self._gesture_model_note = f"{len(dataset)}帧 阈值{threshold:.2f}"

    @property
    def gesture_model_note(self) -> str:
        """给 HUD 用：样本库有没有生效、多少帧。"""

        return self._gesture_model_note

    @property
    def thumb_stop_latched(self) -> bool:
        """给 HUD 用：当前是否处于「拇指张开 = 停住」状态。"""

        return self._thumb_stop_latched

    def _pinch_anchor_active(self, now: int | None = None) -> bool:
        """「停住→捏合」的锚定窗口是否还在。"""

        if self._pinch_anchor_ns is None:
            return False
        moment = monotonic_ns() if now is None else now
        hold_ns = max(
            0,
            int(getattr(self.config, "pinch_anchor_hold_ms", 1000)),
        ) * 1_000_000
        if hold_ns <= 0:
            # 设成 0 就是关掉这个功能。必须显式返回 False：
            # 「差值 <= 0」在相邻两次 monotonic_ns 取值相同时会成立，
            # 于是关闭状态下反而锚定生效一瞬。
            return False
        return moment - self._pinch_anchor_ns <= hold_ns

    @property
    def cursor_position(self) -> tuple[float, float]:
        """最近一次平滑后的光标归一化坐标（只读，供记录/诊断）。"""

        return self._last_smoothed

    @property
    def last_pinch_ratio(self) -> float:
        """最近一帧的捏合比（dist(4,8)/掌宽）。仅用于诊断与记录。"""

        return self._last_pinch_ratio

    @property
    def pinch_anchor_active(self) -> bool:
        """给 HUD 用：当前是否处于「停住→捏合」的锚定窗口内。"""

        return self._pinch_anchor_active()

    def _pinch_approach_detected(
        self,
        points: tuple[Landmark, ...],
        pinch: float,
    ) -> bool:
        """捏合比是否已进入「要捏合了」的过渡带，可以先钉住光标。

        与 `_thumb_stop_latched` 无关 —— 这是替代手动"拇指张开停住"的方案：
        用户瞄准好之后自然捏合，光标在捏合把位置带偏之前就冻住。

        三道闸门：
        * ``pinch <= pinch_anchor_ratio``（默认 0.70，实测标定）；
        * **必须连续 ``_PINCH_ANCHOR_FRAMES`` 帧成立**；
        * **指尖不能跑出画面**。手被画面边缘切掉时二维几何会失真，
          捏合比不可信。注意这里**只判指尖在不在画面内**，
          不再判"光标是否已到屏幕边缘" —— 那个判据在 2026-10-21 实测中
          让用户"点屏幕下部的应用时锚定失效"：底部应用恰恰就在屏幕边缘。

        🔴 为什么必须加"连续帧"这一道（2026-10-22 实测）：
        锚定一旦触发就锁 ``pinch_anchor_hold_ms``(1000ms)，而**他指向前方时
        拇指本来就贴着食指**（他自己 701 帧里 move 类的捏合比 p5 只有 0.47）。
        单帧触发等于"随机冻 1 秒"。逐类模拟锚定实际生效的时间占比：
            单帧触发：移动 30.6% / 五指张开 12.1% / 双指 41.0% 都在被冻住
            连续 3 帧：移动 8.3% / 五指张开 0.0% / 双指 12.0%
        而真正的捏合覆盖率只从 88.5% 掉到 86.2%——几乎白赚。
        """

        ratio = float(getattr(self.config, "pinch_anchor_ratio", 0.70))
        if ratio <= 0.0 or pinch > ratio:
            self._pinch_low_streak = 0
            return False
        self._pinch_low_streak += 1
        if self._pinch_low_streak < _PINCH_ANCHOR_FRAMES:
            return False
        tip = points[8]
        # 手被画面边缘切掉（指尖归一化坐标越界）时几何不可信，不锚定。
        return -0.02 <= tip.x <= 1.02 and -0.02 <= tip.y <= 1.02

    @property
    def cursor_frozen(self) -> bool:
        """光标此刻是否该完全钉住。

        判定层（``_allow_cursor_move``）和插值层（``pipeline._advance_motion``）
        **必须用同一个判据**，否则又会出现"命令拦住了但插值器还在推"的
        老问题——那正是之前"显示已停住、光标却还在动"的根因。
        """

        # 拖拽必须能动，否则拖不动；其余「捏住不放」的阶段一律冻结——
        # 用户在锚定窗口里捏合时若 1 秒到期就解冻，光标会立刻被食指带走
        # （2026-10-20 用户要求：这一秒里可以捏合，别到点就飞）。
        # 用 _left_button_down 而不是 current_gesture：升格成拖拽的那一帧，
        # 手势标签还是 LEFT_PINCH，但左键已经按下了，此时必须放行，
        # 否则拖拽的第一步就被吞掉。
        if self._left_button_down:
            return False
        # 「原地停留满一秒」也算冻结（2026-10-22 用户选定）。
        # 为什么直接用它、不再单独记一个锚点：`_reset_transient()` 会在多处把
        # 跨帧瞬时状态清掉，另立一套"停留型锚定"会被它反复抹掉（实测接了
        # 一次，锚点在同一帧末尾就被 `_reset_transient` 清了，根本没生效）。
        # `_dwell_active` 本来就每帧在指针分支里更新，直接引用最省事：
        # 停住 → 冻；一动 → 自动解冻，不需要任何释放逻辑。
        # 捏合那一段由 `_pinch_anchor_active` 接管，两者互不冲突。
        return (
            self._thumb_stop_latched
            or self._pinch_anchor_active()
            or self._left_active
            or self._dwell_freeze_active
        )

    @property
    def gesture_model_debug(self) -> str:
        """给 HUD 用：本帧样本库的原始判定与距离。"""

        return self._model_debug

    def _model_label(self, points, now: int) -> str | None:
        """样本库判定的姿势；``None`` 表示没启用或没把握，回退几何判据。

        只有**既认得（距离在拒识阈值内）又连续几帧一致**才采信。单帧结果
        不采信——只看单帧正是抖动与误触的来源。
        """

        classifier = self._gesture_classifier
        if classifier is None:
            self._model_distance = None
            return None
        features = normalize(points)
        if features is None:
            self._model_distance = None
            return None
        decision = classifier.predict(features, timestamp_ns=now)
        # HUD 诊断：让用户一眼看出"模型到底在说什么"，而不是只能看最终状态猜。
        self._model_distance = float(decision.distance)
        self._model_debug = (
            f"{decision.raw_label or '-'}@{decision.distance:.2f}"
            + ("" if decision.confident else " 拒识")
        )
        if decision.stable and decision.confident:
            return decision.label
        return None

    def set_paused(
        self,
        paused: bool,
        timestamp_ns: int | None = None,
    ) -> list[MouseCommand]:
        """由托盘或快捷键显式设置暂停状态。"""

        now = timestamp_ns or monotonic_ns()
        commands = self._release_left(now)
        commands.extend(self._flush_click_pending(now))
        self.paused = paused
        self.state = ControlState.PAUSED if paused else ControlState.IDLE
        self.current_gesture = GestureType.PAUSED if paused else GestureType.NONE
        self._reset_transient()
        self.filter.reset()
        commands.append(
            self._command(MouseCommandType.PAUSE, now, paused=paused),
        )
        return commands

    def emergency_stop(self, timestamp_ns: int | None = None) -> list[MouseCommand]:
        """立即停止注入并进入暂停状态。"""

        return self.set_paused(True, timestamp_ns)

    def process(self, observation: HandObservation) -> list[MouseCommand]:
        """处理一帧有效手部观测，并顺带交付双击窗口到期补提交的单击。"""

        commands = self._process_observation(observation)
        if self._due_commands:
            commands = [*self._due_commands, *commands]
            self._due_commands = []
        if not self._allow_cursor_move(observation):
            commands = [
                command
                for command in commands
                if command.kind != MouseCommandType.MOVE
            ]
        else:
            commands = self._suppress_micro_jitter(commands)
        return commands

    def _suppress_micro_jitter(self, commands: list[MouseCommand]) -> list[MouseCommand]:
        """瞄准死区：手指只动一丁点时**不发 MOVE**，让光标钉在原地。

        2026-10-22 用户原话：「当我的手指移动微弱的时候，就要控制抖动，
        不然我瞄准小目标太难了」。手指想停住时，单目关键点仍有高频抖动，
        按原样映射到屏幕就是光标在目标上乱颤。

        做法是纯死区（相对"上次真正落点"），**不带延时也不带滞回计时**：
        抖动小于 ``cursor_micro_deadband`` 就吞掉这一帧的 MOVE；一旦超过就
        立刻跟上。所以只影响"能不能停稳"，不影响"跟不跟手"。

        注意只对 MOVE 生效——点击、滚轮、翻页一律不受影响，否则又会出现
        "手指稍微一动点击就没了"。
        """

        deadband = float(getattr(self.config, "cursor_micro_deadband", 0.0) or 0.0)
        if deadband <= 0.0:
            self._micro_anchor_xy = None
            self._micro_lock = False
            return commands

        # 一批指令里只要出现**事件**（点击/按下/抬起/滚轮/翻页），就重置锚点：
        # 事件代表"这一段瞄准结束了"，下一段的第一帧必须无条件放行，
        # 否则会犯这个错——拖拽开始那一帧的 MOVE 恰好落在上一段锚点的死区里
        # 被吞掉，左键按下后光标不动第一下（2026-10-22 实测踩到）。
        if any(command.kind != MouseCommandType.MOVE for command in commands):
            self._micro_anchor_xy = None
            self._micro_lock = False

        output: list[MouseCommand] = []
        for command in commands:
            if command.kind != MouseCommandType.MOVE or command.x is None or command.y is None:
                output.append(command)
                continue
            anchor = self._micro_anchor_xy
            if anchor is None:
                self._micro_anchor_xy = (command.x, command.y)
                self._micro_lock = False
                output.append(command)
                continue
            gap = hypot(command.x - anchor[0], command.y - anchor[1])
            if gap < deadband:
                # 抖动：这一帧不发 MOVE，光标留在原地。
                self._micro_lock = True
                continue
            self._micro_anchor_xy = (command.x, command.y)
            self._micro_lock = False
            output.append(command)
        return output

    @property
    def micro_locked(self) -> bool:
        """给 HUD 用：此刻是不是因为"手指只动一点点"而锁住了光标。"""

        return self._micro_lock

    def _geo(self, points: tuple[Landmark, ...]) -> tuple[Landmark, ...]:
        """把关键点送进"各向同性度量空间"，供**所有比值/夹角判据**使用。

        🔴 不要拿它的输出做坐标映射（``_cursor`` / ``_pointer_point`` /
        摇杆）——那些地方要的是"占画面宽高的比例"，横向乘了倍率就会跑出
        画面。判据用 geo，定位用原始 ``points``，两者**不能混**。
        """

        return _isotropic(points, self._aspect_k)

    def _allow_cursor_move(self, observation: HandObservation) -> bool:
        """移动光标的姿态门槛：只有食指伸展时才允许移动。

        为什么**只拦 MOVE、不拦点击**：捏合时食指必然弯曲（要靠近拇指），
        若用同一门槛连点击一起拦，捏合就永远点不出去。

        拦掉的三种情况：
        * 五指张开 —— 用户意图是"停住"，但状态机仍会按上一帧位置继续下发移动
        * 握拳 —— 同上，且握拳本来就该退出控制
        * 捏合途中（食指弯曲）—— 顺便把落点冻结住，抬手时不会带着光标漂移
        """

        points = observation.landmarks
        if not points or len(points) != 21:
            return True

        # 比值判据一律走各向同性空间（换摄像头不重标），取景框越界判定
        # 仍用原始归一化坐标（见 `_pinch_approach_detected`）。
        geo = self._geo(points)
        pinch = _distance_2d(geo[4], geo[8]) / _palm_width(geo)
        now = monotonic_ns()


        # ---- 捏合趋向锚定（2026-10-21 新增，不依赖 thumb_stop）----
        # 用户提出：拇指张开手动停住已经关掉，那如果能提前识别"要捏合了"，
        # 就在捏合动作把手带偏之前把光标钉住，就不需要手动停住。
        #
        # 阈值 0.70 由 521 帧实测标定（见 config.pinch_anchor_ratio 注释）：
        # 向下移动全程只有 1 帧低于 0.70，且那一帧光标已钳在屏幕底。
        if self._pinch_anchor_ns is None:
            if self._pinch_approach_detected(points, pinch):
                self._pinch_anchor_ns = now

        # 「停住」= 拇指主动张开，带迟滞：
        #   张开度 >= thumb_stop_threshold 进入停住，
        #   回落到 thumb_resume_threshold 以下才恢复。
        # 单阈值会在边界反复跳，表现为光标一顿一顿。
        # 两个阈值都必须从 config 读：旧实现释放那条用的是硬编码常量，
        # 用户在 config 里改了 thumb_resume_threshold 完全不生效。
        if not getattr(self.config, "thumb_stop_enabled", True):
            # thumb_stop 关闭时原本直接放行，但那会让"捏合趋向锚定"也失效
            # （`cursor_frozen` 靠 _pinch_anchor_active 生效，这里必须同步）。
            # 拖拽例外：按着左键必须能动。
            if pinch <= self.config.pinch_release:
                if self._left_button_down:
                    return True
                if self._pinch_anchor_active(now) or self._left_active:
                    return False
                return True
            if self._pinch_anchor_active(now):
                return False
            # 原地停留满「预捏合」阈值 → 也冻住（见 `cursor_frozen`）。
            # 只拦 MOVE，点击照常——否则停住之后就捏不出点击了。
            # 🔴 用 `_dwell_freeze_active` 而不是 `_dwell_active`：
            # 停留「开磁吸窗口」与「冻结光标」是两件事，后者可以单独关掉
            # （见 config.pinch_pre_dwell_freeze）。
            if self._dwell_freeze_active:
                return False
            return True

        # 捏合时拇指必然伸向食指尖，dist(4,5) 同样会很大，不能判"停住"。
        # 用捏合比先排除这一区间（停住姿势的捏合比在 1.1 以上，不会误入）。

        # 「停住 → 捏合」的锚定：用户停住光标后要捏合，捏合这个动作本身
        # 就会把手带偏（拇指往食指靠、手腕微动），光标跟着漂走，落点就丢了。
        # 所以在停住状态下检测到手指开始收拢，就把光标钉住 anchor_hold_ms。
        #
        # 判据用**相对下降**而不是"经过某个绝对区间"：用户停住时捏合比可能
        # 高达 2.6，捏下去是一路掉到 0，30fps 下很容易一帧就从 2.6 跳到 0.5，
        # 根本不会落在 0.72 附近，锚定也就没机会触发（2026-10-20 实测：
        # HUD 有 ●已停住 但始终没有 ⚓）。
        if (
            self._thumb_stop_latched
            and self._thumb_stop_pinch_base is not None
            and pinch < self._thumb_stop_pinch_base - 0.15
            and self._pinch_anchor_ns is None
        ):
            self._pinch_anchor_ns = now

        if pinch <= self.config.pinch_release:
            # 必须顺手清掉停住锁存。捏合动作本身就会让 dist(4,5) 变得很大
            # （拇指尖虽然在食指旁边，但离食指掌指关节很远），如果上一帧
            # 因为张开拇指把 latch 置了位，这里不清就会一直残留：
            # pipeline._advance_motion 读到 thumb_stop_latched=True 会不停
            # cancel 插值器，于是**拖拽时状态显示"正在拖拽"但光标纹丝不动**
            # （2026-10-20 用户实测）。
            self._thumb_stop_latched = False
            if self._left_button_down:
                return True
            # 锚定窗口到期后，只要还捏着（点击确认中 / 等升级成拖拽），
            # 就继续冻结，别让光标在这个节骨眼上跟着手指跑。
            if self._pinch_anchor_active(now) or self._left_active:
                return False
            return True

        if self._pinch_anchor_ns is not None and not self._pinch_anchor_active(now):
            self._pinch_anchor_ns = None

        # 样本库优先：握拳时不解锁移动。其余情况（指向类 / 没把握）继续走
        # 下面的几何判据——"停住"与"移动/捏合"的区别是拇指张开度这个连续量，
        # 单帧距离分不开（实测类间 0.37 < 类内 p90 0.69），不能交给模型硬判。
        if self._current_model_label == "fist":
            return False

        spread = _thumb_spread_value(geo)
        resume = getattr(self.config, "thumb_resume_threshold", _THUMB_SPREAD_RESUME)
        if self._thumb_stop_latched:
            if spread < resume:
                self._thumb_stop_latched = False
                self._thumb_stop_pinch_base = None
        elif spread >= getattr(self.config, "thumb_stop_threshold", _THUMB_SPREAD_STOP):
            if not self._thumb_stop_latched:
                # 刚进入停住：**当场**记下捏合比当基线。放到下一帧再记会漏掉
                # 快速捏合——用户从 2.6 一路收下去，30fps 下可能一帧就到 0.5，
                # 那一帧同时"进入停住 + 收拢完成"，晚一帧记基线就永远不触发。
                self._thumb_stop_latched = True
                self._thumb_stop_pinch_base = pinch
        return not self._thumb_stop_latched

    def _process_observation(self, observation: HandObservation) -> list[MouseCommand]:
        """处理一帧有效手部观测。"""

        # 到达时刻戳：本帧交付到控制线程的墙钟时刻。tick() 的失联判定用的是
        # 墙钟，last_hand_ns 必须记在同一时钟上；若沿用 observation.timestamp_ns
        # （采集时刻戳），elapsed = 墙钟 now - 采集时刻 会把"取包延迟 + 帧间隔"
        # 算成经过时间，10Hz 下即可逼近甚至越过 hand_lost_ms 而假性失联。
        arrival_ns = monotonic_ns()
        points = observation.landmarks
        # 宽高比补偿：摄像头的画面不是 4:3 时（16:9 的 720p/1080p 很常见），
        # 归一化坐标里的横向长度会被多压一次，所有"距离 ÷ 掌宽"的比值跟着漂。
        # 这里把尺子还原到标定尺度；4:3 设备倍率恒为 1.0，行为完全不变。
        self._aspect_k = _aspect_scale(getattr(observation, "image_aspect", _REFERENCE_ASPECT))
        geo = self._geo(points)

        # ---- KI-002 轨迹纠正（默认关） ----
        # 必须在 handedness 过滤器之前：纠正后标签与 config.handedness 匹配才允许
        # 进入后续判定。观测日志（下方）与纠正逻辑共用 _trajectory_state，保证
        # "日志看到的翻转"就是"被纠正的那个翻转"。
        corrected = self._stabilize_handedness(
            observation.handedness, points, arrival_ns,
        )
        if corrected != observation.handedness:
            observation = replace(observation, handedness=corrected)

        # KI-002 观测：追踪链内存活期的 handedness 翻转（只记日志，不改判定）。
        # 必须位于下方 handedness 过滤器之前——config.handedness != "Any" 时
        # 翻转帧会被过滤器按「无有效观测」吞掉，放过滤器后恰好漏掉最该观测的
        # 事件。Δt ≤ 500ms 才算翻转：间隙回画（lost 后重检采纳新标签）不计，
        # 窗口初拟值见 docs/known-issues.md KI-002。
        if (VERBOSE_GATE or SIGN_REF) and observation.handedness and len(points) >= 1:
            wrist = points[0]
            prev = self._last_obs_handedness
            if prev is not None:
                prev_label, prev_x, prev_y, prev_ns = prev
                if (
                    observation.handedness != prev_label
                    and arrival_ns - prev_ns <= 500_000_000
                ):
                    dist = ((wrist.x - prev_x) ** 2 + (wrist.y - prev_y) ** 2) ** 0.5
                    _DIAG_LOGGER.info(
                        "[handedness-flip] %s->%s wrist_dist=%.4f dt_ms=%d",
                        prev_label,
                        observation.handedness,
                        dist,
                        (arrival_ns - prev_ns) // 1_000_000,
                    )
                    # 同时记录供 HUD debug 层读取。
                    self._last_flip_info = (
                        prev_label,
                        observation.handedness,
                        dist,
                        (arrival_ns - prev_ns) / 1_000_000_000,
                        arrival_ns,
                    )
            self._last_obs_handedness = (
                observation.handedness,
                wrist.x,
                wrist.y,
                arrival_ns,
            )
        if self.config.handedness != "Any" and observation.handedness:
            if observation.handedness != self.config.handedness:
                # handedness 不符时本帧按"无有效观测"处理，同样交给墙钟。
                return self.tick(arrival_ns)

        now = observation.timestamp_ns

        # ---- 进入门候选：每帧只求值一次 ----
        # 符号锁存是有状态的：诊断块与进入门块若各调一次 _palm_presented，会双倍
        # 推进候选帧数、让 K 帧确认失真。故在此统一求值，两处只读结果，保证
        # "判定调用的那一次"就是"日志展示的那一次"。锁存仅在闸门未开时维护。
        entry_five_open = len(points) == 21 and _is_five_open(
            geo, observation.world_landmarks,
        )
        entry_expected = _expected_palm_sign(observation.handedness, self.config.mirror)
        entry_nz = _palm_facing_unit(geo) if len(points) == 21 else 0.0
        entry_palm_presented = False
        if not self._gate_active and len(points) == 21:
            # in_hold = 进入计时已在累积（读上一帧留下的 _entry_started_ns，它在
            # 本帧更下方的进入门块才更新）**且本帧仍五指张开**。hold 期符号立即
            # 跟随、不做 K 帧确认，避免"冷启动锁错过渡帧 → K≈0.94s 翻不过号 →
            # hold=300ms 先到 → 手背顶开闸门"的竞速（H4）。等价于 held_ms <
            # entry_hold_ms：计时一旦到点本帧即开闸并清空 _entry_started_ns，其
            # 存活期间恒在 hold。必须叠加 five_open：pose-break 帧（五指不张开）
            # 只应重置进入计时，不该借 hold 快路径翻掉朝向锁存、污染下一次进入。
            in_hold = self._entry_started_ns is not None and entry_five_open
            entry_palm_presented = self._palm_presented(
                geo, entry_expected, in_hold=in_hold,
            )

        # -------- 进入门诊断：几何判定层 --------
        # 限流：每 5 帧（~6Hz）+ gate_eligible/palm_presented/five_open 任一变 True 时立即打。
        # 相同内容去重用 signature。所有新计算都包在 if VERBOSE_GATE 内。
        # VERBOSE_GATE 或 SIGN_REF 任一开启都进诊断块（SIGN_REF 专用于四姿势参照）。
        if VERBOSE_GATE or SIGN_REF:
            self._gate_diag_frame += 1
            # 到达时刻戳：与本帧 last_hand_ns 同源（同一个 arrival_ns），
            # 因此 elapsed 拆分里的 dwell 段在笔1后恒为 0。
            self._last_hand_arrival_ns = arrival_ns
            should_log = False

            if len(points) == 21:
                # ---- 完整几何量 ----
                finger_exts = _finger_extensions(geo)
                all_ext = all(finger_exts)
                palm_w = _palm_width(geo)
                span = _distance_2d(geo[0], geo[12])
                palm_frontal_ratio = palm_w / span if span >= 1e-6 else 0.0

                hn = observation.handedness
                hn_valid = hn in ("Left", "Right")

                # 读进入门候选（上面每帧唯一求值）：判定与诊断同源，且不会因
                # 两处各调一次 _palm_presented 而双倍推进锁存候选（锁存有状态，
                # 双调即 bug）。
                # 注意 ``palm_presented`` 的语义**不再是几何掌/背真值**，而是"当前
                # 进入判定的求值结果"：闸门已开时进入判定不跑，此处恒 False。见到
                # 它 False 而手明明是掌心属正常（此刻没在做进入判定），不是 bug。
                five_open = entry_five_open
                palm_presented = entry_palm_presented
                palm_nz = entry_nz
                expected_sign = entry_expected
                locked_label = (
                    "None" if self._locked_facing is None
                    else ("+1" if self._locked_facing > 0 else "-1")
                )
                nz_match = (
                    self._locked_facing is not None
                    and (self._locked_facing > 0) == (entry_expected > 0)
                )

                gate_eligible = (not self._gate_active) and five_open and palm_presented
                held_ms = (now - self._entry_started_ns) / 1_000_000 if self._entry_started_ns else 0.0
                # 翻转计数：本帧由"不满足"变"满足"记一次。lost 清零前若 flips 无增量，
                # 说明清零来自 lost 路径；若 flips 增量大，说明是姿势在门槛附近翻动。
                if gate_eligible and not self._gate_diag_prev_eligible:
                    self._gate_diag_eligible_flips += 1
                self._gate_diag_prev_eligible = gate_eligible

                sig = (
                    self._gate_active, gate_eligible, palm_presented,
                    five_open, all_ext, hn_valid,
                )

                # 立即打条件：任一关键布尔量变 True
                # 【仅参照模式】每帧无条件打，便于统计 label 分布与符号翻转；
                # 常规模式维持原限流。
                if SIGN_REF:
                    should_log = True
                elif sig != self._gate_diag_last_sig:
                    should_log = True
                elif self._gate_diag_frame % 5 == 0:
                    should_log = True

                if should_log:
                    self._gate_diag_last_sig = sig
                    arrival_ns = self._last_hand_arrival_ns
                    _DIAG_LOGGER.info(
                        "geom ts=%d arrival=%d arrive_ms=%.0f gate_active=%s "
                        "entry_started=%s held_ms=%.0f flips=%d state=%s gesture=%s",
                        now,
                        arrival_ns if arrival_ns is not None else -1,
                        (arrival_ns - now) / 1_000_000 if arrival_ns is not None else -1.0,
                        self._gate_active,
                        self._entry_started_ns is not None, held_ms,
                        self._gate_diag_eligible_flips,
                        self.state.value, self.current_gesture.value,
                    )
                    _DIAG_LOGGER.info(
                        "  palm_width=%.4f span=%.4f frontal_ratio=%.4f(min=%.2f) "
                        "→ palm_presented=%s hn=%s valid=%s",
                        palm_w, span, palm_frontal_ratio, _ENTRY_PALM_FRONTAL_MIN,
                        palm_presented, hn or '""', hn_valid,
                    )
                    _DIAG_LOGGER.info(
                        "  k0=(%.3f,%.3f,%.3f) k4=(%.3f,%.3f,%.3f) k5=(%.3f,%.3f,%.3f)",
                        points[0].x, points[0].y, points[0].z,
                        points[4].x, points[4].y, points[4].z,
                        points[5].x, points[5].y, points[5].z,
                    )
                    _DIAG_LOGGER.info(
                        "  k8=(%.3f,%.3f,%.3f) k9=(%.3f,%.3f,%.3f) k12=(%.3f,%.3f,%.3f) k17=(%.3f,%.3f,%.3f)",
                        points[8].x, points[8].y, points[8].z,
                        points[9].x, points[9].y, points[9].z,
                        points[12].x, points[12].y, points[12].z,
                        points[17].x, points[17].y, points[17].z,
                    )
                    _DIAG_LOGGER.info(
                        "  finger_exts=(%s,%s,%s,%s) all_ext=%s thumb_spread=%.4f(min=%.2f) "
                        "→ five_open=%s",
                        *finger_exts, all_ext,
                        _distance_2d(points[4], points[17]) / palm_w,
                        _ENTRY_THUMB_SPREAD_MIN, five_open,
                    )
                    if SIGN_REF:
                        # 【仅参照模式】无条件打 nz_unit，替代常规 nz_unit/(skip)。
                        # 上面的判定（palm_presented/gate_eligible/gate/注入）已算完，
                        # 本分支只换日志输出，不改任何判定结果。整块可一键删除 revert。
                        _log_sign_ref_facing(
                            points, hn, hn_valid, palm_frontal_ratio,
                            self.config.mirror, self.config.entry_palm_nz_min,
                            observation.handedness_score,
                        )
                    elif hn_valid and palm_frontal_ratio >= _ENTRY_PALM_FRONTAL_MIN:
                        _DIAG_LOGGER.info(
                            "  nz_unit=%.6f |nz|=%.4f(min=%.2f commit=%.2f frames=%d) "
                            "mirror=%s expected_sign=%s locked=%s nz_match=%s "
                            "→ palm_presented=%s gate_eligible=%s",
                            palm_nz, abs(palm_nz), self.config.entry_palm_nz_min,
                            self.config.entry_palm_nz_commit,
                            self.config.entry_facing_commit_frames,
                            self.config.mirror,
                            "+1" if expected_sign > 0 else ("-1" if expected_sign < 0 else "N/A"),
                            locked_label, nz_match, palm_presented, gate_eligible,
                        )
                    else:
                        _DIAG_LOGGER.info(
                            "  nz_unit=(skip) expected_sign=N/A → gate_eligible=%s",
                            gate_eligible,
                        )
            else:
                # landmarks 不完整（partial 或真 0 点）：判定链必然不满足，
                # 同步翻转状态，避免下一帧被误记成 False→True。
                self._gate_diag_prev_eligible = False
                if self._gate_diag_frame % 5 == 0:
                    _DIAG_LOGGER.info(
                        "geom ts=%d landmarks_len=%d partial_finger=%s "
                        "gate_active=%s held_ms=%.0f",
                        now, len(points),
                        observation.partial_finger or "None",
                        self._gate_active,
                        (now - (self._entry_started_ns or now)) / 1_000_000,
                    )

        # ---- 进入门（测量域闸门）----
        # 未进入控制态时只评估"进入姿势"，捏合/翻页/拖拽等下游判定一概不跑。
        # 这样脸被误判成手时最多停在 gate_idle，幻影手势连控制态都进不去。
        if not self._gate_active:
            self.last_hand_ns = arrival_ns
            self._lost_reported = False
            # 复用上面每帧唯一求值的进入门候选（entry_five_open /
            # entry_palm_presented），不在此重算，避免双倍推进符号锁存候选。
            if entry_five_open and entry_palm_presented:
                if self._entry_started_ns is None:
                    self._entry_started_ns = now
                held_ms = (now - self._entry_started_ns) / 1_000_000
                if held_ms < self.config.entry_hold_ms:
                    self.state = ControlState.IDLE
                    self.current_gesture = GestureType.GATE_IDLE
                    return []
                # 姿势已满时长：本帧开闸，继续走正常手势判定。开闸后不再看掌面
                # （上面的闸门块整体跳过），锁存留着无意义且会污染下次进入，清零。
                self._gate_active = True
                self._reset_facing_latch()
                # 进入门本身就要求"五指张开连续保持 entry_hold_ms"，等于已经
                # 替下文的手势确认计过时了：把确认起点锚到进入起点，否则刚进
                # 控制态还要再悬置一次，用户会看到"进来了但光标还不动"。
                self._five_open_since_ns = self._entry_started_ns
            else:
                self._entry_started_ns = None
                self.state = ControlState.IDLE
                self.current_gesture = GestureType.GATE_IDLE
                return []
            self._entry_started_ns = None

        if observation.partial_finger and observation.partial_position is not None:
            commands = self._release_left(now)
            self._expire_click_pending(now)
            self._reset_transient()
            self.last_hand_ns = arrival_ns
            self._lost_reported = False
            if self.paused:
                self.state = ControlState.PAUSED
                self.current_gesture = GestureType.PAUSED
                return commands
            if observation.partial_finger == "Index":
                self.current_gesture = GestureType.PARTIAL_INDEX
                self.state = ControlState.MOVING
                x, y = self._cursor(observation.partial_position, now)
                commands.append(self._command(MouseCommandType.MOVE, now, x=x, y=y))
            else:
                self.current_gesture = GestureType.PARTIAL_MIDDLE
                self.state = ControlState.IDLE
            return commands
        if len(points) != 21:
            return []

        self.last_hand_ns = arrival_ns
        self._lost_reported = False
        # 双击窗口按观测时间到期：手一直在画面里时，到期的单击由本帧顺带
        # 交付（进 _due_commands，由 process() 包装层拼到本帧命令前面）。
        self._expire_click_pending(now)
        width = _palm_width(geo)
        left_ratio = _distance_2d(geo[4], geo[8]) / width
        self._last_pinch_ratio = left_ratio
        right_ratio = _distance_2d(geo[4], geo[12]) / width
        self.debug = GestureDebug(left_ratio, right_ratio, width)

        open_palm = _is_open_palm(geo)
        pointer_shape = _is_pointer(geo)
        # 五指张开 = 待机/翻页姿势（滚轮已移除，只剩左右挥动翻页）。
        # open_palm 只保留给右键预备使用。
        five_open = _is_five_open(geo, observation.world_landmarks)
        # 样本库优先：它认识"你的手"。几何判据对它认错的角度会误报 five_open，
        # 此时以样本库为准；样本库没把握则返回 None，仍按几何判据走。
        self._current_model_label = model_label = self._model_label(points, now)
        if model_label is not None:
            # 样本库认得这个姿势：它说了算，几何判据完全不参与。
            five_open = model_label == "open"
        elif self._gesture_classifier is not None:
            # 样本库在跑，但这一帧没把握（距离超过拒识阈值）。
            #
            # 这里**绝不能回退几何判据**：用户在侧手角度下只伸食指时，
            # `_is_five_open` 会因为四指投影共线而返回真，一旦被当成五指
            # 张开就会走待机分支、**光标直接卡住不动**——2026-10-19 用户
            # 实测报的就是这个。误判成待机的代价远大于暂时用不了翻页，
            # 所以没把握时一律按"不是五指张开"处理。
            five_open = False
        # 时序确认：几何（或模型）单帧说"张开"不算数，必须连续保持
        # ``_FIVE_OPEN_CONFIRM_MS``。这是最后一道防"光标瞬冻"的闸——
        # 长度闸把假阳率从 9.8% 降到 4.2%，剩下的帧在几何上真的像张开
        # （四指投影都够长），压阈值只会开始漏判真手势。
        raw_five_open = five_open
        five_open = self._confirm_five_open(five_open, now)
        palm_tilt = _palm_depth_tilt(geo)
        middle_pinch_extended = (
            _angle(geo[9], geo[10], geo[12]) >= 105.0
        )
        # 右键总开关关闭时，不得建立 _right_ready：否则握拳时拇指-中指
        # 天然靠近，会经 right_approach 直接 latch 成一次误触右键。
        if (
            self.config.right_click_enabled
            and not self.paused
            and not self._right_active
        ):
            if (
                (open_palm or pointer_shape)
                and right_ratio >= self.config.pinch_arm
            ):
                self._right_ready = True
                self._right_open_ratio = max(self._right_open_ratio, right_ratio)
                # 张掌可直接开始右键动作，但后续接触帧必须仍保留中指
                # 主动伸出；这能区分真正的中指捏合与收起中指时偶然贴近。
                self._right_requires_middle_extension = open_palm
        # 右键捏合始终要求中指伸直：只伸食指时中指自然弯曲贴近拇指，
        # 若不要求中指伸直会被误判成右键捏合。
        right_shape_valid = middle_pinch_extended
        right_approach = self._right_open_ratio - right_ratio
        right_near_candidate = (
            self.config.right_click_enabled
            and (
                self._right_active
                or (
                    self._right_ready
                    and right_shape_valid
                    and right_ratio <= self.config.pinch_arm
                    and right_approach
                    >= self.config.right_pinch_approach_delta * 0.5
                )
            )
        )
        pinch_proximity = (
            left_ratio <= self.config.pinch_arm
            or right_near_candidate
        )

        # 录屏中的“食指指向后向下翻腕”会让二维骨架短暂接近握拳。
        # 若前一帧明确处于光标模式，则以掌面深度变化确认这是翻腕，
        # 继续使用食指尖更新光标，而不是启动握拳暂停计时。
        if (
            not self.paused
            and self.config.rotation_gestures_enabled
            and self._pointer_tilt_anchor is not None
            and not five_open
            and not open_palm
            and not pinch_proximity
        ):
            pointer_tilt_delta = palm_tilt - self._pointer_tilt_anchor
            if (
                self._pointer_rotation_active
                or abs(pointer_tilt_delta) >= self.config.palm_rotation_threshold
            ):
                self._pointer_rotation_active = True
                self._pointer_tilt_anchor = palm_tilt
                self.current_gesture = GestureType.POINTER
                self.state = ControlState.MOVING
                x, y = self._cursor(self._pointer_point(points, now), now)
                return [self._command(MouseCommandType.MOVE, now, x=x, y=y)]

        # ---- 滚轮模式已整体移除（2026-10-19 用户指令）----
        # 曾在此处通过「五指张开 + 翻腕」触发滚轮。实测两个致命问题：
        # 1) 屈伸分类只看二维夹角，用户收拢的四指在侧手投影下近似共线，
        #    `all(_finger_extensions)` 为真，"只伸食指"被判成五指张开；
        # 2) `_rotation_scroll_active` 一旦置位就再无释放条件，HUD 永久
        #    显示「滚轮模式」。前后修了三轮都无法在他实机上稳定区分，
        #    最终按用户要求把滚轮模式连同翻腕滚轮、张掌滚动一并删除。
        # 五指张开现在只承担两件事：进入门姿势、左右挥动翻页。

        fist = _is_fist(geo)
        if fist and not pinch_proximity:
            if self.config.entry_gate_enabled:
                # 进入门语义：握拳 = 退出控制态。
                # 退出判定必须豁免 CLICK/DRAG——捏合本身就会让手形偏离"张开"，
                # 若在点击/拖拽中把拳头当成退出信号，用户一捏合就掉出控制态，
                # 直接自锁。这些状态下拳头只冻结本次动作，不关闸门。
                self.current_gesture = GestureType.FIST
                #: ``fist_exits_gate=False`` 时握拳**只冻结本次动作、不退出控制态**。
                #: 2026-10-21 用户实测：日常使用中手一放松就半握，被 `_is_fist`
                #: 判成握拳 → 进入门被关掉 → HUD 报「未进入控制态」、光标定住，
                #: 他以为是卡死。31002 帧记录里这种状态出现 93 次、共 76 秒。
                #: 关掉后退出控制态只能靠 M / Ctrl+Alt+Esc，反而可控。
                if not getattr(self.config, "fist_exits_gate", True):
                    return []
                if not (
                    self._left_active
                    or self._right_active
                    or self._left_button_down
                ):
                    if self._fist_started_ns is None:
                        self._fist_started_ns = now
                    held_ms = (now - self._fist_started_ns) / 1_000_000
                    if held_ms >= self.config.fist_hold_ms:
                        commands = self._release_left(now)
                        self._fist_started_ns = None
                        commands.extend(self._exit_gate(now))
                        return commands
                return []
            # 进入门未启用：保留"握拳切换暂停"的旧语义。
            commands = self._release_left(now)
            commands.extend(self._flush_click_pending(now))
            self._reset_transient()
            self.current_gesture = GestureType.FIST
            if self._fist_started_ns is None:
                self._fist_started_ns = now
            held_ms = (now - self._fist_started_ns) / 1_000_000
            if held_ms >= self.config.fist_hold_ms and not self._fist_latched:
                self._fist_latched = True
                commands.extend(self.set_paused(not self.paused, now))
            return commands

        self._fist_started_ns = None
        self._fist_latched = False

        # ---- 五指张开唤醒（2026-10-23 用户要求「开机比五根手指就能启动」）----
        # 只能唤醒、不能"停用"：停用仍然走握拳那条（用户已有的语义），
        # 免得上电时一个手势把状态来回翻。
        if self.paused and getattr(self.config, "wake_by_open_palm", False):
            if _is_five_open(geo, observation.world_landmarks):
                if self._wake_started_ns is None:
                    self._wake_started_ns = now
                elif (
                    now - self._wake_started_ns
                ) / 1_000_000 >= self.config.wake_hold_ms:
                    self._wake_started_ns = None
                    LOGGER.info("[唤醒] 五指张开已保持，恢复控制")
                    return self.set_paused(False, now)
            else:
                self._wake_started_ns = None

        if self.paused:
            self.state = ControlState.PAUSED
            self.current_gesture = GestureType.PAUSED
            return []

        commands: list[MouseCommand] = []

        # 捏合是"食指伸出去、拇指凑上来"，不是"食指尖勾下来碰到拇指"。
        #
        # 2026-10-19 用户实测：手指向下指时食指尖自然勾向拇指，二维距离骤降
        # 到 0.55 正好踩进 pinch_engage，光标开始乱点。
        #
        # 判据只能用二维（z 噪声大，项目早有约定不能用它判捏合），所以改成
        # 两个条件同时成立才否定：
        #   * 食指尖明显低于食指掌指关节（手指朝下）；
        #   * 食指本身是弯的（不是伸出去让人捏）。
        # 单独任一条都不够：侧向捏合时食指投影也像弯的，但它是朝前不是朝下。
        #
        # 只约束**进入**、不约束**保持**，否则捏住后食指微弯会被判成松开。
        index_down = points[8].y > (
            points[5].y + getattr(self.config, "pinch_down_margin", 0.08) * width
        )
        index_curled = not _finger_extensions(geo)[0]
        if getattr(self.config, "pinch_down_suppress_enabled", True):
            pinch_pose_valid = not (index_down and index_curled)
        else:
            # 关掉抑制：只看捏合比。该用户的侧手投影让 `_finger_extensions`
            # 不可靠（把真捏合判成"朝下+弯曲"），导致捏合比已经 0.32
            # 却不进入捏合态（2026-10-21 实测）。
            pinch_pose_valid = True

        left_now = left_ratio <= (
            self.config.pinch_release if self._left_active else self.config.pinch_engage
        ) and (self._left_active or pinch_pose_valid)
        right_candidate = self.config.right_click_enabled and (
            self._right_active
            or (
                self._right_ready
                and right_shape_valid
                and right_approach >= self.config.right_pinch_approach_delta
            )
        )
        right_now = right_candidate and right_ratio <= (
            self.config.pinch_release
            if self._right_active
            else self.config.pinch_engage
        ) and (self._right_active or pinch_pose_valid)

        # 已锁定的右键意图优先保持到松开，防止中指向拇指靠近时食指
        # 短暂经过拇指附近，造成一次右键动作被中途窃取成左键。
        if self._right_active:
            if right_now:
                self.current_gesture = GestureType.RIGHT_PINCH
                self._reset_orientation_tracking()
                return []
            self._right_active = False
            self._right_started_ns = None
            self._right_ready = right_ratio >= self.config.pinch_arm
            self._right_open_ratio = right_ratio if self._right_ready else 0.0
            self._right_requires_middle_extension = (
                self._right_ready and open_palm
            )
            self.current_gesture = GestureType.RIGHT_PINCH
            return [self._command(MouseCommandType.RIGHT_CLICK, now)]

        # 已开始的左键捏合保持原有点击/拖拽语义。
        if self._left_active:
            if not left_now:
                self._left_active = False
                self.current_gesture = GestureType.LEFT_PINCH
                if self._left_button_down:
                    commands.extend(self._release_left(now))
                elif self._left_started_ns is not None:
                    held_ms = (now - self._left_started_ns) / 1_000_000
                    # 短于 pinch_hold_ms 的快速掠过视为误触，不触发点击，
                    # 也不挂起双击窗口。实机易误触时调大此阈值。
                    if held_ms < self.config.pinch_hold_ms:
                        pass
                    elif getattr(self.config, "click_immediate", False):
                        # 立即落地：真实鼠标就是这样——单击零延迟，
                        # 双击由**操作系统**按时间+距离自己合并。
                        #
                        # 旧实现把第一次点击挂起 `double_click_window_ms` 等
                        # 第二次捏合，代价是每次单击都延迟半秒。实测该用户
                        # 294 段捏合只出 28 次点击、单击仅 4 次而双击 24 次：
                        # 他等不到反馈就再捏一次，本来想单点变成了双击
                        # （2026-10-21，31002 帧记录）。
                        #
                        # 🔴 2026-10-22 补一刀（他报"对一个应用捏合两次都不会
                        # 打开"）。用他 31002 帧使用日志还原 458 段捏合：
                        #   * 相邻两次捏合的**落点漂移中位 56px**
                        #     （系统双击容差只有 4px → 只有 23% 能被 OS 合并）
                        #   * 两次间隔中位 422ms，只有 51% 落在 450ms 窗口内
                        # 漂移是结构性的：捏合这个动作本身就会把食指尖往拇指
                        # 那边带，所以"第二次点击落在第一次的位置"才是对的。
                        # 现在第二次捏合会先把光标送回第一次的落点再发双击，
                        # 漂移被彻底抵消；时间窗放宽到 700ms 覆盖他更多样本。
                        promoted = False
                        previous = self._last_click
                        if previous is not None and self._left_pinch_x is not None:
                            gap_ns = now - previous[0]
                            drift = hypot(
                                self._left_pinch_x - previous[1],
                                self._left_pinch_y - previous[2],
                            )
                            window_ns = int(
                                getattr(self.config, "double_click_window_ms", 450)
                            ) * 1_000_000
                            max_drift = float(
                                getattr(self.config, "double_click_max_drift", 0.05)
                            )
                            if 0 < gap_ns <= window_ns and drift <= max_drift:
                                # 落点回到第一次的位置：抵消捏合带来的漂移，
                                # 否则两次点击相距几十像素，系统不会认双击。
                                commands.append(
                                    self._command(
                                        MouseCommandType.MOVE,
                                        now,
                                        x=previous[1],
                                        y=previous[2],
                                    ),
                                )
                                commands.append(
                                    self._command(MouseCommandType.LEFT_DOUBLE_CLICK, now),
                                )
                                self._last_click = None
                                promoted = True
                        if not promoted:
                            self._last_click = (
                                now,
                                self._left_pinch_x,
                                self._left_pinch_y,
                            )
                            commands.append(self._command(MouseCommandType.LEFT_CLICK, now))
                        self._double_armed = False
                    elif self._double_armed:
                        # 第二次捏合释放：双击成立。第一次点击从未落地，
                        # 因此不会出现"先单击一次再双击"的三连击。
                        commands.append(
                            self._command(MouseCommandType.LEFT_DOUBLE_CLICK, now),
                        )
                        self._double_armed = False
                    else:
                        # 短捏合不立即发单击：挂起双击窗口，等窗口内是否
                        # 还有第二次同位置捏合；到期由帧/tick 补提交。
                        self._click_pending = True
                        self._click_pending_ns = now
                        self._click_pending_x = self._left_pinch_x
                        self._click_pending_y = self._left_pinch_y
                self._left_started_ns = None
                self._left_pinch_x = None
                self._left_pinch_y = None
                self.state = ControlState.IDLE
                return commands
            self.current_gesture = (
                GestureType.DRAG if self._left_button_down else GestureType.LEFT_PINCH
            )
            self._reset_orientation_tracking()
            if self._left_started_ns is not None:
                held_ms = (now - self._left_started_ns) / 1_000_000
                if held_ms >= self.config.drag_hold_ms and not self._left_button_down:
                    if self._double_armed:
                        # 第二次捏合被按住超过拖拽阈值：双击不成立。先补发
                        # 第一次单击，再按下左键进入拖拽（等价真实鼠标
                        # "点一下后按住拖动"），避免第一次点击凭空消失。
                        commands.append(self._command(MouseCommandType.LEFT_CLICK, now))
                        self._double_armed = False
                    commands.append(self._command(MouseCommandType.LEFT_DOWN, now))
                    self._left_button_down = True
                    self.state = ControlState.DRAGGING
            if self._left_button_down:
                # 锚定式拖拽：掌心(0/5/9/13/17)是刚体点，食指弯缩不改变它，
                # 因此"手不动、只弯食指"不会产生任何 MOVE 位移。
                #
                # 🔴 锚点必须存**画面原始坐标**，不能存映射后的屏幕坐标。
                # 旧实现存的是 ``self._map(palm)``，而 `_map` 会把结果 clamp 到
                # [0,1]：手推到画面右侧时映射值早早就饱和在 1.0，之后掌心再往右
                # 走 delta 恒为 0 —— 用户实测「只要是拖拽的，向右拉都拉不动」
                # （2026-10-21，31002 帧记录里 46 次拖拽全部受影响）。
                palm = _palm_center(points)
                if self._drag_anchor_palm is None:
                    # 进入帧位移为 0，不跳变。
                    self._drag_anchor_palm = (palm.x, palm.y)
                    self._drag_anchor_cursor = self._last_smoothed
                anchor_fx, anchor_fy = self._drag_anchor_palm
                anchor_cx, anchor_cy = self._drag_anchor_cursor
                margin_l, margin_t, margin_r, margin_b = (
                    self.config.effective_margins()
                )
                # 画面位移 ÷ 有效行程 = 屏幕位移（等价于映射但不做 clamp）。
                travel_x = max(0.01, 1.0 - margin_l - margin_r)
                travel_y = max(0.01, 1.0 - margin_t - margin_b)
                gain = self.config.drag_gain
                raw_x = anchor_cx + gain * (palm.x - anchor_fx) / travel_x
                raw_y = anchor_cy + gain * (palm.y - anchor_fy) / travel_y
                x, y = self._smooth(raw_x, raw_y, now)
                commands.append(self._command(MouseCommandType.MOVE, now, x=x, y=y))
            return commands

        # 新右键只有在“先张开、再明显靠近”的时序成立时才允许进入。
        # 若同一帧两指都靠近，距离明显更近的目标指获胜；接近程度相同
        # 时仍保留左键优先，避免改变高频左键操作。
        right_wins = right_now and (
            not left_now or right_ratio + 0.08 < left_ratio
        )
        if right_wins:
            self.current_gesture = GestureType.RIGHT_PINCH
            self._reset_orientation_tracking()
            self._right_active = True
            self._right_started_ns = now
            self._right_ready = False
            self._right_requires_middle_extension = False
            return []

        if left_now:
            self.current_gesture = GestureType.LEFT_PINCH
            self._right_ready = False
            self._right_open_ratio = 0.0
            self._right_requires_middle_extension = False
            self._reset_orientation_tracking()
            self._left_active = True
            self._left_started_ns = now
            self._left_pinch_x = points[8].x
            self._left_pinch_y = points[8].y
            # 双击判定：上一次短捏合挂起的窗口仍在，且本次捏合与其落点
            # 间距在容差内 → 暂存"双击候选"，等本次释放时发双击。
            # 两次捏合之间必然完全释放过（能走到这里 _left_active 必为
            # False，而它只在 pinch_ratio 回到 pinch_release 以上时清除）。
            if self._click_pending and self._click_pending_ns is not None:
                within_window = (
                    (now - self._click_pending_ns) / 1_000_000
                    <= self.config.double_click_window_ms
                )
                drift = hypot(
                    points[8].x - (self._click_pending_x or points[8].x),
                    points[8].y - (self._click_pending_y or points[8].y),
                )
                if within_window and drift <= self.config.double_click_max_drift:
                    self._double_armed = True
                    self._click_pending = False
                    self._click_pending_ns = None
                    self._click_pending_x = None
                    self._click_pending_y = None
                else:
                    # 窗口超时（正常已被到期逻辑清掉）或两次落点相距太远：
                    # 第一次按独立单击立即补提交，本次捏合另起一轮。
                    self._due_commands.extend(self._flush_click_pending(now))
            self.state = ControlState.PINCH_PENDING
            return []

        # 在真正接触阈值之前提前进入点击意图区，停止所有光标轨迹。
        # 这样用户将光标对准目标后，拇指靠近指尖的过程不会拖动点击位置。
        # 同样要过"食指伸直"这一关，否则向下指时手还没靠近就先锁住点击位置。
        left_near = left_ratio <= self.config.pinch_arm and pinch_pose_valid
        right_near = (
            self.config.right_click_enabled
            and self._right_ready
            and right_shape_valid
            and right_ratio <= self.config.pinch_arm
            and right_approach >= self.config.right_pinch_approach_delta * 0.5
        )
        if left_near or right_near:
            self.current_gesture = GestureType.PINCH_ARMED
            self.state = ControlState.PINCH_PENDING
            self._reset_orientation_tracking()
            return []

        # ---- 双指滑动导航（2026-10-21 用户要求）----
        # 姿势：食指+中指伸出，无名指与小指握起。必须**排在"只伸食指=移动"
        # 之前**：`_is_pointer` 只要求食指伸展，若不先拦，双指会被当成移动。
        # 🔴 还必须排在「五指张开待机」**之前**（2026-10-21 用户实测截图）：
        # 侧手时收拢的无名指/小指投影近似共线，角度判据把它们全判成"伸直"，
        # 整只手先被 five_open 抢进待机 → 光标冻结；待机/控制来回跳在用户
        # 眼里就是"移动鼠标掉帧"。两类判据不相交（双指要求无名指/小指
        # 收拢、待机要求四指全伸），先判双指不会误伤真正的五指张开。
        if getattr(self.config, "swipe_navigation_enabled", True):
            # 🔴 判据维度 2026-10-22 第五次修订后**整体换掉**，为什么换见
            # `_SWIPE_TIP_FLOOR` 上方那段长注释（核心：旧判据 ÷ 掌宽只在
            # 侧手成立，正手永远不触发；换 ÷ 手长也救不了，必须换成
            # 同一帧内的相对比值）。这里只保留结论：
            #   * 进入：`_is_two_finger(diag)`
            #   * 持有：`_is_two_finger_hold(diag)`
            #   * 模型：只能放宽"无名指/小指收拢"这一条，**核心闸
            #     `_two_finger_main_ok` 模型说什么都不能绕**——这是
            #     2026-10-22 两次实机事故（只伸食指被判 swipe、光标冻住 +
            #     幻影滑动）换来的铁律。
            two_finger_diag = _two_finger_diag(geo)
            # 并联的第二个尺子已在 `_two_finger_hit` 里并入：逐指"伸出来了吗"
            # （见 `_FINGER_STRAIGHT_MIN` 上方注释）。两个尺子的盲区互补——
            # 跨指指尖距离抓不住"整根手指折向掌心"，逐指伸直抓不住
            # "折在指根但三节仍共线"。
            # 🔴 第三条路：米制 3D 世界坐标（见 `_FINGER3D_STRAIGHT_MIN`
            # 上方注释）。2D 那两条都在"归一化图像坐标"里算，而 x/y 除以的
            # 是**不同的数**（宽 vs 高），手指横着指和竖着指量出来的比例
            # 本来就不一样——这是"换个角度又失灵"的物理来源。3D 判据在
            # 真正的三维空间里量，没有这个问题，所以它排在最前面。
            # 判据：有 3D 就只听 3D，2D 只在没有 3D 时兜底（见 `_two_finger_hit`）。
            # 3D 是他要求的"看手指本体"，实测误触从 13.2%/12.1% 降到 3.3%/0%。
            geom_two_finger = _two_finger_hit(
                geo, two_finger_diag, observation.world_landmarks,
            )
            self._last_finger3d = (
                _finger3d_feats(observation.world_landmarks)
                if len(observation.world_landmarks) == 21
                else {}
            )
            # 模型的唯一作用域：核心闸成立（食指、中指都量出来够长）时，
            # 允许它把"收拢"闸放宽一点点。它不能发起一次"核心闸都没过"
            # 的滑动，所以不会再把指向姿势变成滚动。
            model_two_finger = (
                model_label == "swipe"
                and _two_finger_main_ok(two_finger_diag)
                and two_finger_diag[3] <= _SWIPE_MODEL_CURL
                and two_finger_diag[4] <= _SWIPE_MODEL_CURL
            )
            two_finger = geom_two_finger or model_two_finger

            # 进入需要**两次成立落在 150ms 内**（实机 15fps ≈ 2 帧）。
            # 单帧就进入会把孤立误报变成一次真的滚动；而"连续 N 帧"计数
            # 在这里会死锁（第一帧未确认 → 走持有失败 → 重置计数 → 又是
            # 第一帧），所以用时间窗而不是帧计数。
            if two_finger:
                self._swipe_hit_prev_ns = self._swipe_hit_ns
                self._swipe_hit_ns = arrival_ns
            confirm_ns = _SWIPE_CONFIRM_MS * 1_000_000
            confirmed = (
                self._swipe_hit_prev_ns is not None
                and self._swipe_hit_ns - self._swipe_hit_prev_ns <= confirm_ns
            )
            if two_finger and not pinch_proximity:
                if confirmed or self.current_gesture == GestureType.SWIPE:
                    self._swipe_last_seen_ns = arrival_ns
                    self._reset_pointer_rotation()
                    self._reset_scroll_tracking()
                    self.current_gesture = GestureType.SWIPE
                    self.state = ControlState.IDLE
                    commands.extend(
                        self._swipe_update(_palm_center(points), arrival_ns),
                    )
                    return commands
                # 判据成立但还没凑满两次确认：**悬置**——既不发 MOVE 也不进
                # SWIPE，保持上一帧不动。为什么不按移动处理：双指姿势下
                # 食指尖常落在画面另一处，跟过去就是"光标先飞一下再冻住"，
                # 比直接不动更糟（与 `raw_five_open` 的悬置同一套道理）。
                self.state = ControlState.IDLE
                return commands

            # ---- 滞回：滚动中判据偶发失败 → 短暂"持有"，绝不掉回移动 ----
            # 掉回移动的那一帧会把光标瞬移到食指位置——2026-10-22 用户
            # 实机反馈"连续滑动的时候鼠标老是跟着我的手指乱飞"就是它。
            # 运动模糊、划动末端手指回弯、掌面轻微翻转都会让判据抖一两帧；
            # 持有判据比进入松一档（只要中指还大体伸着、无名指小指没完全
            # 张开）就按住不动，判据回来后无缝继续滚。
            #
            # 实测这条是"双指能不能用"的**主要**杠杆，比进入判据本身还大：
            # 单帧判据命中率在他 16 个采集会话里是 35%~96%（不是"某些姿势
            # 永不成立"，就是单帧抖动），靠持有窗口把 53.2% 拉到 81.9%。
            hold_ns = (
                max(0, int(getattr(self.config, "swipe_hold_ms", 250)))
                * 1_000_000
            )
            if (
                hold_ns > 0
                and self.current_gesture == GestureType.SWIPE
                and self._swipe_last_seen_ns is not None
                and _is_two_finger_hold(two_finger_diag)
                and not five_open
                and arrival_ns - self._swipe_last_seen_ns <= hold_ns
            ):
                # 只保鲜位置、不累计滚动：恢复后的第一帧位移仍是帧间小位移。
                self._swipe_prev = (_palm_center(points).x, _palm_center(points).y)
                self.current_gesture = GestureType.SWIPE
                self.state = ControlState.IDLE
                return []
        self._reset_swipe_tracking()

        if raw_five_open and not five_open:
            # 判据成立，但还没满 ``_FIVE_OPEN_CONFIRM_MS``：**悬置**——
            # 既不发 MOVE 也不进待机，保持上一帧的手势。
            # 为什么不按移动处理：张开姿势下食指尖常落在画面边缘，若这 200ms
            # 仍跟食指走，用户会看到光标先"飞"一下再冻住，比直接不动更糟。
            # 为什么不立即进待机：那就是回到"单帧误报冻光标"的老毛病。
            self.state = ControlState.IDLE
            return commands

        if five_open:
            # 五指张开 = 待机姿势：不移动光标、不发滚轮，只保留左右挥动翻页。
            # （滚轮已按用户要求整体移除——见上方注释。）
            self._reset_pointer_rotation()
            self.current_gesture = GestureType.OPEN_PALM
            self.state = ControlState.IDLE
            center = _palm_center(points)
            if self._open_anchor_x is None or self._scroll_anchor_y is None:
                self._open_anchor_x = center.x
                self._scroll_anchor_y = center.y
                return []
            movement_x = (center.x - self._open_anchor_x) / width
            movement_y = (center.y - self._scroll_anchor_y) / width
            horizontal = abs(movement_x)
            vertical = abs(movement_y)
            if (
                self.config.page_gestures_enabled
                and not self._page_swipe_latched
                and horizontal >= self.config.page_swipe_threshold
                and horizontal > vertical * 1.35
            ):
                self._page_swipe_latched = True
                self.current_gesture = (
                    GestureType.PAGE_NEXT
                    if movement_x < 0
                    else GestureType.PAGE_PREVIOUS
                )
                kind = (
                    MouseCommandType.PAGE_NEXT
                    if movement_x < 0
                    else MouseCommandType.PAGE_PREVIOUS
                )
                return [self._command(kind, now)]
            if vertical >= self.config.scroll_threshold:
                # 张掌上下平移只刷新锚点（原为滚轮），不再产生任何输出。
                self._open_anchor_x = center.x
                self._scroll_anchor_y = center.y
            return commands

        self._reset_swipe_tracking()

        self._reset_scroll_tracking()
        # 走到这里说明MediaPipe已经给出完整的21点手部观测，并且该帧
        # 不是握拳、捏合或张掌滚动。移动模式不再额外要求食指维持
        # ≥135°的僵硬姿势；自然微弯、侧手或中指半伸时仍可控制光标。
        self.current_gesture = GestureType.POINTER
        self.state = ControlState.MOVING
        self._pointer_tilt_anchor = palm_tilt
        self._pointer_rotation_active = False
        if self.config.cursor_joystick_enabled:
            # 摇杆模式：食指尖偏离画面中心驱动光标速度，手臂可保持稳定。
            x, y = self._joystick_update(self._pointer_point(points, arrival_ns), arrival_ns)
        else:
            x, y = self._cursor(self._pointer_point(points, now), now)
        # 原地把光标喂给"停留检测"：停留满阈值才点亮预捏合磁吸窗口。
        # 用平滑后的光标位置（_last_smoothed）而不是原始食指尖，避免手抖
        # 触发的微小位移让计时反复清零。
        self._update_dwell(self._last_smoothed, arrival_ns)
        commands.append(self._command(MouseCommandType.MOVE, now, x=x, y=y))
        return commands

    def tick(self, timestamp_ns: int | None = None) -> list[MouseCommand]:
        """检查手部数据超时并执行失联保护。"""

        now = timestamp_ns or monotonic_ns()
        # 控制线程每 ~8ms 调一次 tick：即使没有新观测帧，挂起单击的双击
        # 窗口也能在这里按时到期，单击延迟被严格钳在一个窗口长度。
        self._expire_click_pending(now)
        due = self._due_commands
        self._due_commands = []
        if self.last_hand_ns is None:
            return list(due)
        elapsed_ms = (now - self.last_hand_ns) / 1_000_000
        if elapsed_ms < self.config.hand_lost_ms or self._lost_reported:
            return list(due)

        commands = list(due)
        # 手已离画：挂起的双击等不到第二次捏合了，按单击兑现，避免一次
        # 正常点击因为手随后出画而凭空消失。
        commands.extend(self._flush_click_pending(now))
        commands.extend(self._release_left(now))
        self._reset_transient()
        self.filter.reset()
        self.state = ControlState.LOST
        self.current_gesture = GestureType.LOST
        self._lost_reported = True
        # 手已离画：五指张开的连续计时必须清零。否则手回来后的第一帧会拿
        # 出画前的旧起始时刻算出"已保持很久"，直接把光标冻进待机。
        self._five_open_since_ns = None
        if VERBOSE_GATE:
            # 只在 lost 真正触发时打（_lost_reported 保证每 episode 一行）。
            # elapsed = dwell(取包延迟) + gap(消费后空转)，两段之和恒等于 elapsed_ms。
            arrival = self._last_hand_arrival_ns
            if arrival is None:
                _DIAG_LOGGER.info(
                    "lost-trigger now=%d last_hand_ns=%s elapsed_ms=%.0f "
                    "arrival=None（本 episode 未记录到达时刻，无法拆分）"
                    "flips=%d hand_lost_ms=%d",
                    now, self.last_hand_ns, elapsed_ms,
                    self._gate_diag_eligible_flips, self.config.hand_lost_ms,
                )
            else:
                _DIAG_LOGGER.info(
                    "lost-trigger now=%d last_hand_ns=%s elapsed_ms=%.0f "
                    "arrival=%d dwell_ms=%.0f gap_ms=%.0f flips=%d hand_lost_ms=%d",
                    now, self.last_hand_ns, elapsed_ms, arrival,
                    (arrival - self.last_hand_ns) / 1_000_000,
                    (now - arrival) / 1_000_000,
                    self._gate_diag_eligible_flips,
                    self.config.hand_lost_ms,
                )
        if self.config.entry_gate_enabled:
            # 手出画 ≥ hand_lost_ms 也是进入门的退出条件：重新入画后必须再
            # 做一次进入姿势，避免"手一晃出去又晃回来"直接恢复控制。
            self._gate_active = False
            self._entry_started_ns = None
            self._reset_facing_latch()
        return commands
