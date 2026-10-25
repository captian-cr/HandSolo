"""``hand_pose`` 姿态分类测试（用合成关键点，不依赖摄像头）。"""

from __future__ import annotations

from handmouse.hand_pose import classify_pose, extended_names, is_extended

Point = tuple[float, float]


def make_hand(
    *,
    index: bool = False,
    middle: bool = False,
    ring: bool = False,
    pinky: bool = False,
    thumb: bool = False,
) -> list[Point]:
    """构造一只手的 21 个归一化关键点。

    指根固定在一条横线上；手指伸直时"指尖离指根远"，弯曲时"指尖折回指根附近"。
    """

    points: list[Point] = [(0.50, 0.62)] * 21  # 手腕占位

    roots = {"index": (0.44, 0.50), "middle": (0.49, 0.50),
             "ring": (0.54, 0.50), "pinky": (0.59, 0.50)}
    joints = {"index": (5, 6, 8), "middle": (9, 10, 12),
              "ring": (13, 14, 16), "pinky": (17, 18, 20)}

    for name, (mcp, pip, tip) in joints.items():
        rx, ry = roots[name]
        points[mcp] = (rx, ry)
        if (name == "index" and index) or (name != "index" and locals()[name]):
            points[pip] = (rx, ry - 0.08)
            points[tip] = (rx, ry - 0.20)
        else:
            points[pip] = (rx, ry - 0.08)
            points[tip] = (rx, ry - 0.04)

    # 拇指：收拢时指尖贴近食指根，张开时远离
    points[2] = (0.42, 0.55)
    points[4] = (0.36, 0.44) if thumb else (0.43, 0.53)

    return points


def test_only_index_is_point() -> None:
    assert classify_pose(make_hand(index=True)) == "point"


def test_all_fingers_open() -> None:
    hand = make_hand(index=True, middle=True, ring=True, pinky=True)

    assert classify_pose(hand) == "open"


def test_closed_fist() -> None:
    assert classify_pose(make_hand()) == "fist"


def test_index_plus_middle_is_other() -> None:
    """比"V"手势既不是操控也不是翻页，应当忽略，避免误判。"""

    assert classify_pose(make_hand(index=True, middle=True)) == "other"


def test_middle_only_is_other() -> None:
    assert classify_pose(make_hand(middle=True)) == "other"


def test_thumbs_up_still_fist() -> None:
    """拇指不影响分类：竖大拇指时其余四指收着，仍算空闲。"""

    assert classify_pose(make_hand(thumb=True)) == "fist"


def test_too_few_points() -> None:
    assert classify_pose([(0.0, 0.0)] * 10) == "other"


def test_is_extended_per_finger() -> None:
    hand = make_hand(index=True)

    assert is_extended(hand, "index") is True
    assert is_extended(hand, "middle") is False
    assert is_extended(hand, "pinky") is False


def test_extended_names() -> None:
    hand = make_hand(index=True, ring=True)

    assert set(extended_names(hand)) == {"index", "ring"}


def test_degenerate_joint_is_not_extended() -> None:
    """指节与指根重合时不可判，按弯曲处理（宁可漏触发不误触发）。"""

    hand = make_hand(index=True)
    hand[6] = hand[5]

    assert is_extended(hand, "index") is False
