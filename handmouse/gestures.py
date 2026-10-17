"""基于关键点几何关系的手势状态机。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from math import acos, degrees, hypot, sqrt
from time import monotonic_ns

from handmouse.config import AppConfig
from handmouse.contracts import (
    GestureType,
    HandObservation,
    Landmark,
    MouseCommand,
    MouseCommandType,
)
from handmouse.filters import OneEuroPointerFilter, map_activity_region


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


def _distance_2d(a: Landmark, b: Landmark) -> float:
    """手势阈值使用图像平面距离，避开单目深度坐标的帧间噪声。"""

    return hypot(a.x - b.x, a.y - b.y)


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
    index, middle, ring, pinky = _finger_extensions(points)
    return index and middle and sum((index, middle, ring, pinky)) >= 3


def _is_fist(points: tuple[Landmark, ...]) -> bool:
    if any(_finger_extensions(points)):
        return False
    center = _palm_center(points)
    width = _palm_width(points)
    tips = (8, 12, 16, 20)
    mean_distance = sum(_distance(points[index], center) for index in tips) / len(tips)
    return mean_distance / width < 1.25


#: 进入门阈值：拇指外展程度 = 拇指尖到小指掌指关节的距离 / 掌宽。
_ENTRY_THUMB_SPREAD_MIN = 0.9
#: 进入门阈值：掌面正对镜头程度 = 掌宽 / 腕到中指指尖的距离，用于排除侧掌。
_ENTRY_PALM_FRONTAL_MIN = 0.25


def _is_five_open(points: tuple[Landmark, ...]) -> bool:
    """五指张开：四指伸直且拇指明显外展。

    ``_is_open_palm`` 只要 3/4 指伸直（照顾日常滚轮姿势），进不了进入门。
    进入门是全局闸门，判据必须更严，否则"半握"也能把闸门顶开。
    """

    if not all(_finger_extensions(points)):
        return False
    return (
        _distance_2d(points[4], points[17]) / _palm_width(points)
        >= _ENTRY_THUMB_SPREAD_MIN
    )


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
        self.state = ControlState.IDLE
        self.current_gesture = GestureType.NONE
        self.paused = False
        # 最近一帧有效观测"到达控制线程"的墙钟时刻。tick() 的失联判定用的也是
        # 墙钟，两者必须同源；若记采集时刻戳，elapsed 会混入取包延迟而假性失联。
        self.last_hand_ns: int | None = None
        self.debug = GestureDebug()
        # 进入门：entry_gate_enabled 关闭时恒为 True，与旧行为完全等价。
        self._gate_active = not config.entry_gate_enabled
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
        self._rotation_tilt_anchor: float | None = None
        self._rotation_scroll_active = False
        self._rotation_last_open_ns: int | None = None
        self._pointer_tilt_anchor: float | None = None
        self._pointer_rotation_active = False
        self._page_swipe_latched = False
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
        """landmark 画面归一化坐标 → 屏幕归一化坐标（无状态纯映射）。"""

        return map_activity_region(
            point.x,
            point.y,
            self.config.pointer_margin_x,
            self.config.pointer_margin_y,
        )

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

    def _reset_orientation_tracking(self) -> None:
        self._reset_scroll_tracking()
        self._reset_pointer_rotation()

    def _reset_scroll_tracking(self) -> None:
        self._open_anchor_x = None
        self._scroll_anchor_y = None
        self._rotation_tilt_anchor = None
        self._rotation_scroll_active = False
        self._rotation_last_open_ns = None
        self._page_swipe_latched = False

    def _reset_pointer_rotation(self) -> None:
        self._pointer_tilt_anchor = None
        self._pointer_rotation_active = False

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
        return commands

    def _process_observation(self, observation: HandObservation) -> list[MouseCommand]:
        """处理一帧有效手部观测。"""

        # 到达时刻戳：本帧交付到控制线程的墙钟时刻。tick() 的失联判定用的是
        # 墙钟，last_hand_ns 必须记在同一时钟上；若沿用 observation.timestamp_ns
        # （采集时刻戳），elapsed = 墙钟 now - 采集时刻 会把"取包延迟 + 帧间隔"
        # 算成经过时间，10Hz 下即可逼近甚至越过 hand_lost_ms 而假性失联。
        arrival_ns = monotonic_ns()
        points = observation.landmarks
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
        entry_five_open = len(points) == 21 and _is_five_open(points)
        entry_expected = _expected_palm_sign(observation.handedness, self.config.mirror)
        entry_nz = _palm_facing_unit(points) if len(points) == 21 else 0.0
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
                points, entry_expected, in_hold=in_hold,
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
                finger_exts = _finger_extensions(points)
                all_ext = all(finger_exts)
                palm_w = _palm_width(points)
                span = _distance_2d(points[0], points[12])
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
        width = _palm_width(points)
        left_ratio = _distance_2d(points[4], points[8]) / width
        right_ratio = _distance_2d(points[4], points[12]) / width
        self.debug = GestureDebug(left_ratio, right_ratio, width)

        open_palm = _is_open_palm(points)
        pointer_shape = _is_pointer(points)
        palm_tilt = _palm_depth_tilt(points)
        middle_pinch_extended = (
            _angle(points[9], points[10], points[12]) >= 105.0
        )
        if not self.paused and not self._right_active:
            if (
                (open_palm or pointer_shape)
                and right_ratio >= self.config.pinch_arm
            ):
                self._right_ready = True
                self._right_open_ratio = max(self._right_open_ratio, right_ratio)
                # 张掌可直接开始右键动作，但后续接触帧必须仍保留中指
                # 主动伸出；这能区分真正的中指捏合与收起中指时偶然贴近。
                self._right_requires_middle_extension = open_palm
        right_shape_valid = (
            not self._right_requires_middle_extension
            or middle_pinch_extended
        )
        right_approach = self._right_open_ratio - right_ratio
        right_near_candidate = (
            self._right_active
            or (
                self._right_ready
                and right_shape_valid
                and right_ratio <= self.config.pinch_arm
                and right_approach
                >= self.config.right_pinch_approach_delta * 0.5
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
                x, y = self._cursor(points[8], now)
                return [self._command(MouseCommandType.MOVE, now, x=x, y=y)]

        # 张掌后允许通过前后翻腕控制滚轮。该判定必须先于握拳：翻腕时
        # 四指的二维投影会缩短，旧逻辑会把平展的手误判成握拳并暂停。
        if (
            not self.paused
            and self.config.rotation_gestures_enabled
            and self._rotation_tilt_anchor is not None
            and not open_palm
            and not pinch_proximity
        ):
            tilt_delta = palm_tilt - self._rotation_tilt_anchor
            if abs(tilt_delta) >= self.config.palm_rotation_threshold:
                notches = round(
                    -tilt_delta * self.config.scroll_sensitivity * 6,
                )
                if notches == 0:
                    notches = -1 if tilt_delta > 0 else 1
                notches = max(-3, min(3, notches))
                self._rotation_tilt_anchor = palm_tilt
                self._rotation_scroll_active = True
                self.current_gesture = GestureType.SCROLL
                self.state = ControlState.SCROLLING
                return [
                    self._command(
                        MouseCommandType.WHEEL,
                        now,
                        delta=notches * 120,
                    ),
                ]
            if self._rotation_scroll_active and not open_palm:
                self.current_gesture = GestureType.SCROLL
                self.state = ControlState.SCROLLING
                return []

        fist = _is_fist(points)
        if fist and not pinch_proximity:
            # 从张掌进入翻腕的最初几帧可能尚未跨过滚轮阈值。给深度
            # 姿态一个短暂判定窗口，避免先启动500ms握拳暂停计时。
            if (
                not self.paused
                and self.config.rotation_gestures_enabled
                and self._rotation_last_open_ns is not None
                and (now - self._rotation_last_open_ns) / 1_000_000 <= 350
            ):
                self.current_gesture = GestureType.SCROLL
                self.state = ControlState.SCROLLING
                return []
            if self.config.entry_gate_enabled:
                # 进入门语义：握拳 = 退出控制态。
                # 退出判定必须豁免 CLICK/DRAG——捏合本身就会让手形偏离"张开"，
                # 若在点击/拖拽中把拳头当成退出信号，用户一捏合就掉出控制态，
                # 直接自锁。这些状态下拳头只冻结本次动作，不关闸门。
                self.current_gesture = GestureType.FIST
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
        if self.paused:
            self.state = ControlState.PAUSED
            self.current_gesture = GestureType.PAUSED
            return []

        commands: list[MouseCommand] = []
        left_now = left_ratio <= (
            self.config.pinch_release if self._left_active else self.config.pinch_engage
        )
        right_candidate = self._right_active or (
            self._right_ready
            and right_shape_valid
            and right_approach >= self.config.right_pinch_approach_delta
        )
        right_now = right_candidate and right_ratio <= (
            self.config.pinch_release
            if self._right_active
            else self.config.pinch_engage
        )

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
                    if self._double_armed:
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
                palm_x, palm_y = self._map(_palm_center(points))
                if self._drag_anchor_palm is None:
                    # 进入帧位移为 0：raw 等于上次平滑输出，滤波器 deadzone
                    # 内严格回传同一坐标，不跳变。
                    self._drag_anchor_palm = (palm_x, palm_y)
                    self._drag_anchor_cursor = self._last_smoothed
                anchor_px, anchor_py = self._drag_anchor_palm
                anchor_cx, anchor_cy = self._drag_anchor_cursor
                gain = self.config.drag_gain
                raw_x = anchor_cx + gain * (palm_x - anchor_px)
                raw_y = anchor_cy + gain * (palm_y - anchor_py)
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
        left_near = left_ratio <= self.config.pinch_arm
        right_near = (
            self._right_ready
            and right_shape_valid
            and right_ratio <= self.config.pinch_arm
            and right_approach >= self.config.right_pinch_approach_delta * 0.5
        )
        if left_near or right_near:
            self.current_gesture = GestureType.PINCH_ARMED
            self.state = ControlState.PINCH_PENDING
            self._reset_orientation_tracking()
            return []

        if open_palm:
            self._reset_pointer_rotation()
            self.current_gesture = GestureType.SCROLL
            self.state = ControlState.SCROLLING
            center = _palm_center(points)
            if self._open_anchor_x is None or self._scroll_anchor_y is None:
                self._open_anchor_x = center.x
                self._scroll_anchor_y = center.y
                self._rotation_tilt_anchor = palm_tilt
                self._rotation_last_open_ns = now
                return []
            self._rotation_last_open_ns = now
            movement_x = (center.x - self._open_anchor_x) / width
            movement_y = (center.y - self._scroll_anchor_y) / width
            horizontal = abs(movement_x)
            vertical = abs(movement_y)
            tilt_delta = palm_tilt - (
                self._rotation_tilt_anchor
                if self._rotation_tilt_anchor is not None
                else palm_tilt
            )
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
                self._rotation_tilt_anchor = palm_tilt
                return [self._command(kind, now)]
            if (
                self.config.rotation_gestures_enabled
                and not pinch_proximity
                and abs(tilt_delta) >= self.config.palm_rotation_threshold
            ):
                notches = round(
                    -tilt_delta * self.config.scroll_sensitivity * 6,
                )
                if notches == 0:
                    notches = -1 if tilt_delta > 0 else 1
                notches = max(-3, min(3, notches))
                self._rotation_tilt_anchor = palm_tilt
                self._rotation_scroll_active = True
                return [
                    self._command(
                        MouseCommandType.WHEEL,
                        now,
                        delta=notches * 120,
                    ),
                ]
            if (
                vertical >= self.config.scroll_threshold
                and vertical > horizontal * 1.15
            ):
                notches = max(
                    -3,
                    min(3, round(-movement_y * self.config.scroll_sensitivity * 4)),
                )
                if notches:
                    commands.append(
                        self._command(
                            MouseCommandType.WHEEL,
                            now,
                            delta=notches * 120,
                        ),
                    )
                self._open_anchor_x = center.x
                self._scroll_anchor_y = center.y
                self._rotation_tilt_anchor = palm_tilt
            return commands

        self._reset_scroll_tracking()
        # 走到这里说明MediaPipe已经给出完整的21点手部观测，并且该帧
        # 不是握拳、捏合或张掌滚动。移动模式不再额外要求食指维持
        # ≥135°的僵硬姿势；自然微弯、侧手或中指半伸时仍可控制光标。
        self.current_gesture = GestureType.POINTER
        self.state = ControlState.MOVING
        self._pointer_tilt_anchor = palm_tilt
        self._pointer_rotation_active = False
        x, y = self._cursor(points[8], now)
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
