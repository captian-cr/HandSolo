"""网格选择的两级判定：捏合确认 + 目标稳定。

为什么分成两个类
----------------
* ``PinchJudger`` 只回答"这一刻是否发生了一次确认动作"（边沿触发，一次
  捏合只出一次）。
* ``TargetStabilizer`` 只回答"现在稳定指向哪个图标"（跨帧去抖，避免在
  图标边界来回闪）。

两者互不知道对方，便于分别测试，也便于以后换确认方式（比如改成停留确认）
而不动目标判定。

捏合比值为什么不用掌宽归一化
----------------------------
旧实现用 ``dist(拇指尖,食指尖) / dist(5,17)``（掌宽）。掌宽在斜视、手占
画面比例小时会被投影急剧压缩——实机 HUD 曾出现捏合比 1.93（正常 0.9~1.3），
说明分母被低估 40% 以上，所有阈值随之集体漂移。
改用**食指自身长度** ``dist(5,8)`` 归一化：它和捏合距离同处一条指链上，
随手的远近、朝向同步缩放，比值稳定得多。
"""

from __future__ import annotations

from dataclasses import dataclass, field


def pinch_ratio(
    thumb_tip: tuple[float, float],
    index_tip: tuple[float, float],
    index_root: tuple[float, float],
) -> float:
    """拇指尖-食指尖距离 ÷ 食指长度。越小越接近捏合。

    食指长度用 ``dist(5,8)``（掌指关节到指尖）。手完全侧对镜头时该长度
    会趋近 0，此时返回 ``inf`` 让上层按"无法判定"处理，而不是给出一个
    虚假的小比值把捏合误触发出来。
    """

    from math import hypot

    length = hypot(index_tip[0] - index_root[0], index_tip[1] - index_root[1])
    if length < 1e-4:
        return float("inf")
    distance = hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
    return distance / length


@dataclass(slots=True)
class PinchJudger:
    """边沿触发的捏合确认器。

    ``enter_ratio`` 与 ``release_ratio`` 之间是滞回带：比值落在带内保持原
    状态，避免手指在阈值附近抖动导致连续误触发。
    """

    enter_ratio: float = 0.40
    release_ratio: float = 0.62
    hold_ms: float = 120.0

    _closed: bool = field(default=False, init=False)
    _closed_since_ms: float = field(default=0.0, init=False)
    _fired: bool = field(default=False, init=False)
    _ratio: float = field(default=float("inf"), init=False)
    _just_closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.enter_ratio <= 0:
            raise ValueError("enter_ratio 必须为正")
        if self.release_ratio <= self.enter_ratio:
            raise ValueError("release_ratio 必须大于 enter_ratio")
        if self.hold_ms < 0:
            raise ValueError("hold_ms 不能为负")

    @property
    def ratio(self) -> float:
        return self._ratio

    @property
    def just_closed(self) -> bool:
        """上一帧是否刚刚进入捏合（边沿）。

        捏合动作本身会让食指往拇指方向移动，而食指正是定位锚点。若不在这
        一刻把目标钉住，球会跟着手指滑走，确认时就点到了相邻图标。上层应在
        本帧用 ``just_closed`` 锁定目标。
        """

        return self._just_closed

    @property
    def is_closed(self) -> bool:
        """当前是否处于捏合（用于 UI 高亮，不代表已触发）。"""

        return self._closed

    def update(self, ratio: float, now_ms: float) -> bool:
        """喂入当前比值与时间，返回"此刻是否应执行一次确认"。"""

        self._just_closed = False
        self._ratio = ratio
        if ratio == float("inf"):
            # 手丢失或姿态不可判：直接复位，不留下悬挂的捏合状态。
            self.reset()
            return False

        if not self._closed:
            if ratio <= self.enter_ratio:
                self._closed = True
                self._just_closed = True
                self._closed_since_ms = now_ms
                self._fired = False
            return False

        if ratio >= self.release_ratio:
            self._closed = False
            self._fired = False
            return False

        if not self._fired and now_ms - self._closed_since_ms >= self.hold_ms:
            self._fired = True
            return True
        return False

    def reset(self) -> None:
        self._closed = False
        self._closed_since_ms = 0.0
        self._fired = False
        self._just_closed = False


@dataclass(slots=True)
class ClickGate:
    """把连续的捏合确认合成为单击 / 双击。

    隔空捏合没有触觉反馈，误触的代价比鼠标大得多，所以默认要求**两次捏合**
    才真正执行打开：第一次只选中并亮起，窗口内再捏一次才触发。窗口内没有
    第二次，就回落成单击——由上层决定是否执行（通常只做选中提示）。

    ``window_ms`` 比系统双击间隔（约 500ms）稍宽，因为抬手再捏一次比动手指
    慢，用系统值会让人觉得"明明双击了却没反应"。
    """

    window_ms: float = 600.0

    _first_ms: float = field(default=0.0, init=False)
    _first_target: int | None = field(default=None, init=False)
    _armed: bool = field(default=False, init=False)

    def reset(self) -> None:
        self._first_ms = 0.0
        self._first_target = None
        self._armed = False

    @property
    def armed(self) -> bool:
        """是否已记下一次捏合、正在等第二次。"""

        return self._armed

    @property
    def pending_target(self) -> int | None:
        return self._first_target if self._armed else None

    def register(self, target: int, now_ms: float) -> str | None:
        """登记一次捏合确认。返回 ``"double"`` 表示应执行，``None`` 表示还要等。"""

        if (
            self._armed
            and target == self._first_target
            and now_ms - self._first_ms <= self.window_ms
        ):
            self.reset()
            return "double"
        self._first_ms = now_ms
        self._first_target = target
        self._armed = True
        return None

    def poll(self, now_ms: float) -> int | None:
        """窗口过期则返回那次单次点击的目标，并清空状态。"""

        if self._armed and now_ms - self._first_ms > self.window_ms:
            target = self._first_target
            self.reset()
            return target
        return None


@dataclass(slots=True)
class TargetStabilizer:
    """目标跨帧去抖：连续 ``confirm_frames`` 帧指向同一图标才算稳定。

    手指在图标边界附近时，最近邻结果会逐帧来回跳。不做跨帧确认的话，
    高亮会闪烁、确认会打偏。靠近安全区（``None``）同样需要连续帧，
    避免手刚滑出图标就被立刻判成"无目标"。
    """

    confirm_frames: int = 3

    _candidate: int | None = field(default=None, init=False)
    _streak: int = field(default=0, init=False)
    _stable: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.confirm_frames < 1:
            raise ValueError("confirm_frames 必须 >= 1")

    @property
    def stable(self) -> int | None:
        return self._stable

    def update(self, target: int | None) -> int | None:
        if target == self._candidate:
            self._streak += 1
        else:
            self._candidate = target
            self._streak = 1
        if self._streak >= self.confirm_frames:
            self._stable = self._candidate
        return self._stable

    def reset(self) -> None:
        self._candidate = None
        self._streak = 0
        self._stable = None
