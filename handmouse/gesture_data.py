"""个人手势样本库：特征归一化与落盘。

为什么要采集而不是继续调通用阈值
--------------------------------
通用判据（例如"指节夹角 >= 135° 就算伸直"）隐含一个假设：所有人的手在
镜头里的投影规律都一样。实测不成立——用户收拢的四指在侧手角度下投影近似
共线，夹角判据直接失效，同一个手势在不同手腕角度下被判成完全不同的结果。
滚轮模式前后修了三轮都修不好，根因就在这里（2026-10-19 已整体移除）。

本模块换成"用用户自己的手当标准"：把 MediaPipe 的 21 个关键点归一化成
**不随平移和远近变化**的特征向量存起来，识别时找最像的那一批样本。
不再写任何"伸直/弯曲"的几何规则。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from handmouse.config import PROJECT_ROOT
from handmouse.contracts import Landmark

#: 第一版手势集，与当前控制语义一一对应。
LABELS: tuple[str, ...] = ("move", "stop", "pinch", "open", "fist", "swipe")

#: 采集界面用中文名。
LABEL_NAMES: dict[str, str] = {
    "move": "移动（只伸食指）",
    "stop": "停住（食指+拇指张开）",
    "pinch": "捏合（点击）",
    "open": "五指张开",
    "fist": "握拳",
    "swipe": "双指滑动（食指+中指伸开，其余握拳）",
}

#: 特征维度 = 除手腕外的 20 个点 × (x, y)。
FEATURE_DIM = 40

#: 归一化后单坐标的合理上限。超过说明这一帧关键点飘了，直接丢弃。
_MAX_ABS = 6.0


def palm_width(points: Sequence[Landmark]) -> float:
    """掌宽 = 食指掌指关节到小指掌指关节的二维距离。"""

    return hypot(points[5].x - points[17].x, points[5].y - points[17].y)


def hand_length(points: Sequence[Landmark]) -> float:
    """手长 = 腕（0）到中指掌指关节（9）的二维距离。

    **这是归一化用的尺度**，不是掌宽。2026-10-19 实测数据说明原因：

    用户 641 帧样本里「手长/掌宽」的中位数是 1.4~1.9（正常手约 1.1），
    说明他的手在画面里普遍是**侧着**的。掌宽在侧手时被透视压扁 40%~50%，
    而它当分母 → 同一手势换个角度，特征值能差一倍 → 类内最近邻中位距离
    高达 0.55（正常应 <0.25），跨类边界彻底糊掉。

    手长沿手掌纵向，恰恰是绕前臂轴旋转时**几乎不变**的那条边，所以用它
    当尺子能消掉绝大部分角度带来的尺度漂移。换成它以后类内距离降了 40%。
    """

    return hypot(points[9].x - points[0].x, points[9].y - points[0].y)


def wrist_rotation_ratio(points: Sequence[Landmark]) -> float:
    """手腕旋转程度的观测量 = 掌宽 / 手掌长度。

    绕前臂轴转手腕时，掌宽在图像平面上被透视压缩，而手掌长度（腕到中指
    掌指关节）基本不变，所以这个比值能直接反映"手掌侧到什么程度"：

    * 掌心正对镜头 → 掌宽充分展开，比值接近 1.0
    * 手掌侧过来   → 掌宽被压扁，比值掉到 0.3 上下

    采集时用它统计角度覆盖：**用户正常使用一段手势，往往只停留在一两个
    角度上**，而识别失败的根源正是没见过的角度。程序盯着这个值分成若干档，
    哪一档没采到就提示换角度，而不是强制用户按脚本转手腕。
    """

    width = palm_width(points)
    length = hypot(points[9].x - points[0].x, points[9].y - points[0].y)
    if length < 1e-4:
        return 0.0
    return width / length


def normalize(points: Sequence[Landmark]) -> tuple[float, ...] | None:
    """把 21 个关键点归一化成 40 维特征，帧不可用时返回 ``None``。

    两步归一化：

    1. **平移**：全部点减去手腕（0 号点），手的左右上下移动不影响特征；
    2. **缩放**：除以**手长**（不是掌宽，原因见 :func:`hand_length`），
       手离摄像头的远近不影响特征。

    0 号点归一化后恒为原点，不含信息，直接去掉。

    刻意**不做二维旋转对齐**：旋转对齐会把"手掌朝上/朝下/侧手"压成同一种
    形状，而这些朝向恰恰携带信息；更关键的是，透视投影造成的畸变不是二维
    旋转能补偿的。正确的做法是采集时让手自然扫过各个角度，让样本自己覆盖，
    交给最近邻去学。
    """

    if len(points) != 21:
        return None
    scale = hand_length(points)
    if scale < 1e-4:
        return None
    wrist = points[0]
    features: list[float] = []
    for point in points[1:]:
        x = (point.x - wrist.x) / scale
        y = (point.y - wrist.y) / scale
        if abs(x) > _MAX_ABS or abs(y) > _MAX_ABS:
            return None
        features.append(x)
        features.append(y)
    if len(features) != FEATURE_DIM:
        return None
    return tuple(features)


def rescale_features(features: Sequence[float]) -> tuple[float, ...] | None:
    """把"以掌宽为尺"的旧特征换算成"以手长为尺"。

    旧库里的 |腕→中指掌指关节| 在掌宽尺度下就是 ``features[16:18]`` 的模，
    所以除以这个模即可精确换尺，不需要重新采集。用于迁移 2026-10-19 之前
    采的样本库。
    """

    if len(features) != FEATURE_DIM:
        return None
    scale = hypot(features[16], features[17])
    if scale < 1e-4:
        return None
    return tuple(value / scale for value in features)


#: 手腕旋转角分档数量，用于采集时的角度覆盖度统计。
COVERAGE_BINS = 8
#: 旋转比值的合理区间，超出按端点截断。
_RATIO_MIN = 0.25
_RATIO_MAX = 1.25


class CoverageTracker:
    """记录采集过程中扫过多少种手腕角度。

    用法：每录一帧 ``mark(wrist_rotation_ratio(points))``，界面读
    ``covered`` / ``total`` 判断角度覆盖是否够。用户自由做手势时往往只停在
    一两个角度，这个计数器负责在缺角度时提醒，而不是强行要求转手腕。
    """

    def __init__(self, bins: int = COVERAGE_BINS) -> None:
        self.bins = max(1, bins)
        self._seen: set[int] = set()

    def _bin(self, ratio: float) -> int:
        span = _RATIO_MAX - _RATIO_MIN
        position = int((ratio - _RATIO_MIN) / span * self.bins)
        return max(0, min(self.bins - 1, position))

    def mark(self, ratio: float) -> None:
        self._seen.add(self._bin(ratio))

    @property
    def covered(self) -> int:
        return len(self._seen)

    @property
    def total(self) -> int:
        return self.bins

    @property
    def complete(self) -> bool:
        return self.covered >= self.bins

    def reset(self) -> None:
        self._seen.clear()

    def hint(self) -> str:
        """给界面的提示：角度够了就别打扰，不够才提醒换角度。"""

        if self.complete:
            return "角度覆盖已足够"
        return f"角度覆盖 {self.covered}/{self.bins}，换个手腕角度再来一点"


@dataclass(frozen=True, slots=True)
class GestureSample:
    """一条带标签的手势样本。"""

    label: str
    features: tuple[float, ...]
    captured_ns: int = 0
    #: 采集批次标识，便于回溯"这批样本是哪次采的、什么光照"。
    source: str = ""
    #: 采集时的**米制 3D 世界坐标**（21 个点，扁平存 x,y,z…）。空元组表示
    #: 该样本没有 3D（旧数据、回放、后端不提供）。存它是为了以后能离线
    #: 重算任何基于 3D 的判据，**不必重采**——40 维的 ``features`` 只有
    #: x/y，任何涉及深度的问题在旧数据上都永远测不了。
    world: tuple[float, ...] = ()

    def to_json(self) -> dict:
        raw = {
            "label": self.label,
            "features": [round(value, 6) for value in self.features],
            "captured_ns": self.captured_ns,
            "source": self.source,
        }
        if self.world:
            raw["world"] = [round(value, 5) for value in self.world]
        return raw

    @classmethod
    def from_json(cls, raw: dict) -> GestureSample | None:
        """从一行 json 还原。维度或标签不合法返回 ``None``。"""

        label = raw.get("label")
        raw_features = raw.get("features")
        if not isinstance(label, str) or not isinstance(raw_features, list):
            return None
        if len(raw_features) != FEATURE_DIM:
            return None
        try:
            features = tuple(float(value) for value in raw_features)
        except (TypeError, ValueError):
            return None
        captured = raw.get("captured_ns", 0)
        source = raw.get("source", "")
        # ``world`` 是可选字段：老样本没有它，读出来就是空元组。
        # 长度不对就整段丢掉，不要拿半个坐标去过判据。
        raw_world = raw.get("world")
        world: tuple[float, ...] = ()
        if isinstance(raw_world, list) and len(raw_world) == 63:
            try:
                world = tuple(float(value) for value in raw_world)
            except (TypeError, ValueError):
                world = ()
        return cls(
            label=label,
            features=features,
            captured_ns=int(captured) if isinstance(captured, (int, float)) else 0,
            source=source if isinstance(source, str) else "",
            world=world,
        )


class GestureDataset:
    """手势样本集合，支持增量追加与 jsonl 落盘。"""

    def __init__(self, samples: Iterable[GestureSample] = ()) -> None:
        self._samples: list[GestureSample] = list(samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[GestureSample]:
        return iter(self._samples)

    @property
    def samples(self) -> tuple[GestureSample, ...]:
        return tuple(self._samples)

    def add(self, sample: GestureSample) -> None:
        self._samples.append(sample)

    def add_frame(
        self,
        label: str,
        points: Sequence[Landmark],
        *,
        source: str = "",
        captured_ns: int = 0,
        world: Sequence[Landmark] = (),
    ) -> bool:
        """归一化一帧并入库。帧不可用（丢点/关键点飘了）返回 ``False``。"""

        features = normalize(points)
        if features is None:
            return False
        flat_world: tuple[float, ...] = ()
        if len(world) == 21:
            flat_world = tuple(
                value for point in world for value in (point.x, point.y, point.z)
            )
        self._samples.append(
            GestureSample(
                label=label,
                features=features,
                captured_ns=captured_ns,
                source=source,
                world=flat_world,
            ),
        )
        return True

    def counts(self) -> dict[str, int]:
        counts = {label: 0 for label in LABELS}
        for sample in self._samples:
            counts[sample.label] = counts.get(sample.label, 0) + 1
        return counts

    def for_label(self, label: str) -> list[GestureSample]:
        return [sample for sample in self._samples if sample.label == label]

    def labels_present(self) -> tuple[str, ...]:
        """返回真正有样本的标签。

        🔴 2026-10-23 修复：原实现只遍历写死的 ``LABELS``（细分六类），
        而**粗分类**（``gesture_model.COARSE_LABELS`` 里的 ``point``）不在里面——
        症状是粗分类数据集 ``is_ready()`` 恒 False、``clean_dataset()`` 把
        ``point`` 的样本整类跳过（实测 739 帧 / 83%），**且不报任何错**。
        生产流程清洗的是细分数据集，所以一直没被触发，是埋着的坑。

        现在以**实际样本**为准：先按 ``LABELS`` 的固定顺序输出已知标签
        （保持旧顺序，界面按顺序展示的地方不会跳），再补上其它出现过的标签
        （按名字排序，结果稳定可复现，不依赖字典插入序）。
        """

        counts = self.counts()
        ordered = [label for label in LABELS if counts.get(label)]
        extra = sorted(
            label
            for label, count in counts.items()
            if count and label not in LABELS
        )
        return tuple(ordered + extra)

    def is_ready(self, min_per_label: int = 20) -> bool:
        """每种**已采集**的手势都凑够最少样本才算可用。

        只检查真正出现的标签（``labels_present()``），而不是全部 ``LABELS``：
        这样早期只采了 5 类、后来才加入第 6 类 ``swipe`` 的旧样本库依然算就绪，
        不会让已训练好的模型突然加载失败；同时允许"只补采某一类"（如
        ``--only swipe``）后就能看到那一类自己的准确率。
        """

        counts = self.counts()
        present = self.labels_present()
        if not present:
            return False
        return all(counts.get(label, 0) >= min_per_label for label in present)

    def merge(self, other: GestureDataset) -> None:
        self._samples.extend(other.samples)

    def save(self, path: Path) -> int:
        """写入 jsonl（一行一条）。返回写入条数，并自动建父目录。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("w", encoding="utf-8") as handle:
            for sample in self._samples:
                handle.write(json.dumps(sample.to_json(), ensure_ascii=False))
                handle.write("\n")
                written += 1
        return written

    def append(self, path: Path) -> int:
        """追加写入（低置信度回流样本走这条，不覆盖已有库）。"""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with path.open("a", encoding="utf-8") as handle:
            for sample in self._samples:
                handle.write(json.dumps(sample.to_json(), ensure_ascii=False))
                handle.write("\n")
                written += 1
        return written

    @classmethod
    def load(cls, path: Path) -> GestureDataset:
        """读取 jsonl。文件不存在返回空库；坏行跳过而不是整个崩掉。"""

        path = Path(path)
        if not path.exists():
            return cls()
        samples: list[GestureSample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(raw, dict):
                    continue
                sample = GestureSample.from_json(raw)
                if sample is not None:
                    samples.append(sample)
        return cls(samples)


def default_dataset_path() -> Path:
    """正式样本库位置。"""

    return PROJECT_ROOT / "gesture_data" / "samples.jsonl"


def default_pending_path() -> Path:
    """待补标样本位置：在线识别拿不准时自动落到这里。"""

    return PROJECT_ROOT / "gesture_data" / "pending.jsonl"
