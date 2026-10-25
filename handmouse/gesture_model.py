"""个人手势识别：最近邻模板匹配 + 时序共识 + 明确拒识。

为什么不训练神经网络
--------------------
用户第一版每种手势也就几百帧样本。这个量级下，最近邻（KNN）比神经网络更
合适，而且有三个神经网络给不了的好处：

1. **能拒识**。距离太远就明确回答"我不认识这个姿势"。现有方案所有误触的
   根源，就是几何规则永远会强行把每一帧塞进某一类——它不敢说不知道。
2. **补采即时生效**。往库里加样本就行，不用重新训练、不用调超参。
3. **可解释**。能直接指出"你现在最像你采集的哪一帧"，采错了还是判错了
   一眼看出来。

样本量上到几千帧之后，再换成小 MLP 也不迟，接口不用动。

时序共识
--------
单帧最近邻会抖：手在两类之间过渡时，逐帧结果来回跳。所以看最近几帧的
投票，够票数才切换状态。这一步只作用于**状态类手势**（移动/停住/张开/
握拳），不作用于光标坐标——坐标另有一套低延迟平滑，两者不混。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from math import sqrt
from typing import Iterator, Sequence

from handmouse.gesture_data import FEATURE_DIM, GestureDataset, GestureSample

#: 粗分类映射：把"只伸食指 / 停住 / 捏合"合并成同一类 ``point``。
#:
#: 为什么必须合并（2026-10-19 实测 641 帧）：
#: 这三个手势的区别只是**拇指离食指多远**，在二维投影里是一个连续量，
#: 类间最近距离 0.37 反而小于类内 p90 0.69 —— 单帧距离根本没有边界。
#: 强行三分类只会得到一堆误判。
#:
#: 它们的真正区分靠**时序与过程**：捏合是"捏住-松开"的瞬时动作（已有
#: 停留时长 + 双击确认），停住是持续状态（已有拇指张开度迟滞）。这些
#: 现有逻辑本来就是对的，不需要模型插手。
#:
#: 模型只负责它确实分得开的三类：point / open / fist
#: （实测跨类最近距离 1.0~2.4，类内中位 0.21~0.30，边界干净）。
COARSE_MAP: dict[str, str] = {
    "move": "point",
    "stop": "point",
    "pinch": "point",
    "open": "open",
    "fist": "fist",
    # 双指滑动是"食中二指伸开、其余握拳"的静止姿势，在特征空间里与
    # "只伸食指"明显可分（中指尖位置不同），所以单独成一类，不并入 point。
    # 这样模型能直接认出双指姿势，手斜着的时候比纯几何判据更稳。
    "swipe": "swipe",
}
COARSE_LABELS: tuple[str, ...] = ("point", "open", "fist", "swipe")

#: 粗分类的中文名，采集报告与 HUD 用。
COARSE_NAMES: dict[str, str] = {
    "point": "指向类（移动/停住/捏合）",
    "open": "五指张开",
    "fist": "握拳",
    "swipe": "双指滑动",
}

#: 参与投票的最近邻个数。
DEFAULT_K = 3
#: 时序窗口帧数与切换所需的最少票数。
DEFAULT_WINDOW = 3
DEFAULT_MIN_VOTES = 2
#: 「更像 A 而不是 B」的倍数门槛：第二近的异类样本要比最近邻远这么多倍，
#: 才认为这次识别是明确的（用于多峰样本库，绝对距离不可靠时兜底）。
DEFAULT_MARGIN_RATIO = 1.8

#: 默认拒识距离。采集后由 :func:`suggest_reject_distance` 按用户自己的
#: 样本重新算，写回 config，这里只是没有样本时的兜底。
DEFAULT_REJECT_DISTANCE = 0.55
#: 低置信度样本回流的节流间隔：不节流的话 8fps 会一直往盘里灌同一姿势。
DEFAULT_PENDING_INTERVAL_NS = 400_000_000


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        delta = a - b
        total += delta * delta
    return sqrt(total)


@dataclass(frozen=True, slots=True)
class Decision:
    """一帧的识别结论。"""

    #: 时序共识后的标签；``None`` 表示这一帧没能达成共识。
    label: str | None
    #: 本帧最近邻的标签（未做时序平滑），用于 HUD 诊断。
    raw_label: str | None
    #: 到最近邻的距离，越小越像。
    distance: float
    #: 距离是否落在拒识阈值内。False 就是"我不认识这个姿势"。
    confident: bool
    #: 最近几帧是否达成共识。False 时 ``label`` 沿用上一次的稳定结论。
    stable: bool


@dataclass(slots=True)
class TemporalVote:
    """最近若干帧的标签投票箱。"""

    window: int = DEFAULT_WINDOW
    min_votes: int = DEFAULT_MIN_VOTES
    _recent: list[str | None] = field(default_factory=list)

    def reset(self) -> None:
        self._recent.clear()

    def push(self, label: str | None) -> str | None:
        """投一票，返回达成共识的标签；没够票返回 ``None``。

        ``None``（拒识）也占一个位置但不计入任何标签的票数——连续拒识会
        把窗口里真正有效票挤掉，从而让共识自然消失，这正是我们要的。
        """

        self._recent.append(label)
        if len(self._recent) > self.window:
            del self._recent[0]
        if label is None:
            return None
        votes = sum(1 for item in self._recent if item == label)
        return label if votes >= self.min_votes else None


class GestureClassifier:
    """在用户自己的样本库上做最近邻识别。"""

    def __init__(
        self,
        dataset: GestureDataset | None = None,
        *,
        k: int = DEFAULT_K,
        reject_distance: float = DEFAULT_REJECT_DISTANCE,
        window: int = DEFAULT_WINDOW,
        min_votes: int = DEFAULT_MIN_VOTES,
        pending_path=None,
        pending_interval_ns: int = DEFAULT_PENDING_INTERVAL_NS,
        coarse: bool = True,
        margin_ratio: float = DEFAULT_MARGIN_RATIO,
    ) -> None:
        self.k = max(1, k)
        self.coarse = coarse
        self.reject_distance = reject_distance
        self.margin_ratio = max(1.05, float(margin_ratio))
        self.pending_path = pending_path
        self.pending_interval_ns = pending_interval_ns
        self._dataset = GestureDataset()
        self._vote = TemporalVote(window=window, min_votes=min_votes)
        self._last_stable: str | None = None
        # 初值为 None：第一次遇到不确定帧无条件记录。若初始成 0，而运行期
        # 时间戳本来就很大，第一次会被节流逻辑误判成"刚记过"而丢掉。
        self._last_pending_ns: int | None = None
        if dataset is not None:
            self.fit(dataset)

    # ---- 装载 ----

    def fit(self, dataset: GestureDataset) -> int:
        """装入样本库，返回有效样本数。

        ``coarse=True`` 时按 :data:`COARSE_MAP` 把细分标签合并成
        point / open / fist 三类再训练——理由见 COARSE_MAP 的注释。
        """

        if self.coarse:
            self._dataset = GestureDataset(
                GestureSample(
                    label=COARSE_MAP.get(sample.label, sample.label),
                    features=sample.features,
                    captured_ns=sample.captured_ns,
                    source=sample.source,
                )
                for sample in dataset
            )
        else:
            self._dataset = dataset
        self._vote.reset()
        self._last_stable = None
        return len(self._dataset)

    @property
    def sample_count(self) -> int:
        return len(self._dataset)

    @property
    def labels(self) -> tuple[str, ...]:
        return self._dataset.labels_present()

    def is_ready(self, min_per_label: int = 20) -> bool:
        return self._dataset.is_ready(min_per_label)

    # ---- 识别 ----

    def raw_predict(self, features: Sequence[float]) -> tuple[str | None, float]:
        """返回 ``(k 近邻投票得到的标签, 最近邻距离)``。

        标签和距离必须来自**同一批样本**。旧实现拿"全部样本的距离倒数和"
        投票，却把"全局最近邻的距离"当置信度返回，样本数一不均就会出现
        自相矛盾的组合——2026-10-19 实测：只伸食指被判成 ``point``，
        同时距离报的却是 ``open`` 那 2.487，导致拒识判断用错了对象。
        """

        if not len(self._dataset):
            return None, float("inf")
        nearest: list[tuple[float, str]] = []
        for sample in self._dataset:
            distance = _distance(features, sample.features)
            if len(nearest) < self.k:
                nearest.append((distance, sample.label))
                nearest.sort(key=lambda item: item[0])
            elif distance < nearest[-1][0]:
                nearest[-1] = (distance, sample.label)
                nearest.sort(key=lambda item: item[0])
        nearest_distance = nearest[0][0]
        # 只在这 k 个最近邻里按距离倒数加权，避免样本多的类靠数量取胜。
        votes: dict[str, float] = {}
        for distance, label in nearest:
            votes[label] = votes.get(label, 0.0) + 1.0 / (distance + 1e-3)
        label = max(votes.items(), key=lambda item: item[1])[0]
        return label, nearest_distance

    def _runner_up_distance(
        self,
        features: Sequence[float],
        label: str,
    ) -> float:
        """到"最近的非 ``label`` 样本"的距离。

        用来判断这次识别是否**明确**：同一个标签内部可能包含好几种差异很大
        的姿势（用户实测"手指朝前"和"手指朝下"在特征空间是两团），这时
        到最近邻的绝对距离会被撑得很大，但"更像 A 而不是 B"这个相对关系
        依然成立。所以不能只看绝对距离。
        """

        best = float("inf")
        for sample in self._dataset:
            if sample.label == label:
                continue
            distance = _distance(features, sample.features)
            if distance < best:
                best = distance
        return best

    def predict(
        self,
        features: Sequence[float],
        *,
        timestamp_ns: int | None = None,
        collect_uncertain: bool = True,
    ) -> Decision:
        """识别一帧，含拒识、时序共识与不确定样本回流。"""

        now = timestamp_ns if timestamp_ns is not None else time.monotonic_ns()
        raw_label, distance = self.raw_predict(features)
        confident = distance <= self.reject_distance
        if not confident and raw_label is not None and distance > 1e-6:
            # 绝对距离大，但若明显更像当前类（领先第二类 margin_ratio 倍），
            # 仍然采信。这是"多峰样本库"下唯一能用的判据。
            runner_up = self._runner_up_distance(features, raw_label)
            if runner_up >= distance * self.margin_ratio:
                confident = True
        effective = raw_label if confident else None

        if not confident and collect_uncertain:
            self._collect_uncertain(features, raw_label, distance, now)

        agreed = self._vote.push(effective)
        if agreed is not None:
            self._last_stable = agreed
        return Decision(
            label=agreed if agreed is not None else self._last_stable,
            raw_label=raw_label,
            distance=distance,
            confident=confident,
            stable=agreed is not None,
        )

    # ---- 不确定样本回流 ----

    def _collect_uncertain(
        self,
        features: Sequence[float],
        raw_label: str | None,
        distance: float,
        now: int,
    ) -> None:
        """把拿不准的姿势存下来，攒够一批由用户补标。

        这是整个方案能"越用越准"的关键：模型不会猜，只会把不会的题记下来。
        """

        if self.pending_path is None:
            return
        if (
            self._last_pending_ns is not None
            and now - self._last_pending_ns < self.pending_interval_ns
        ):
            return
        self._last_pending_ns = now
        pending = GestureDataset()
        pending.add(
            GestureSample(
                label="unknown",
                features=tuple(features),
                captured_ns=now,
                source=f"nearest={raw_label or '-'}:{distance:.3f}",
            ),
        )
        pending.append(self.pending_path)

    def load_pending(self) -> GestureDataset:
        """读出待补标样本，供补标流程使用。"""

        if self.pending_path is None:
            return GestureDataset()
        return GestureDataset.load(self.pending_path)


def _median_vector(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """逐维取中位数，得到对离群点免疫的"类中心"。"""

    return tuple(
        statistics.median(vector[index] for vector in vectors)
        for index in range(FEATURE_DIM)
    )


def clean_dataset(
    dataset: GestureDataset,
    *,
    mad_factor: float = 4.0,
    max_rounds: int = 6,
    min_keep: int = 20,
) -> tuple[GestureDataset, dict[str, int]]:
    """按类剔除离群帧，返回 ``(清洗后的库, {标签: 丢弃帧数})``。

    2026-10-19 实测：641 帧里约三成是垃圾——手侧到接近平行于视线时，
    MediaPipe 的关键点本来就是猜的，特征离群到类内中位距离的几十倍。
    这些帧会把拒识阈值一路推到 1.45，连正常帧也被判成"不认识"。

    用 MAD（中位数绝对偏差）而不是标准差：个别飘掉的帧不会把清洗线
    自己拉高，所以可以反复迭代直到集合稳定。
    """

    kept: list[GestureSample] = []
    dropped: dict[str, int] = {}
    for label in dataset.labels_present():
        samples = dataset.for_label(label)
        keep = list(samples)
        if len(keep) > min_keep:
            for _ in range(max_rounds):
                center = _median_vector([sample.features for sample in keep])
                distances = [_distance(sample.features, center) for sample in keep]
                median = statistics.median(distances)
                mad = statistics.median(abs(value - median) for value in distances)
                cut = median + mad_factor * max(mad, 1e-3)
                survived = [
                    sample
                    for sample, value in zip(keep, distances)
                    if value <= cut
                ]
                if len(survived) == len(keep) or len(survived) < min_keep:
                    break
                keep = survived
        dropped[label] = len(samples) - len(keep)
        kept.extend(keep)
    return GestureDataset(kept), dropped


def suggest_reject_distance(
    dataset: GestureDataset,
    *,
    keep_ratio: float = 0.95,
) -> float:
    """用留一法给一个"尽量少误判"的拒识阈值。

    对每个样本，找**除自己以外**的最近邻：

    * 最近邻同类 → 判对，记下这次距离；
    * 最近邻异类 → 判错。

    取"判对的那些距离"的 ``keep_ratio`` 分位数当阈值。也就是说默认保留
    95% 的正确识别，剩下 5% 宁可拒识也不乱猜——用户明确说过误触比没反应
    更烦，所以这个方向是保守的。
    """

    samples = dataset.samples
    if len(samples) < 2:
        return DEFAULT_REJECT_DISTANCE
    correct: list[float] = []
    for index, sample in enumerate(samples):
        best_distance = float("inf")
        best_label: str | None = None
        for other_index, other in enumerate(samples):
            if other_index == index:
                continue
            distance = _distance(sample.features, other.features)
            if distance < best_distance:
                best_distance = distance
                best_label = other.label
        if best_label == sample.label:
            correct.append(best_distance)
    if not correct:
        return DEFAULT_REJECT_DISTANCE
    correct.sort()
    position = int(len(correct) * keep_ratio)
    position = min(position, len(correct) - 1)
    return max(correct[position], 1e-3)


def _leave_one_out(dataset: GestureDataset) -> Iterator[tuple[str, str, float]]:
    """逐个样本找出"除自己以外的最近邻"，产出 ``(真实标签, 预测标签, 距离)``。"""

    samples = dataset.samples
    for index, sample in enumerate(samples):
        best_distance = float("inf")
        best_label = ""
        for other_index, other in enumerate(samples):
            if other_index == index:
                continue
            distance = _distance(sample.features, other.features)
            if distance < best_distance:
                best_distance = distance
                best_label = other.label
        yield sample.label, best_label, best_distance


def evaluate(
    dataset: GestureDataset,
    reject_distance: float,
) -> dict[str, float]:
    """留一法粗评：正确率 / 误判率 / 拒识率。用于采集后自检。"""

    total = 0
    correct = wrong = rejected = 0
    for truth, predicted, distance in _leave_one_out(dataset):
        total += 1
        if distance > reject_distance:
            rejected += 1
        elif predicted == truth:
            correct += 1
        else:
            wrong += 1
    if not total:
        return {"total": 0.0, "correct": 0.0, "wrong": 0.0, "rejected": 0.0}
    return {
        "total": float(total),
        "correct": correct / total,
        "wrong": wrong / total,
        "rejected": rejected / total,
    }


#: 混淆矩阵里代表"拒识"的键。
REJECTED = "?"


def confusion(
    dataset: GestureDataset,
    reject_distance: float,
) -> dict[str, dict[str, int]]:
    """留一法混淆矩阵：``matrix[真实标签][预测标签] = 帧数``。

    预测为 :data:`REJECTED` 表示这一帧被拒识（程序回答"不认识"），
    不属于混淆，不计入冲突。
    """

    matrix: dict[str, dict[str, int]] = {}
    for truth, predicted, distance in _leave_one_out(dataset):
        bucket = matrix.setdefault(truth, {})
        key = REJECTED if distance > reject_distance else predicted
        bucket[key] = bucket.get(key, 0) + 1
    return matrix


def conflicts(
    dataset: GestureDataset,
    reject_distance: float,
    *,
    top: int = 3,
) -> list[tuple[str, str, int]]:
    """找出最容易互相混淆的手势对，返回 ``[(a, b, 混淆帧数)]``。

    这是给**用户裁定**用的，不是给程序自己猜的：两种手势长得太像时，程序
    没有立场决定谁对谁错——可能是用户做的时候没做到位，也可能这两个手势
    本来就太接近、该换个更夸张的姿势。把数字摆出来，一句话就能解决。
    """

    matrix = confusion(dataset, reject_distance)
    merged: dict[tuple[str, str], int] = {}
    for truth, bucket in matrix.items():
        for predicted, count in bucket.items():
            if predicted == REJECTED or predicted == truth:
                continue
            key = (truth, predicted) if truth < predicted else (predicted, truth)
            merged[key] = merged.get(key, 0) + count
    ranked = sorted(merged.items(), key=lambda item: item[1], reverse=True)
    return [(left, right, count) for (left, right), count in ranked[:top]]
