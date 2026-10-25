"""个人手势采集与识别（gesture_data / gesture_model）的测试。"""

from __future__ import annotations

import pytest

from conftest import (
    entry_palm_points,
    fist_points,
    left_pinch_points,
    observation,
    open_palm_points,
    pointer_points,
    swipe_points,
    thumb_open_pointer_points,
    translate_points,
)
from handmouse.contracts import Landmark, MouseCommandType
from handmouse.gesture_data import (
    COVERAGE_BINS,
    FEATURE_DIM,
    CoverageTracker,
    GestureDataset,
    normalize,
    palm_width,
    wrist_rotation_ratio,
)
from handmouse.config import AppConfig
from handmouse.gestures import GestureEngine, _is_five_open
from handmouse.gesture_model import (
    COARSE_LABELS,
    GestureClassifier,
    clean_dataset,
    conflicts,
    evaluate,
    suggest_reject_distance,
)


def _full_dataset(tmp_path, move_points=open_palm_points):
    """五种手势各 22 帧的样本库。

    ``move`` 刻意用张掌形状：模拟真实故障——用户的手在几何上被判成
    五指张开，但采集时他把它标成了"移动"。
    """

    dataset = GestureDataset()
    makers = {
        "move": move_points,
        "stop": thumb_open_pointer_points,
        "pinch": left_pinch_points,
        "open": entry_palm_points,
        "fist": fist_points,
    }
    for label, maker in makers.items():
        for step in range(22):
            dataset.add_frame(label, translate_points(maker(), dx=step * 0.002))
    path = tmp_path / "samples.jsonl"
    dataset.save(path)
    return path


def _engine_with_model(path) -> GestureEngine:
    config = AppConfig(
        gesture_model_enabled=True,
        gesture_model_path=str(path),
    )
    return GestureEngine(config)


def test_engine_reports_model_state_when_enabled(tmp_path) -> None:
    engine = _engine_with_model(_full_dataset(tmp_path))
    assert "帧" in engine.gesture_model_note


def test_engine_degrades_quietly_when_library_missing(tmp_path) -> None:
    """没采过样本就该老老实实走几何判据，不能报错、不能行为突变。"""

    engine = _engine_with_model(tmp_path / "nope.jsonl")
    # 要能在 HUD 上说清"为什么没生效"，所以提示语非空，但行为必须回退。
    assert "样本不足" in engine.gesture_model_note
    commands = engine.process(observation(pointer_points(), BASE))
    assert [command.kind.value for command in commands] == ["move"]


def test_model_label_needs_consecutive_frames(tmp_path) -> None:
    """单帧不采信，连着两帧一样才算数——抖动就是这么滤掉的。"""

    engine = _engine_with_model(_full_dataset(tmp_path))
    assert engine._model_label(open_palm_points(), BASE) is None
    # 引擎里用的是粗分类：采样时标成 move 的样本被合并进 point。
    assert engine._model_label(open_palm_points(), BASE + 1_000) == "point"


def test_model_label_returns_none_when_frame_is_degenerate(tmp_path) -> None:
    engine = _engine_with_model(_full_dataset(tmp_path))
    collapsed = tuple(Landmark(0.5, 0.5) for _ in range(21))
    assert engine._model_label(collapsed, BASE) is None


def test_reject_never_falls_back_to_geometry_open(tmp_path) -> None:
    """模型没把握时，几何判据说"五指张开"也不能采信。

    2026-10-19 用户实测：只伸食指，HUD 却显示「五指张开（待机）」，
    食指怎么动光标都不动。根因是模型拒识后回退几何判据，而几何判据在
    侧手投影下把只伸食指判成了五指张开。误判成待机 = 光标卡死，
    所以没把握时必须按"不是五指张开"处理。
    """

    path = _full_dataset(tmp_path)
    points = open_palm_points()
    assert _is_five_open(points)  # 前提：几何判据确实会误报

    engine = GestureEngine(
        AppConfig(
            gesture_model_enabled=True,
            gesture_model_path=str(path),
            # 阈值极小 → 任何输入都被拒识，逼出"模型无话可说"的路径。
            gesture_model_reject_distance=0.001,
        ),
    )
    for step in range(4):
        engine.process(observation(points, BASE + step * 60_000_000))

    assert engine.current_gesture.value != "open_palm"
    assert engine.current_gesture.value == "pointer"


def test_geometry_still_used_when_library_disabled() -> None:
    """没采集样本的用户，行为必须和以前完全一致。"""

    engine = GestureEngine(AppConfig())
    engine.process(observation(open_palm_points(), BASE))
    # 2026-10-22 起"进待机"要连续保持 FIVE_OPEN_CONFIRM_MS(300ms)。
    engine.process(observation(open_palm_points(), BASE + 350_000_000))
    assert engine.current_gesture.value == "open_palm"


def test_samples_suppress_geometry_misfire(tmp_path) -> None:
    """核心回归：几何判据说"五指张开"，样本库说"移动"，必须听样本库的。

    这正是滚轮模式修三轮修不好的场景——用户的手在侧角度下被几何规则
    误判。有了自己的样本后，同样的输入应该老老实实当移动处理。
    """

    points = open_palm_points()
    assert _is_five_open(points)  # 前提：几何规则确实会误报

    engine = _engine_with_model(_full_dataset(tmp_path))
    for step in range(4):
        engine.process(observation(points, BASE + step * 60_000_000))

    assert engine.current_gesture.value == "pointer"


BASE = 1_000_000_000


def _two_class_dataset(per_class: int = 6) -> GestureDataset:
    """两类形状明显不同的样本：指向 vs 握拳。"""

    dataset = GestureDataset()
    for step in range(per_class):
        dataset.add_frame(
            "move",
            translate_points(pointer_points(), dx=step * 0.004),
            captured_ns=BASE + step,
        )
        dataset.add_frame(
            "fist",
            translate_points(fist_points(), dx=step * 0.004),
            captured_ns=BASE + step,
        )
    return dataset


# ---- 特征归一化 ----


def test_normalize_has_expected_dimension() -> None:
    features = normalize(pointer_points())
    assert features is not None
    assert len(features) == FEATURE_DIM


def test_normalize_is_invariant_to_translation() -> None:
    """手在画面里左右上下移动，特征必须完全不变。"""

    base = normalize(pointer_points())
    moved = normalize(translate_points(pointer_points(), dx=0.3, dy=-0.2))
    assert base is not None and moved is not None
    for left, right in zip(base, moved):
        assert left == pytest.approx(right, abs=1e-9)


def test_normalize_is_invariant_to_distance_from_camera() -> None:
    """手离摄像头远近变化（整体缩放），特征必须不变。"""

    base_points = pointer_points()
    scaled = tuple(
        Landmark(0.5 + (point.x - 0.5) * 0.5, 0.5 + (point.y - 0.5) * 0.5, point.z)
        for point in base_points
    )
    base = normalize(base_points)
    shrunk = normalize(scaled)
    assert base is not None and shrunk is not None
    for left, right in zip(base, shrunk):
        assert left == pytest.approx(right, abs=1e-6)


def test_normalize_rejects_incomplete_or_exploded_frames() -> None:
    assert normalize(pointer_points()[:10]) is None
    # 关键点飘了：某一个指尖离手腕远到 20 倍掌宽，正常手绝不会这样。
    points = list(pointer_points())
    width = palm_width(points)
    points[8] = Landmark(points[0].x + 20.0 * width, points[0].y)
    assert normalize(tuple(points)) is None


def test_wrist_rotation_ratio_shrinks_when_palm_is_sideways() -> None:
    """手掌侧过来时掌宽被压扁，比值必须下降——角度覆盖靠这个量分档。"""

    frontal = wrist_rotation_ratio(open_palm_points())
    sideways = tuple(
        Landmark(0.5 + (point.x - 0.5) * 0.15, point.y, point.z)
        for point in open_palm_points()
    )
    assert wrist_rotation_ratio(sideways) < frontal
    assert frontal > 0.0


# ---- 角度覆盖度 ----


def test_coverage_tracker_counts_distinct_angles() -> None:
    tracker = CoverageTracker()
    assert tracker.covered == 0
    tracker.mark(0.3)
    tracker.mark(0.31)
    assert tracker.covered == 1
    tracker.mark(1.2)
    assert tracker.covered == 2
    assert not tracker.complete
    assert f"{tracker.covered}/{COVERAGE_BINS}" in tracker.hint()


def test_coverage_tracker_completes_when_all_bins_seen() -> None:
    tracker = CoverageTracker(bins=4)
    for step in range(4):
        tracker.mark(0.25 + step * 0.25)
    assert tracker.complete
    assert tracker.hint() == "角度覆盖已足够"


# ---- 样本库 ----


def test_dataset_add_frame_skips_unusable_frames() -> None:
    dataset = GestureDataset()
    assert dataset.add_frame("move", pointer_points()) is True
    assert dataset.add_frame("move", pointer_points()[:8]) is False
    assert len(dataset) == 1


def test_dataset_stores_world_landmarks_for_offline_3d_replay(tmp_path) -> None:
    """采集时要存下米制 3D 坐标，否则任何 3D 判据永远测不了。

    40 维 ``features`` 只有 x/y，旧样本上**无法离线重算任何涉及深度的判据**，
    只能重采。把 ``world`` 存进去，以后换任何 3D 方案都能用旧数据复核。
    """

    from conftest import world_hand

    world = world_hand()
    dataset = GestureDataset()
    assert dataset.add_frame("swipe", pointer_points(), world=world) is True

    path = tmp_path / "samples.jsonl"
    dataset.save(path)
    loaded = GestureDataset.load(path)
    restored = loaded.samples[0].world
    assert len(restored) == 63, "3D 坐标没有落盘"
    assert restored[0] == pytest.approx(world[0].x, abs=1e-4)
    assert restored[2] == pytest.approx(world[0].z, abs=1e-4)


def test_old_samples_without_world_still_load(tmp_path) -> None:
    """老样本没有 ``world`` 字段，必须照常加载（向后兼容）。"""

    dataset = GestureDataset()
    dataset.add_frame("move", pointer_points())
    path = tmp_path / "samples.jsonl"
    dataset.save(path)
    loaded = GestureDataset.load(path)
    assert len(loaded) == 1
    assert loaded.samples[0].world == ()


def test_dataset_roundtrip(tmp_path) -> None:
    dataset = _two_class_dataset()
    path = tmp_path / "samples.jsonl"
    assert dataset.save(path) == len(dataset)
    loaded = GestureDataset.load(path)
    assert len(loaded) == len(dataset)
    assert loaded.counts() == dataset.counts()
    assert loaded.samples[0].features == pytest.approx(
        dataset.samples[0].features,
        abs=1e-5,
    )


def test_dataset_load_ignores_broken_lines(tmp_path) -> None:
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"label": "move", "features": [1, 2]}\nnot-json\n\n'
        '{"label": "fist"}\n',
        encoding="utf-8",
    )
    assert len(GestureDataset.load(path)) == 0


def test_dataset_is_ready_checks_only_present_labels() -> None:
    """is_ready 只要求"已采集到的"每类都够样本，而不是全部 LABELS 都齐。

    这样两个需求同时满足：
    * 早期只采了 5 类、后来新增第 6 类 swipe 的旧样本库仍算就绪
      （否则已训练好的模型会突然加载失败）；
    * 只补采某一类（如 --only swipe）后，那一类自己就能达到就绪阈值。
    """

    # 只采了 move 一类，且够 20 帧 → 对"已采集标签"而言就绪。
    dataset = GestureDataset()
    for _ in range(30):
        dataset.add_frame("move", pointer_points())
    assert dataset.is_ready()

    # 但只要有任意一类已采集却不足 20 帧，就不就绪。
    small = GestureDataset()
    for _ in range(30):
        small.add_frame("move", pointer_points())
    for _ in range(5):
        small.add_frame("swipe", fist_points())
    assert not small.is_ready()

    # 空库不就绪。
    assert not GestureDataset().is_ready()


# ---- 识别 ----


def test_classifier_picks_the_nearest_label() -> None:
    dataset = _two_class_dataset()
    classifier = GestureClassifier(dataset, reject_distance=10.0, coarse=False)
    features = normalize(pointer_points())
    assert features is not None
    label, distance = classifier.raw_predict(features)
    assert label == "move"
    assert distance == pytest.approx(0.0, abs=1e-6)


def test_classifier_rejects_unfamiliar_pose() -> None:
    """核心保证：不像任何样本时必须说"不认识"，而不是硬塞进某一类。"""

    dataset = _two_class_dataset()
    classifier = GestureClassifier(dataset, reject_distance=0.5)
    faraway = tuple(3.0 for _ in range(FEATURE_DIM))
    label, distance = classifier.raw_predict(faraway)
    assert distance > 0.5
    decision = classifier.predict(faraway, timestamp_ns=BASE)
    assert decision.confident is False
    assert decision.raw_label == label


def test_temporal_vote_requires_repeated_frames() -> None:
    """单帧不作数，连续两帧一致才切换——单帧抖动就是这么消掉的。"""

    dataset = _two_class_dataset()
    classifier = GestureClassifier(
        dataset,
        reject_distance=10.0,
        window=3,
        min_votes=2,
        coarse=False,
    )
    features = normalize(pointer_points())
    assert features is not None
    first = classifier.predict(features, timestamp_ns=BASE)
    assert first.stable is False
    assert first.label is None
    second = classifier.predict(features, timestamp_ns=BASE + 1000)
    assert second.stable is True
    assert second.label == "move"


def test_temporal_vote_holds_last_stable_when_uncertain() -> None:
    """中间夹一帧别的标签不该让状态闪断。"""

    dataset = _two_class_dataset()
    classifier = GestureClassifier(
        dataset,
        reject_distance=10.0,
        window=3,
        min_votes=2,
        coarse=False,
    )
    move = normalize(pointer_points())
    other = normalize(fist_points())
    assert move is not None and other is not None
    classifier.predict(move, timestamp_ns=BASE)
    classifier.predict(move, timestamp_ns=BASE + 1000)
    blip = classifier.predict(other, timestamp_ns=BASE + 2000)
    assert blip.stable is False
    assert blip.label == "move"


def test_uncertain_frames_are_collected_for_review(tmp_path) -> None:
    """拿不准的姿势要落盘待补标，这是"越用越准"的来源。"""

    dataset = _two_class_dataset()
    pending = tmp_path / "pending.jsonl"
    classifier = GestureClassifier(
        dataset,
        reject_distance=0.5,
        pending_path=pending,
        pending_interval_ns=0,
    )
    faraway = tuple(3.0 for _ in range(FEATURE_DIM))
    classifier.predict(faraway, timestamp_ns=BASE)
    loaded = classifier.load_pending()
    assert len(loaded) == 1
    assert loaded.samples[0].label == "unknown"


def test_uncertain_collection_is_throttled(tmp_path) -> None:
    dataset = _two_class_dataset()
    pending = tmp_path / "pending.jsonl"
    classifier = GestureClassifier(
        dataset,
        reject_distance=0.5,
        pending_path=pending,
        pending_interval_ns=10_000_000_000,
    )
    faraway = tuple(3.0 for _ in range(FEATURE_DIM))
    classifier.predict(faraway, timestamp_ns=BASE)
    classifier.predict(faraway, timestamp_ns=BASE + 1000)
    assert len(classifier.load_pending()) == 1


# ---- 自检：阈值与冲突 ----


def test_suggest_reject_distance_keeps_most_correct_samples() -> None:
    dataset = _two_class_dataset(per_class=8)
    threshold = suggest_reject_distance(dataset, keep_ratio=0.95)
    stats = evaluate(dataset, threshold)
    assert stats["wrong"] == 0.0
    assert stats["correct"] >= 0.9


def test_conflicts_reported_when_two_labels_are_identical() -> None:
    """两类样本完全一样时，必须报告冲突，交给人裁定。"""

    dataset = GestureDataset()
    for _ in range(8):
        dataset.add_frame("move", pointer_points())
        dataset.add_frame("stop", pointer_points())
    pairs = conflicts(dataset, reject_distance=10.0)
    assert pairs
    left, right, count = pairs[0]
    assert {left, right} == {"move", "stop"}
    assert count > 0


def test_no_conflicts_when_labels_are_distinct() -> None:
    dataset = _two_class_dataset(per_class=8)
    assert conflicts(dataset, reject_distance=10.0) == []


# ---- 双指滑动（swipe）采集与识别 ----


def test_classifier_separates_swipe_from_point() -> None:
    """模型能把"双指伸开"和"只伸食指"分开，而不是把双指误判成指向。

    这两类在特征上的唯一差别是中指那一根，正是用户实测最容易被混的点。
    """

    dataset = GestureDataset()
    for step in range(14):
        dataset.add_frame(
            "move",
            translate_points(pointer_points(), dx=step * 0.004),
            captured_ns=BASE + step,
        )
        dataset.add_frame(
            "swipe",
            translate_points(swipe_points(), dx=step * 0.004),
            captured_ns=BASE + 1000 + step,
        )
    # coarse=True：move→point，swipe→swipe（见 gesture_model.COARSE_MAP）。
    classifier = GestureClassifier(dataset, reject_distance=10.0, coarse=True)
    # 连续喂同一帧，模拟真实运行时时序共识收敛（需 2 帧共识）。
    swipe_label = None
    for _ in range(4):
        swipe_label = classifier.predict(normalize(swipe_points())).label
    assert swipe_label == "swipe"
    point_label = None
    for _ in range(4):
        point_label = classifier.predict(normalize(pointer_points())).label
    assert point_label == "point"


def test_model_cannot_override_geometry_core_gate(monkeypatch) -> None:
    """模型票**不能绕过核心闸**：食指/中指量出来不够长，模型说什么都不算。

    2026-10-22 第五次修订把旧的三级信任（深信区直通 / 犹豫区过几何闸）
    整体撤掉了。原因有三条，都是实测换来的：

    1. 旧几何闸 ÷ 掌宽 ``reach>=1.3`` **只在侧手成立**，正手永远过不去，
       于是实际起作用的就只剩"模型直通"——而模型在指向姿势上会自信地
       错（swipe@0.94 / @0.91，都 < 拒识阈值）。
    2. 现在的几何判据是标度不变的，正手/侧手都能认（他 3378 帧真实样本上
       可用度 80.7%），不需要模型来补洞。
    3. 用户定的铁律："模型永远无权推翻几何尺子"。所以模型只剩**一处**
       合法作用域：核心闸已成立时，把"无名指/小指收拢"这条放宽
       （见 ``test_model_can_relax_curl_gate_only``）。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    monkeypatch.setattr(engine, "_model_label", lambda points, now: "swipe")

    # 食指很长、中指投影只有食指尖的 55%（tm/peak≈0.55 < 0.70）：核心闸不认。
    points = list(pointer_points())
    points[8] = Landmark(0.42, -0.05)
    points[10] = Landmark(0.50, 0.49)
    points[11] = Landmark(0.50, 0.37)
    points[12] = Landmark(0.50, 0.26)

    engine.process(observation(tuple(points), BASE))
    assert engine.current_gesture.value != "swipe", "模型绕过了几何核心闸"


def test_model_can_relax_curl_gate_only(monkeypatch) -> None:
    """模型的**唯一**合法作用域：核心闸成立时放宽"无名指/小指收拢"。

    正常进入要求无名指、小指都 ≤0.65×最长指尖距离；模型报 swipe 时可以
    放宽到 0.78——因为这两根手指在侧视下投影天然偏长，而食指+中指
    是否伸直已经由几何量死，模型放宽它不会把指向姿势变成滚动。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)

    from conftest import swipe_points

    points = list(swipe_points())
    # 无名指稍微被撑直：tr 落在 0.65~0.78 之间，几何不认、模型能补。
    points[16] = Landmark(0.58, 0.33)
    relaxed = tuple(points)

    plain = GestureEngine(config)
    plain.set_paused(False)
    plain.process(observation(relaxed, BASE))
    assert plain.current_gesture.value != "swipe", "几何本不该放行这帧（夹具前提）"

    monkeypatch.setattr(engine, "_model_label", lambda p, now: "swipe")
    engine.process(observation(relaxed, BASE))
    engine.process(observation(relaxed, BASE + 40_000_000))
    assert engine.current_gesture.value == "swipe", "模型没有补上收拢闸"


def test_model_swipe_vote_vetoed_when_middle_folded(monkeypatch) -> None:
    """只伸食指（中指收拢）时，模型就算报 swipe 也必须被几何闸否决。

    2026-10-22 用户实机截图×2：只伸食指想移动光标，HUD 模型行先后显示
    ``swipe@0.94``、``swipe@0.91``（都 < 拒识阈值 0.98）——补采的双指
    样本和指向姿势在特征空间部分重叠，OR 直通就把指向强行判成滑动：
    光标冻结（用户说的"卡"）、pointer↔swipe 反复横跳、HUD 出现幻影
    滑动识别。现在模型的 swipe 票必须过核心闸与收拢闸，中指收拢时一律
    否决，照常进 pointer 移动。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    monkeypatch.setattr(engine, "_model_label", lambda points, now: "swipe")

    commands = engine.process(observation(pointer_points(), BASE))

    assert engine.current_gesture.value == "pointer"
    assert MouseCommandType.MOVE in [c.kind for c in commands]


def test_model_swipe_vote_vetoed_when_middle_half_extended(monkeypatch) -> None:
    """中指半伸（指尖刚过掌指关节一点）时模型票同样无效。

    这类姿势的指尖到掌心距离只有食指的 ~45%（``t_m/peak ≈ 0.45``），
    但**关节角是 153°**——角度判据单独用会放行。所以进入判据必须以
    标度不变的长度比为主、关节角只做正交兜底（防"折 90° 往侧面翘"
    那种指尖依然远的姿势），两条都不成立才算不是双指。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    monkeypatch.setattr(engine, "_model_label", lambda points, now: "swipe")

    points = list(pointer_points())
    points[10] = Landmark(0.50, 0.48)
    points[11] = Landmark(0.51, 0.46)
    points[12] = Landmark(0.52, 0.44)

    commands = engine.process(observation(tuple(points), BASE))

    assert engine.current_gesture.value == "pointer"
    assert MouseCommandType.MOVE in [c.kind for c in commands]


def test_model_swipe_vote_vetoed_when_middle_folds_at_pip(monkeypatch) -> None:
    """中指在 PIP 处折 90° 往侧面翘时，模型票同样无效（2026-10-22 14:17 事故）。

    用户原话："我用这个角度向右划就会触发……一只手指为什么也能触发"。
    这个姿势的**指尖离掌心很远**（``t_m/peak`` 能到 0.89），任何只看长度
    的判据都会放行；只有关节角能拦下它。所以核心闸里必须带
    ``a_m ≥ 110°`` 这一条，且模型无权绕过。
    """

    config = AppConfig()
    config.entry_gate_enabled = False
    engine = GestureEngine(config)
    engine.set_paused(False)
    monkeypatch.setattr(engine, "_model_label", lambda points, now: "swipe")

    points = list(pointer_points())
    points[10] = Landmark(0.50, 0.32)
    points[11] = Landmark(0.64, 0.32)
    points[12] = Landmark(0.78, 0.32)

    commands = engine.process(observation(tuple(points), BASE))

    assert engine.current_gesture.value == "pointer"
    assert MouseCommandType.MOVE in [c.kind for c in commands]


def test_swipe_label_is_collectable() -> None:
    """swipe 已进入标签集与采集步骤，采集管线能接收这个标签。"""

    from handmouse.collect import STEPS
    from handmouse.gesture_data import LABELS, LABEL_NAMES

    assert "swipe" in LABELS
    assert LABEL_NAMES["swipe"]
    assert any(step.label == "swipe" for step in STEPS)


def test_labels_present_includes_coarse_only_label() -> None:
    """粗分类 ``point`` 不在 ``LABELS`` 里，但必须算「已采集」。

    2026-10-23 修的真 bug：``labels_present()`` 只遍历写死的 ``LABELS``
    六类细分标签，``point`` 不在其中 → 粗分类数据集 ``is_ready()`` 恒 False、
    ``clean_dataset()`` 把 ``point`` 整类跳过（实测 739 帧 / 83%），
    而且**不报任何错**。生产流程清洗的是细分类数据集，所以一直没被触发。
    """

    assert "point" in COARSE_LABELS
    dataset = GestureDataset()
    for step in range(22):
        dataset.add_frame("point", translate_points(pointer_points(), dx=step * 0.002))

    assert dataset.labels_present() == ("point",)
    assert dataset.is_ready(min_per_label=20)


def test_labels_present_keeps_fixed_order_then_extras() -> None:
    """已知标签按 ``LABELS`` 的固定顺序在前，未知标签按名字排在后面。

    顺序不能依赖 dict 插入序：采集顺序变了就会让界面上的标签顺序跳动。
    """

    dataset = GestureDataset()
    for _ in range(2):
        dataset.add_frame("point", pointer_points())
        dataset.add_frame("open", open_palm_points())
        dataset.add_frame("move", pointer_points())

    assert dataset.labels_present() == ("move", "open", "point")


def test_clean_dataset_does_not_skip_coarse_label() -> None:
    """清洗必须真的跑到 ``point`` 这一类（以前是整类跳过，离群帧全留着）。"""

    dataset = GestureDataset()
    for step in range(22):
        dataset.add_frame("point", translate_points(pointer_points(), dx=step * 0.002))
    # 一帧明显飘掉的关键点：食指指尖甩到手外。
    # ⚠️ 不能用 `translate_points(dx=40)` 造离群帧——归一化先减手腕、再除手长，
    # **平移与整体缩放都是不变的**，那样造出来的帧和正常帧一模一样。
    outlier = list(pointer_points())
    outlier[8] = Landmark(outlier[0].x + 0.30, outlier[0].y + 0.30)
    dataset.add_frame("point", tuple(outlier))

    cleaned, dropped = clean_dataset(dataset)

    assert dropped.get("point", 0) >= 1
    assert len(cleaned.for_label("point")) >= 20

