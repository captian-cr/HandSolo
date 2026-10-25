"""引导式手势采集：把用户自己的手录成模板库。

用法
----
::

    python -m handmouse.collect                 # 六种手势各录一遍
    python -m handmouse.collect --append        # 在已有库上追加样本
    python -m handmouse.collect --only swipe    # 只补采双指
    python -m handmouse.collect --seconds 12    # 每种录久一点

采集方式：按你的真实习惯录，而不是按脚本表演
----------------------------------------------
程序一次只让你用一种手势，**不规定你做什么动作**——你平时怎么用就怎么用，
它整段逐帧记录。这样采到的是你真实的使用分布（手放哪、怎么微动、习惯停在
什么角度），而不是配合脚本摆出来的姿势。

唯一会自动介入的是**角度覆盖**：用户的手在侧角度下收拢的指节投影会近似
共线，通用几何判据彻底失效（滚轮模式修三轮修不好就是这个原因）。所以样本
必须覆盖各种手腕角度。程序盯着掌宽压缩比分成 8 档，**只有发现哪个角度没
采到，才提示你换个角度**；角度够了就不打扰你。

采集**不会移动鼠标**：只开摄像头与推理线程，不接 GestureEngine。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from handmouse import APP_NAME, debug_view
from handmouse.config import AppConfig, ConfigStore, PROJECT_ROOT
from handmouse.contracts import FramePacket, DetectionPacket
from handmouse.gesture_data import (
    COVERAGE_BINS,
    LABEL_NAMES,
    LABELS,
    CoverageTracker,
    GestureDataset,
    GestureSample,
    default_dataset_path,
    wrist_rotation_ratio,
)
from handmouse.gesture_model import (
    COARSE_MAP,
    COARSE_NAMES,
    clean_dataset,
    conflicts,
    evaluate,
    suggest_reject_distance,
)
from handmouse.latest_slot import LatestSlot
from handmouse.vision import CameraWorker, HandLandmarkerWorker, default_model_path


@dataclass(frozen=True)
class CollectStep:
    """一个手势的采集阶段。"""

    label: str
    #: 摆好姿势的准备时间（不录）。
    prep_seconds: float
    #: 边转手腕边录的时间。
    record_seconds: float
    #: 屏幕提示语。
    cue: str


#: 第一版手势集。提示语刻意不规定动作：让程序去学用户平时怎么用，
#: 而不是让用户配合脚本。角度覆盖由 CoverageTracker 在缺档时才提醒。
#:
#: 双指（swipe）的采集**只录静止姿势**：食指+中指伸开、无名指与小指握起，
#: 边转手腕边录。滑动的"方向"由运行时几何逻辑根据掌心位移判断，不在采集
#: 范围内——所以提示语特意写"不用真的滑动"，避免录进一堆过渡帧。
STEPS: tuple[CollectStep, ...] = (
    CollectStep("move", 2.5, 9.0, "就像平时操控光标那样，用【移动】手势随便动一动"),
    CollectStep("stop", 2.5, 9.0, "用【停住】手势，保持不动即可"),
    CollectStep("pinch", 2.5, 9.0, "反复捏合、松开，像平时点击那样"),
    CollectStep("open", 2.5, 9.0, "五指张开，随意保持"),
    CollectStep("fist", 2.5, 9.0, "握拳，随意保持"),
    CollectStep(
        "swipe",
        2.5,
        12.0,
        "保持【食指+中指伸开、其余握拳】，缓慢转动手腕、上下换换手的位置"
        "（不用真的滑动，只录姿势）",
    ),
)

#: 首次向导用的练习集：同样的七种手势，但时间短一半。
#: 理由：向导是"教一遍 + 认一遍"，不是"录够样本"；9 秒×6 会让新用户坐到烦躁。
#: 想补足样本，向导结束时会提示跑 `--append`。
PRACTICE_STEPS: tuple[CollectStep, ...] = tuple(
    CollectStep(step.label, 2.0, 5.0, step.cue) for step in STEPS
)

#: 🔴 双指（swipe）必须**分多次采集**（2026-10-22 实机教训）：用户一次只录
#: 9 秒（219 帧 ≈ 一个手腕角度），结果"手指几乎指向镜头"的极端侧角在特征
#: 空间里离整个样本库最近都有 1.56（拒识线 0.98）——模型等于从没见过这个
#: 姿势，分级信任也救不了（深信区需要 ≤0.8）。2D 投影在这种角度把伸直的
#: 手指压短，骨架几何判据同样集体失效。
#: 第二次实机教训（同日下午）：手放低到画面下方也会失灵（模型 @1.14 > 0.99）
#: ——摄像头在屏幕上方，手越低相机越俯视，掌面更侧、手指指向变横，又是
#: 一副没见过的模样。**要换的维度有两个：手腕角度 + 手在画面里的位置。**
_SWIPE_MULTI_ANGLE_HINT = (
    "\n[提示] 双指滑动建议再补采 2~3 遍：每次运行同一命令前，先换一个条件再录，"
    "\n       样本库会自动追加合并。要换的是两个维度——"
    "\n       ① 手腕角度：正面 → 斜 45° → 侧到手指快指向镜头；"
    "\n       ② 手的位置：画面中部 → 手放低到画面下方（低处手掌对着摄像头"
    "\n       更侧，是另一副模样，也得让模型见过）。"
    "\n       覆盖越全，模型在你真实使用的每个姿势都能认出你。"
)

#: 同 debug_view.WINDOW_NAME：cv2 窗口标题在 Windows 上按 GBK 解释 UTF-8，
#: 中文必然乱码，只能用 ASCII。
_WINDOW = f"{APP_NAME} - Collect"
_ABORT_KEYS = (27, ord("q"), ord("Q"))


def _open_workers(config: AppConfig):
    """启动摄像头与推理线程，返回控制句柄。不动鼠标。"""

    stop_event = threading.Event()
    frames: LatestSlot[FramePacket] = LatestSlot()
    detections: LatestSlot[DetectionPacket] = LatestSlot()
    errors: list[str] = []

    def on_error(message: str) -> None:
        errors.append(message)
        stop_event.set()

    camera = CameraWorker(
        config,
        frames,
        stop_event,
        on_error=on_error,
        on_frame=lambda _packet: None,
    )
    landmarker = HandLandmarkerWorker(
        config,
        default_model_path(),
        frames,
        detections,
        stop_event,
        on_error=on_error,
        on_detection=lambda _packet: None,
        paused_provider=lambda: False,
    )
    camera.start()
    landmarker.start()
    return stop_event, frames, detections, errors, (camera, landmarker)


def _blank_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _run_phase(
    *,
    frames: LatestSlot[FramePacket],
    detections: LatestSlot[DetectionPacket],
    seconds: float,
    rows_fn,
    on_new_detection=None,
) -> bool:
    """跑一个阶段（准备或录制），返回是否被用户中断。

    ``rows_fn(remaining)`` 返回要显示的 HUD 行；``on_new_detection(obs)``
    在每个**新的**检测结果上调用一次——用 revision 去重，避免同一个推理
    结果被重复录成多样本。
    """

    deadline = time.monotonic() + seconds
    last_revision = -1
    while time.monotonic() < deadline:
        packet, _ = frames.get_latest()
        detection, revision = detections.get_latest()
        if (
            detection is not None
            and revision != last_revision
            and on_new_detection is not None
        ):
            last_revision = revision
            on_new_detection(detection)

        view = packet.frame.copy() if packet is not None else _blank_frame()
        remaining = max(0.0, deadline - time.monotonic())
        if detection is not None and detection.observations:
            observation = detection.observations[0]
            if len(observation.landmarks) == 21:
                debug_view.draw_skeleton(view, observation.landmarks)
        debug_view.draw_hud(view, rows_fn(remaining))
        cv2.imshow(_WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in _ABORT_KEYS:
            return True
    return False


def _pinch_ready(observation) -> bool:
    """粗判一次捏合（只用于"确认开始"，不参与任何手势判定）。

    刻意不复用 `gestures` 里的判据：那套带宽高比补偿与阈值常量，是**产品判定**；
    这里只需要"用户做了个捏合动作"这一件事，用最朴素的 dist(4,8)/dist(5,17)
    就够，也不怕将来产品阈值改动把向导卡住。
    """

    points = observation.landmarks
    if len(points) != 21:
        return False
    width = (
        (points[5].x - points[17].x) ** 2 + (points[5].y - points[17].y) ** 2
    ) ** 0.5
    if width < 1e-6:
        return False
    pinch = (
        (points[4].x - points[8].x) ** 2 + (points[4].y - points[8].y) ** 2
    ) ** 0.5
    return pinch / width <= 0.60


def _wait_for_confirm(
    frames: LatestSlot[FramePacket],
    detections: LatestSlot[DetectionPacket],
    step: CollectStep,
    *,
    seconds: float = 120.0,
) -> tuple[bool, bool]:
    """把题目显示出来，**等用户捏合（或按空格）确认**之后才返回。

    返回 ``(是否被中断, 是否已确认)``。

    🔴 为什么必须有这一步（用户原话）：「录手势的时候你要用户捏合确认后再录，
    不然你给的倒计时用户还没有看清题目，就开始录了」。第一版是"准备 2 秒"
    硬倒计时——2 秒连提示语都读不完，录进去的全是"还在找题"的帧，
    既污染样本，也让用户觉得自己被催。

    超时（120 秒）返回 ``(False, False)``：不确认也不卡死，按原流程继续。
    """

    deadline = time.monotonic() + seconds
    last_revision = -1
    streak = 0
    confirmed = False

    def rows(remaining: float):
        return [
            (f"准备做：{LABEL_NAMES[step.label]}", debug_view.YELLOW),
            (step.cue, debug_view.WHITE),
            ("", debug_view.WHITE),
            ("读完了就【捏合一下】开始（也可以按空格）", debug_view.GREEN),
            ("按 ESC 退出整个录制", debug_view.GRAY),
            (f"{remaining:.0f}s 后自动开始", debug_view.GRAY),
        ]

    while time.monotonic() < deadline:
        packet, _ = frames.get_latest()
        detection, revision = detections.get_latest()
        if detection is not None and revision != last_revision:
            last_revision = revision
            if detection.observations and _pinch_ready(detection.observations[0]):
                streak += 1
                if streak >= 3:  # 3 帧 ≈ 0.1s，压掉单帧抖动
                    confirmed = True
            else:
                streak = 0
        view = packet.frame.copy() if packet is not None else _blank_frame()
        if detection is not None and detection.observations:
            observation = detection.observations[0]
            if len(observation.landmarks) == 21:
                debug_view.draw_skeleton(view, observation.landmarks)
        remaining = max(0.0, deadline - time.monotonic())
        debug_view.draw_hud(view, rows(remaining))
        cv2.imshow(_WINDOW, view)
        key = cv2.waitKey(30) & 0xFF
        if key in _ABORT_KEYS:
            return True, False
        if key in (ord(" "), 13):  # 空格 / 回车
            return False, True
        if confirmed:
            return False, True
    return False, False


def run_collect(
    config: AppConfig,
    *,
    steps: tuple[CollectStep, ...] = STEPS,
    dataset: GestureDataset | None = None,
    confirm_before_record: bool = False,
) -> tuple[GestureDataset, bool, dict[str, int]]:
    """执行一轮采集，返回 ``(样本库, 是否被中断, 每种手势的角度覆盖档数)``。

    ``confirm_before_record=True`` 时，每种手势**先显示题目、等用户捏合（或按空格）
    确认**，才进准备/录制阶段。首次向导一定要开：`prep_seconds` 只有 2 秒，
    用户连提示语都读不完就开始录了。
    """

    coverage_map: dict[str, int] = {}
    result = dataset if dataset is not None else GestureDataset()
    source = time.strftime("collect-%Y%m%d-%H%M%S")
    stop_event, frames, detections, errors, workers = _open_workers(config)
    aborted = False
    try:
        # 等第一帧检测出来，否则前面几秒全是空录。
        warmup_deadline = time.monotonic() + 6.0
        while time.monotonic() < warmup_deadline and not stop_event.is_set():
            _, revision = detections.get_latest()
            if revision > 0:
                break
            time.sleep(0.1)
        if errors:
            raise RuntimeError(errors[0])

        for step in steps:
            if confirm_before_record:
                aborted, confirmed = _wait_for_confirm(frames, detections, step)
                if aborted:
                    break
                if not confirmed:
                    print(f"[采集] {LABEL_NAMES[step.label]}：没等到确认，按默认节奏开始。")
            recorded = 0
            total = len(result)
            coverage = CoverageTracker()

            def handle(detection, _step=step) -> None:
                nonlocal recorded
                if not detection.observations:
                    return
                observation = detection.observations[0]
                if len(observation.landmarks) == 21:
                    coverage.mark(wrist_rotation_ratio(observation.landmarks))
                if result.add_frame(
                    _step.label,
                    observation.landmarks,
                    source=source,
                    captured_ns=observation.timestamp_ns,
                    world=observation.world_landmarks,
                ):
                    recorded += 1

            def prep_rows(remaining, _step=step) -> list:
                return [
                    (f"准备：{LABEL_NAMES[_step.label]}", debug_view.YELLOW),
                    (_step.cue, debug_view.WHITE),
                    (f"{remaining:.1f} 秒后开始录制", debug_view.GRAY),
                    ("ESC / Q 退出", debug_view.GRAY),
                ]

            def record_rows(remaining, _step=step) -> list:
                return [
                    (f"录制中：{LABEL_NAMES[_step.label]}", debug_view.GREEN),
                    (_step.cue, debug_view.WHITE),
                    (
                        f"剩余 {remaining:.1f} 秒　本手势已录 {recorded} 帧"
                        f"　总计 {total + recorded} 帧",
                        debug_view.WHITE,
                    ),
                    (
                        coverage.hint(),
                        debug_view.GRAY if coverage.complete else debug_view.YELLOW,
                    ),
                ]

            aborted = _run_phase(
                frames=frames,
                detections=detections,
                seconds=step.prep_seconds,
                rows_fn=prep_rows,
            )
            if aborted:
                break
            aborted = _run_phase(
                frames=frames,
                detections=detections,
                seconds=step.record_seconds,
                rows_fn=record_rows,
                on_new_detection=handle,
            )
            if aborted:
                break
            coverage_map[step.label] = coverage.covered
            note = (
                ""
                if coverage.complete
                else f"　角度覆盖 {coverage.covered}/{coverage.total}（未采全）"
            )
            print(f"[采集] {LABEL_NAMES[step.label]}：{recorded} 帧{note}")
            if not coverage.complete:
                print(
                    f"       └ 换个手腕角度补录："
                    f"python -m handmouse.collect --append --only {step.label}",
                )
    finally:
        stop_event.set()
        for worker in workers:
            worker.join(timeout=2.0)
        try:
            cv2.destroyWindow(_WINDOW)
        except cv2.error:
            pass
    return result, aborted, coverage_map


def _report(
    dataset: GestureDataset,
    coverage: dict[str, int] | None = None,
    collected: "list[str] | None" = None,
) -> float | None:
    """采集后自检：样本够不够、角度覆盖够不够、冲突在哪、阈值该取多少。"""

    counts = dataset.counts()
    print("\n[样本统计]")
    for label in LABELS:
        covered = (coverage or {}).get(label)
        suffix = "" if covered is None else f"　角度 {covered}/{COVERAGE_BINS}"
        print(f"  {LABEL_NAMES[label]:<20} {counts.get(label, 0)} 帧{suffix}")
    missing = [label for label in LABELS if counts.get(label, 0) < 20]
    if missing:
        names = "、".join(LABEL_NAMES[label] for label in missing)
        print(f"\n[警告] {names} 样本不足 20 帧，识别会不稳，建议补采。")
    if not dataset.is_ready():
        return None
    # 评估按**粗分类**做：移动/停住/捏合三者在单帧距离上没有边界
    # （实测类间最近 0.37 小于类内 p90 0.69），报告它们"冲突"是误导。
    # 它们的区分交给现有时序逻辑，模型只负责 point / open / fist / swipe
    # 这四类的姿势层面（其中 swipe 是食中二指伸开的静止姿势）。
    coarse = GestureDataset(
        GestureSample(
            label=COARSE_MAP.get(sample.label, sample.label),
            features=sample.features,
            captured_ns=sample.captured_ns,
            source=sample.source,
        )
        for sample in dataset
    )
    threshold = suggest_reject_distance(coarse)
    stats = evaluate(coarse, threshold)
    print(f"\n[建议拒识阈值] {threshold:.4f}")
    print(
        "[留一法自检·粗分类] 正确 {:.1%}　误判 {:.1%}　拒识 {:.1%}".format(
            stats["correct"],
            stats["wrong"],
            stats["rejected"],
        ),
    )
    print("（误判率越低越好；拒识率高说明样本还不够全，补采即可）")

    pairs = conflicts(coarse, threshold)
    if pairs:
        print("\n[冲突] 这几组我分不开，需要你裁定：")
        for left, right, count in pairs:
            print(f"  {COARSE_NAMES.get(left, left)} ↔ {COARSE_NAMES.get(right, right)}：{count} 帧混淆")
        print("  处理方式：把其中一种做得更夸张些再补采。")
    else:
        print("\n[冲突] point / 五指张开 / 握拳 三类互不混淆，可以直接用。")
    if collected and "swipe" in collected:
        print(_SWIPE_MULTI_ANGLE_HINT)
    return threshold


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m handmouse.collect",
        description="采集个人手势样本，用于替代通用几何判据。",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=9.0,
        help="每种手势的录制时长（秒），默认 9",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="只采集指定手势，逗号分隔，例如 --only move,pinch",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="在已有样本库上追加，而不是覆盖",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help=f"输出路径，默认 {default_dataset_path()}",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = ConfigStore().load()

    steps = STEPS
    if args.only:
        wanted = [item.strip() for item in args.only.split(",") if item.strip()]
        unknown = [item for item in wanted if item not in LABELS]
        if unknown:
            print(f"[错误] 未知手势：{unknown}；可选：{list(LABELS)}")
            return 2
        steps = tuple(
            CollectStep(
                label=step.label,
                prep_seconds=step.prep_seconds,
                record_seconds=args.seconds,
                cue=step.cue,
            )
            for step in STEPS
            if step.label in wanted
        )
    else:
        steps = tuple(
            CollectStep(
                label=step.label,
                prep_seconds=step.prep_seconds,
                record_seconds=args.seconds,
                cue=step.cue,
            )
            for step in STEPS
        )

    out_path = PROJECT_ROOT / args.out if args.out else default_dataset_path()
    dataset = GestureDataset.load(out_path) if args.append else GestureDataset()
    if args.append and len(dataset):
        print(f"[采集] 已载入 {len(dataset)} 条旧样本，本次追加。")

    print("[采集] 摄像头启动中……（画面出来后按提示做手势）")
    coverage: dict[str, int] = {}
    try:
        dataset, aborted, coverage = run_collect(config, steps=steps, dataset=dataset)
    except RuntimeError as exc:
        print(f"[错误] {exc}")
        return 1
    if aborted:
        print("\n[采集] 已中断。")

    if not len(dataset):
        print("[采集] 没有采到任何样本，未写盘。")
        return 1

    # 清洗：手侧到接近平行视线时关键点是猜的，这些帧会毒化拒识阈值。
    before = len(dataset)
    dataset, dropped = clean_dataset(dataset)
    removed = before - len(dataset)
    if removed:
        print(f"\n[清洗] 剔除离群帧 {removed} 帧（{removed / before:.0%}），保留 {len(dataset)} 帧")
        for label in LABELS:
            if dropped.get(label):
                print(f"       {LABEL_NAMES[label]} 丢弃 {dropped[label]} 帧")
    written = dataset.save(out_path)
    print(f"\n[采集] 已写入 {written} 条样本 → {out_path}")
    labels = [step.label for step in steps]
    threshold = _report(dataset, coverage, collected=labels)
    if threshold is not None:
        # 把阈值落到配置里：启动时现算要跑 400×400 次两两距离，会让程序
        # 卡好几秒。采集时算一次、写进 config，启动直接读。
        store = ConfigStore()
        config.gesture_model_reject_distance = round(threshold, 4)
        if args.out:
            config.gesture_model_path = str(out_path)
        config.gesture_model_enabled = True
        try:
            store.save(config)
            print(f"\n[配置] 已写入拒识阈值 {threshold:.4f} 并开启手势库")
            print("       想临时关掉，把 config.json 里 gesture_model_enabled 改成 false")
        except OSError as exc:
            print(f"\n[警告] 配置写入失败（{exc}），请手动设置：")
            print(f"       gesture_model_enabled = true")
            print(f"       gesture_model_reject_distance = {threshold:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
