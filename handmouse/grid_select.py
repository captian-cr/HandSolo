"""固定图标网格选择：独立入口。

启动：``python -m handmouse.grid_select``

与本仓库主线的 ``python -m handmouse`` 完全独立：不导入 ``GestureEngine``、
不注入任何系统鼠标事件、不改动 ``config.json``。跑不通随时删掉这四个
``grid_*.py`` 文件即可，主线不受影响。

设计取舍
--------
* **不做连续光标注入**。只判断"手指现在稳定指着哪个图标"，命中后靠捏合
  确认触发。因此端到端延迟（当前实机约 107ms）几乎不影响手感——晚 100ms
  一样选得中。
* **磁吸**把"连续定位精度"换成"离散容错"：球可以抖，只要落在图标附近的
  吸引区内就会被拉正。
* **只读食指 + 拇指两点**。不再判断中指/无名指/小指状态，判定面从十几个
  降到两个。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

import cv2

from handmouse import grid_winctl as winctl
from handmouse.grid_judge import ClickGate, PinchJudger, TargetStabilizer, pinch_ratio
from handmouse.grid_map import GridLayout, map_roi, scale_roi
from handmouse.grid_ui import GridOverlay, PointerState

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "grid_config.json"

#: 窗口控制面板的四个按钮，顺序与格子索引一致。
WINDOW_LABELS = ("最小化", "最大化", "关闭", "返回")


def ascii_safe_model_path(source: Path) -> Path:
    """返回一个 ASCII 路径下的模型副本。

    MediaPipe 的 C++ 层在 Windows 上无法打开含非 ASCII 字符的路径——项目
    放在中文目录下（例如 ``D:/某个/中文路径/手势追踪``）时会直接抛
    ``RuntimeError: Unable to open file``。这里在路径非 ASCII 时把模型复制
    到临时目录的一份 ASCII 路径上使用；复制后按文件大小做一次幂等校验。
    """

    if not source.exists():
        return source
    if str(source).isascii():
        return source

    import shutil
    import tempfile

    for base in (Path(tempfile.gettempdir()), Path(os.environ.get("PUBLIC", "C:/"))):
        if not str(base).isascii():
            continue
        target_dir = base / "handmouse_models"
        target = target_dir / source.name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != source.stat().st_size:
                shutil.copy2(source, target)
            return target
        except OSError:
            continue
    return source

DEFAULT_CONFIG: dict = {
    "camera_index": 0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "mirror": True,
    "cols": 2,
    "rows": 2,
    "margin_x": 0.16,
    "margin_y": 0.20,
    "gap": 0.10,
    "snap_radius": 0.16,
    "roi": [0.15, 0.85, 0.15, 0.85],
    "sensitivity": 1.8,
    "pinch_enter": 0.40,
    "pinch_release": 0.62,
    "pinch_hold_ms": 120,
    "confirm_frames": 3,
    "pointer_smoothing_s": 0.06,
    "snap_enabled": True,
    "pointer_snap_px": 25,
    "pointer_snap_release_px": 50,
    # ---- 主题桌面（theme_select）的外观 ----
    # 壁纸："mesh" 内置渐变 / "#1b2a5e" 纯色 / "D:/wall/a.jpg" 图片
    "wallpaper": "mesh",
    "wallpaper_dim": 0.42,
    "icon": 168,
    "theme_cols": 4,
    "theme_rows": 3,
    "col_pitch": 280,
    "row_pitch": 232,
    "panel_cols": 4,
    "panel_rows": 2,
    "pill_width": 460,
    "pill_height": 112,
    "pill_bottom": 70,
    "items": [
        {"label": "记事本", "command": "notepad.exe"},
        {"label": "资源管理器", "command": "explorer.exe"},
        {"label": "窗口控制", "command": "@window"},
        {"label": "退出", "command": "@quit"},
    ],
}


def load_config(path: Path) -> dict:
    """读配置；文件不存在时写出一份默认值，方便用户直接改。"""

    if not path.exists():
        path.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return json.loads(json.dumps(DEFAULT_CONFIG))
    data = json.loads(path.read_text(encoding="utf-8"))
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    merged.update(data)
    return merged


@dataclass(slots=True)
class HandSample:
    """检测线程发布的最新一帧结论。"""

    index_tip: tuple[float, float]
    thumb_tip: tuple[float, float]
    index_root: tuple[float, float]
    ratio: float
    captured_ms: float
    #: 全部 21 个归一化关键点，供姿态分类使用。
    points: list[tuple[float, float]] = field(default_factory=list)
    #: 掌心近似位置（手腕与中指根的中点），挥手翻页用。
    palm: tuple[float, float] = (0.5, 0.5)


class HandPoller(threading.Thread):
    """同步摄像头+推理循环，只保留最新一帧结论。

    这里没有沿用主线的三线程 ``LatestSlot`` 管线：本模块每帧只读两个关键点，
    推理耗时远低于采集间隔，单线程足够；独立实现也让本模块能整块删除。
    """

    def __init__(self, config: dict, stop_event: threading.Event) -> None:
        super().__init__(name="grid-hand-poller", daemon=True)
        self.config = config
        self.stop_event = stop_event
        self._lock = threading.Lock()
        self._sample: HandSample | None = None
        self._fps = 0.0
        self._error: str = ""
        self._frame_count = 0
        self._fps_mark = time.monotonic()

    def snapshot(self) -> tuple[HandSample | None, float, str]:
        with self._lock:
            return self._sample, self._fps, self._error

    def _publish(self, sample: HandSample | None) -> None:
        with self._lock:
            self._sample = sample

    def run(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not MODEL_PATH.exists():
            with self._lock:
                self._error = f"找不到模型文件: {MODEL_PATH}"
            return

        model_path = ascii_safe_model_path(MODEL_PATH)
        if model_path != MODEL_PATH:
            print(
                f"[模型] 项目路径含非 ASCII 字符，MediaPipe 无法直接打开；"
                f"已改用副本 {model_path}",
                file=sys.stderr,
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            # 比主线默认值更严：本模块按"稳定优先"设计，宁可丢手也不接受
            # 把脸/衣物拟合成一只不存在的 21 点手。
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.6,
        )
        landmarker = mp_vision.HandLandmarker.create_from_options(options)

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        capture = cv2.VideoCapture(self.config["camera_index"], backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera_width"])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera_height"])
        capture.set(cv2.CAP_PROP_FPS, self.config["camera_fps"])
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                if self.config["mirror"]:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now_ms = time.monotonic() * 1000.0
                result = landmarker.detect_for_video(mp_image, int(now_ms))

                if result.hand_landmarks:
                    points = result.hand_landmarks[0]
                    thumb = (points[4].x, points[4].y)
                    index_tip = (points[8].x, points[8].y)
                    index_root = (points[5].x, points[5].y)
                    cloud = [(p.x, p.y) for p in points]
                    self._publish(
                        HandSample(
                            index_tip=index_tip,
                            thumb_tip=thumb,
                            index_root=index_root,
                            ratio=pinch_ratio(thumb, index_tip, index_root),
                            captured_ms=now_ms,
                            points=cloud,
                            palm=(
                                (cloud[0][0] + cloud[9][0]) / 2,
                                (cloud[0][1] + cloud[9][1]) / 2,
                            ),
                        ),
                    )
                else:
                    self._publish(None)

                self._frame_count += 1
                elapsed = time.monotonic() - self._fps_mark
                if elapsed >= 0.5:
                    with self._lock:
                        self._fps = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_mark = time.monotonic()
        finally:
            capture.release()
            landmarker.close()


@dataclass(slots=True)
class Runner:
    """把检测结果接到判定与触发上。"""

    config: dict
    layout: GridLayout
    judge: PinchJudger
    stabilizer: TargetStabilizer
    state: PointerState
    stop_event: threading.Event
    dry_run: bool = False
    _overlay: GridOverlay | None = None
    _message_until_ms: float = field(default=0.0, init=False)
    _mode: str = field(default="grid", init=False)
    _target_hwnd: int = field(default=0, init=False)
    _locked_target: int | None = field(default=None, init=False)
    _click_gate: ClickGate = field(default_factory=ClickGate, init=False)

    def attach(self, overlay: GridOverlay) -> None:
        self._overlay = overlay

    def tick(self) -> None:
        sample, fps, error = self.poller.snapshot()
        self.state.fps = fps
        if error:
            self.state.message = error

        now_ms = time.monotonic() * 1000.0
        if sample is None:
            self.state.hand_visible = False
            self.state.target = None
            self.state.pinching = False
            self.state.ratio = float("inf")
            self.judge.reset()
            self.stabilizer.reset()
            return

        self.state.hand_visible = True
        self.state.ratio = sample.ratio

        roi = scale_roi(tuple(self.config["roi"]), float(self.config["sensitivity"]))
        px, py = map_roi(sample.index_tip[0], sample.index_tip[1], roi)
        self.state.x = min(max(px, 0.0), 1.0)
        self.state.y = min(max(py, 0.0), 1.0)

        raw_target = self.layout.hit(self.state.x, self.state.y, self.config["snap_radius"])
        stable_target = self.stabilizer.update(raw_target)

        triggered = self.judge.update(sample.ratio, now_ms)
        self.state.pinching = self.judge.is_closed

        # 捏合时食指必然向拇指靠拢，而食指尖正是定位锚点——不钉住目标的话
        # 球会跟着手指滑走，确认瞬间点到的可能是隔壁格子。因此在这一刻把
        # 目标冻结，整个捏合过程都以它为准。
        if self.judge.just_closed:
            self._locked_target = (
                stable_target if stable_target is not None else raw_target
            )
        elif self.judge.is_closed:
            if self._locked_target is not None:
                # 手整体移开很远则视为放弃本次捏合，避免"指着别处却点了原来那格"。
                cx, cy = self.layout.center(self._locked_target)
                drift = hypot(self.state.x - cx, self.state.y - cy)
                limit = max(float(self.config["snap_radius"]) * 2.0, 0.15)
                if drift > limit:
                    self._locked_target = None
        else:
            self._locked_target = None

        active_target = (
            self._locked_target if self.judge.is_closed else stable_target
        )
        self.state.target = active_target

        if triggered and active_target is not None:
            self._trigger(active_target, now_ms)

        expired = self._click_gate.poll(now_ms)
        if expired is not None:
            self.state.message = (
                f"已选中：{self.labels[expired]}（再捏一次执行）"
            )
            self._message_until_ms = now_ms + 1600

        if now_ms < self._message_until_ms:
            return
        if active_target is None:
            if self.judge.is_closed:
                self.state.message = "捏合时目标丢失，请先指稳再捏"
            else:
                self.state.message = "指着留白（安全区），捏合无效"
        else:
            lock_text = "（已锁定）" if self.judge.is_closed else ""
            hint = (
                "再捏一次执行"
                if self._click_gate.armed
                else "捏合两次执行"
            )
            self.state.message = (
                f"指向：{self.labels[active_target]}{lock_text}　{hint}"
            )

    poller: HandPoller = field(default=None, init=False)  # type: ignore[assignment]

    @property
    def labels(self) -> list[str]:
        return [item["label"] for item in self.config["items"]]

    def _trigger(self, index: int, now_ms: float) -> None:
        if self._mode == "window":
            self._trigger_window(index, now_ms)
            return

        item = self.config["items"][index]
        command = str(item.get("command", "")).strip()
        label = item["label"]

        if self.dry_run:
            self.state.message = f"[dry-run] 命中：{label}"
            self._message_until_ms = now_ms + 1500
            return

        if command == "@window":
            self._enter_window_mode(now_ms)
            return

        if command == "@quit":
            self.state.message = "已退出"
            self.stop_event.set()
            if self._overlay is not None:
                self._overlay.close()
            return

        if not command:
            self.state.message = f"{label}：未配置命令"
            self._message_until_ms = now_ms + 1500
            return

        # 启动程序要求捏合两次：一次只是选中，避免抬手时误开一堆东西。
        # 导航类动作（切面板 / 退出）在上面已单击即执行。
        if self._click_gate.register(index, now_ms) != "double":
            self.state.message = f"{label}　再捏一次启动"
            self._message_until_ms = now_ms + 900
            return

        try:
            candidate = Path(command)
            if candidate.exists():
                # 快捷方式/文件夹交给 Shell 处理：.lnk 会用其记录的目标启动，
                # 直接 Popen 一个 .lnk 并不可靠。
                os.startfile(str(candidate))  # noqa: S606
            else:
                subprocess.Popen(command, shell=False)
        except Exception as exc:
            self.state.message = f"启动失败：{exc}"
        else:
            self.state.message = f"已启动：{label}"
        self._message_until_ms = now_ms + 1500

        if self._overlay is not None:
            self._overlay.flash(index)

    # ---- 窗口控制面板 ----

    def _enter_window_mode(self, now_ms: float) -> None:
        """切到"最小化/最大化/关闭/返回"面板，并锁定要操作的目标窗口。

        目标窗口只在进入面板时取一次：面板本身是置顶的，如果每次点击都重新
        取"当前前台窗口"，很可能取到叠加层自己或者被它挡住的窗口。取一次并
        把标题显示在 HUD 上，用户能确认要关的是哪一个。
        """

        if self._overlay is None:
            return
        if len(self.config["items"]) != len(WINDOW_LABELS):
            self.state.message = f"窗口控制需要 {len(WINDOW_LABELS)} 格布局"
            self._message_until_ms = now_ms + 1500
            return

        self._target_hwnd = winctl.foreground_window(
            exclude=self._overlay.hwnd,
        )
        self._mode = "window"
        self._overlay.set_labels(list(WINDOW_LABELS))

        if self._target_hwnd:
            title = winctl.window_title(self._target_hwnd)
            self.state.message = f"目标窗口：{title}"
        else:
            self.state.message = "没有可操作的窗口（先点开一个程序）"
        self._message_until_ms = now_ms + 2000

    def _exit_window_mode(self, now_ms: float) -> None:
        self._mode = "grid"
        self._target_hwnd = 0
        if self._overlay is not None:
            self._overlay.set_labels(self.labels)
        self.state.message = "已返回启动器"
        self._message_until_ms = now_ms + 1200

    def _trigger_window(self, index: int, now_ms: float) -> None:
        if index == 3:
            self._exit_window_mode(now_ms)
            return

        if self.dry_run:
            self.state.message = f"[dry-run] 窗口操作：{WINDOW_LABELS[index]}"
            self._message_until_ms = now_ms + 1500
            return

        hwnd = self._target_hwnd
        if not hwnd:
            self.state.message = "没有目标窗口，先退回去点开一个程序"
            self._message_until_ms = now_ms + 1500
            return

        title = winctl.window_title(hwnd)
        if index == 0:
            ok = winctl.minimize(hwnd)
            action = "最小化"
        elif index == 1:
            ok = winctl.toggle_maximize(hwnd)
            action = "最大化切换"
        else:
            ok = winctl.close(hwnd)
            action = "关闭"

        self.state.message = (
            f"已{action}：{title}" if ok else f"{action}失败：窗口可能已关闭"
        )
        self._message_until_ms = now_ms + 1500
        if ok and self._overlay is not None:
            self._overlay.flash(index)
        if not ok:
            self._target_hwnd = 0


def save_config(path: Path, config: dict) -> None:
    """把配置写回文件（标定结果要能持久化）。"""

    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_calibration(config: dict, config_path: Path, seconds: float = 6.0) -> int:
    """测量食指实际活动范围，写入 ROI。

    为什么需要这一步：``roi`` 是"手在画面里会活动的区域"。写死一个默认值
    不可能适配所有人——手离摄像头远近、坐姿、手臂展开幅度都不同。写大了
    要大幅移动才够得着屏幕边缘（就是"累"的来源），写小了抖动被放大。
    让用户移动一遍、取分位数，一次标定长期有效。

    标定后 ``sensitivity`` 归 1.0：ROI 已经是实测取值范围，不需要再放大。
    """

    stop_event = threading.Event()
    poller = HandPoller(config, stop_event)
    poller.start()

    print(
        f"\n标定开始：请在 {seconds:.0f} 秒内，用食指尖在**你觉得舒服的范围内**"
        "\n从左上角慢慢移到右下角，再移回来，重复几次（画个圈也行）。"
        "\n范围不用求大，平时怎么用就怎么动。\n",
        file=sys.stderr,
    )

    xs: list[float] = []
    ys: list[float] = []
    last_report = time.monotonic()
    deadline = time.monotonic() + seconds

    try:
        while time.monotonic() < deadline and not stop_event.is_set():
            sample, _fps, error = poller.snapshot()
            if error:
                print(f"[标定] {error}", file=sys.stderr)
                break
            if sample is not None:
                xs.append(sample.index_tip[0])
                ys.append(sample.index_tip[1])
            if time.monotonic() - last_report >= 1.0:
                print(f"[标定] 已采集 {len(xs)} 个点", file=sys.stderr)
                last_report = time.monotonic()
            time.sleep(0.02)
    finally:
        stop_event.set()
        poller.join(timeout=2.0)

    if len(xs) < 20:
        print(
            f"\n[标定] 只采到 {len(xs)} 个点，样本太少无法标定。"
            "\n        先确认摄像头能看到手、画面不过暗，再重试。",
            file=sys.stderr,
        )
        return 1

    xs.sort()
    ys.sort()

    def quantile(values: list[float], q: float) -> float:
        return values[int((len(values) - 1) * q)]

    # 取 3%~97% 分位，丢掉极端离群点（手刚进画面、检测跳变）。
    x_min, x_max = quantile(xs, 0.03), quantile(xs, 0.97)
    y_min, y_max = quantile(ys, 0.03), quantile(ys, 0.97)

    # 范围过窄说明手几乎没动，此时映射会被抖动主导，拒绝保存。
    if x_max - x_min < 0.08 or y_max - y_min < 0.08:
        print(
            f"\n[标定] 检测到的活动范围太小"
            f"（x {x_min:.3f}~{x_max:.3f}, y {y_min:.3f}~{y_max:.3f}）。"
            "\n        请把手移动的幅度再放大一些后重试。",
            file=sys.stderr,
        )
        return 1

    config["roi"] = [round(x_min, 4), round(x_max, 4), round(y_min, 4), round(y_max, 4)]
    config["sensitivity"] = 1.0
    save_config(config_path, config)

    print(
        "\n[标定] 完成：\n"
        f"        roi = {config['roi']}\n"
        f"        sensitivity = {config['sensitivity']}（标定后归 1.0）\n"
        f"        已写入 {config_path}\n"
        "        觉得还是累，就把 sensitivity 调到 1.3~1.6；"
        "觉得太飘就调到 0.8。",
        file=sys.stderr,
    )
    return 0


def layout_presets(cols: int, rows: int) -> tuple[float, float, float, int]:
    """按格子数返回 ``(margin_x, margin_y, gap, font_size)``。

    格子越多，留白与间距越小、字号越小，但每格仍然远大于 80px 的精度门槛。
    """

    cells = cols * rows
    if cells <= 4:
        return 0.16, 0.20, 0.10, 30
    if cells <= 9:
        return 0.10, 0.12, 0.05, 28
    return 0.06, 0.08, 0.03, 24


def build_runner(config: dict, stop_event: threading.Event) -> Runner:
    layout = GridLayout(
        cols=int(config["cols"]),
        rows=int(config["rows"]),
        margin_x=float(config["margin_x"]),
        margin_y=float(config["margin_y"]),
        gap=float(config["gap"]),
    )
    snapshot_radius = float(config["snap_radius"])
    if snapshot_radius > layout.max_snap_radius:
        # 超过上限会让相邻图标的吸引区重叠，重叠区只能猜。这里主动夹回，
        # 并在启动时报出来——静默夹取会让人以为参数生效了。
        print(
            f"[配置] snap_radius={snapshot_radius:.3f} 超过安全上限 "
            f"{layout.max_snap_radius:.3f}（图标间距一半），已夹到上限。",
            file=sys.stderr,
        )
        config["snap_radius"] = layout.max_snap_radius

    judge = PinchJudger(
        enter_ratio=float(config["pinch_enter"]),
        release_ratio=float(config["pinch_release"]),
        hold_ms=float(config["pinch_hold_ms"]),
    )
    stabilizer = TargetStabilizer(confirm_frames=int(config["confirm_frames"]))
    runner = Runner(
        config=config,
        layout=layout,
        judge=judge,
        stabilizer=stabilizer,
        state=PointerState(),
        stop_event=stop_event,
    )
    runner.poller = HandPoller(config, stop_event)
    return runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m handmouse.grid_select",
        description="固定图标网格选择：手指指向 + 捏合确认，不移动系统光标。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="配置文件路径")
    parser.add_argument("--camera", type=int, default=None, help="摄像头序号，覆盖配置")
    parser.add_argument("--cols", type=int, default=None, help="列数，覆盖配置")
    parser.add_argument("--rows", type=int, default=None, help="行数，覆盖配置")
    parser.add_argument("--no-mirror", action="store_true", help="关闭镜像")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印判定结果，不开叠加窗口（无窗口环境下自检用）",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="测量食指实际活动范围并写入 roi，然后退出",
    )
    parser.add_argument(
        "--source", choices=("config",), default="config",
        help="条目来源（桌面图标请用 python -m handmouse.desktop_select）",
    )
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    config = load_config(config_path)
    if args.camera is not None:
        config["camera_index"] = args.camera
    if args.cols is not None:
        config["cols"] = args.cols
    if args.rows is not None:
        config["rows"] = args.rows
    if args.no_mirror:
        config["mirror"] = False

    _mx, _my, _gap, font_size = layout_presets(
        int(config["cols"]), int(config["rows"]),
    )

    item_count = int(config["cols"]) * int(config["rows"])
    if len(config["items"]) != item_count:
        print(
            f"[配置] items 数量 {len(config['items'])} 与 cols×rows={item_count} 不一致，"
            "已按数量截断/补空。",
            file=sys.stderr,
        )
        items = list(config["items"])[:item_count]
        while len(items) < item_count:
            items.append({"label": f"空位{len(items) + 1}", "command": ""})
        config["items"] = items

    stop_event = threading.Event()
    runner = build_runner(config, stop_event)
    runner.dry_run = args.dry_run

    if args.calibrate:
        return run_calibration(config, config_path, seconds=6.0)

    print(
        f"[网格] {config['cols']}×{config['rows']}　"
        f"磁吸半径={float(config['snap_radius']):.3f}　"
        f"上限={runner.layout.max_snap_radius:.3f}　"
        f"灵敏度={float(config['sensitivity']):.2f}",
        file=sys.stderr,
    )
    print(f"[配置] {config_path}", file=sys.stderr)

    if args.dry_run:
        runner.poller.start()
        try:
            while not stop_event.is_set():
                runner.tick()
                if runner.state.hand_visible:
                    print(
                        f"手可见 x={runner.state.x:.3f} y={runner.state.y:.3f} "
                        f"ratio={runner.state.ratio:.2f} "
                        f"目标={runner.state.target} 捏合={runner.state.pinching}",
                        flush=True,
                    )
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            runner.poller.join(timeout=2.0)
        return 0

    overlay = GridOverlay(
        layout=runner.layout,
        labels=runner.labels,
        state=runner.state,
        font_size=font_size,
        title="手势选择",
    )
    runner.state.message = "正在加载手部模型…"
    # 先把界面显示出来再去加载模型：模型初始化要几秒，这段时间主线程如果
    # 还没进消息循环，窗口不会刷新，用户会以为程序卡住。
    if not overlay.show():
        stop_event.set()
        return 1
    runner.attach(overlay)
    runner.poller.start()
    try:
        overlay.run(runner.tick)
    finally:
        stop_event.set()
        runner.poller.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
