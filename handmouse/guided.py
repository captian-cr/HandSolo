"""引导式基线脚本：把 6 步验证变成固定时长的节拍提示。

动机：实机验证的前两轮（r1/r2）失败在"动作靠临场发挥、时间点靠事后回忆"，
日志无法归因。本模块把每一步的**开始时刻与时长**写死，由线程按节拍把
``[步骤]`` 提示打进 stdout（与 ``[状态]``/``[采样]`` 同一份日志），复盘时
按时间戳对齐即可，不需要操作者再手工报时。

使用：``python -m handmouse --safe-mode --guided``。引导模式隐含
``--auto-control``（引擎必须处于接管态，否则看不到 pointer/lost 迁移），
配合 ``--safe-mode`` 时 LAUNCH 走 dry-run，没有任何系统副作用。
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

import cv2


def build_steps() -> list[tuple[float, str]]:
    """返回 (时长秒, 提示语) 序列；总时长即基线一轮的长度。"""

    steps: list[tuple[float, str]] = [
        (8, "准备：把手完全移出画面，之后严格按提示操作"),
        (20, "无手基线：手保持在画面外（此阶段任何 [状态] 迁移/[注入] 都是污染证据）"),
    ]
    for i in range(1, 4):
        steps.append((5, f"第1步 {i}/3：整只手进画面，食指指向前方，停住"))
        steps.append((4, f"第1步 {i}/3：整只手完全出画（不是挪到角落）"))
    steps += [
        (60, "第2步：手在画面内，像平时一样慢速移动光标，不做双指悬停/捏合/翻页"),
        (10, "第3步：食指指住一处 → 捏合 → 松开（只做一次）"),
        (12, "第4步：捏住并保持 6 秒以上再松开（只做一次）"),
    ]
    for shape in ("W", "C", "X"):
        steps.append((14, f"第5步 {shape}：双指悬停 ~300ms → 画 {shape} → 收指；不要捏合"))
        steps.append((6, f"第5步 {shape}：现在捏合确认"))
    steps += [
        (15, "第6步：用手掌或纸板完全盖住镜头 3 秒，然后拿开"),
        (45, "第6步：到设备管理器禁用摄像头，再启用；然后手回画面动一下"),
        (5, "结束：保持不动，等待诊断输出"),
    ]
    return steps


def total_seconds(steps: list[tuple[float, str]]) -> float:
    return sum(duration for duration, _prompt in steps)


def start_prompter(
    stop_event: threading.Event,
    on_step: Callable[[int, str], None] | None = None,
) -> threading.Thread:
    """后台线程：按节拍打印 ``[步骤] HH:MM:SS 提示语``，播完即止。

    ``on_step(index, prompt)`` 在每个节拍打印后回调，用于让取证抓帧在
    节拍边界留一张"此刻画面"。
    """

    steps = build_steps()

    def _run() -> None:
        for index, (duration, prompt) in enumerate(steps, start=1):
            if stop_event.is_set():
                return
            print(f"[步骤] {time.strftime('%H:%M:%S')} ({index}/{len(steps)}) {prompt}")
            if on_step is not None:
                on_step(index, prompt)
            stop_event.wait(duration)

    thread = threading.Thread(target=_run, name="guided-prompter", daemon=True)
    thread.start()
    return thread


class FrameRecorder:
    """引导基线的取证抓帧：事件触发，主循环用最新调试视图落盘。

    为什么拆成 request/capture 两步：状态迁移、注入、轨迹事件都发生在
    gesture-control 线程，而画面帧在主线程手里。所以事件线程只记"原因"，
    主循环在下一帧把**调试视图**（已含 landmarks 与状态叠加，即操作者
    当时亲眼看到的画面）存成 JPG，文件名带时间戳与原因。

    上一轮基线（r1）的教训：无手段出现 phantom 检出后，操作者事后无法
    回忆画面内容，无法定污染性质。取证帧让"画面里当时有什么"变成可
    复核的物证，不靠记忆。
    """

    #: 命中这些标签的日志行会触发抓帧（与落盘日志同一事件流，保证帧行对应）
    TRIGGER_TAGS = ("[状态]", "[注入]", "[轨迹]")

    def __init__(
        self,
        directory: Path,
        debounce_seconds: float = 0.15,
        max_reason: int = 60,
    ) -> None:
        self.directory = directory
        self.debounce_seconds = debounce_seconds
        self.max_reason = max_reason
        self.saved = 0
        self._pending: deque[str] = deque()
        self._lock = threading.Lock()
        self._last_capture = 0.0
        directory.mkdir(parents=True, exist_ok=True)

    def request(self, reason: str) -> None:
        """任意线程：登记一个待抓帧事件原因。"""
        with self._lock:
            self._pending.append(reason)

    def capture(self, frame_bgr) -> bool:
        """主循环：若有待存事件且过了防抖窗口，把当前帧存为 JPG。"""
        if frame_bgr is None:
            return False
        with self._lock:
            if not self._pending:
                return False
            now = time.monotonic()
            if now - self._last_capture < self.debounce_seconds:
                return False
            reasons = list(self._pending)
            self._pending.clear()
            self._last_capture = now
        now_wall = time.time()
        stamp = time.strftime("%H-%M-%S", time.localtime(now_wall))
        stamp += f"-{int(now_wall * 1000) % 1000:03d}"
        filename = f"{stamp}_{self._sanitize(';'.join(reasons))}.jpg"
        try:
            # cv2.imwrite 在 Windows 下按 ANSI 代码页解释路径，中文文件名会被
            # 写花；改成 imencode 出字节流、由 Path 按 UTF-8 落盘。
            ok, buffer = cv2.imencode(".jpg", frame_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                return False
            (self.directory / filename).write_bytes(buffer.tobytes())
        except (cv2.error, OSError):
            return False  # 取证失败不阻塞基线本身
        self.saved += 1
        return True

    def _sanitize(self, reason: str) -> str:
        """把事件原因变成合法文件名片段：保留中日文/字母数字与箭头。"""
        chars = [ch if (ch.isalnum() or ch in "→-") else "_" for ch in reason]
        text = "".join(chars)
        while "__" in text:
            text = text.replace("__", "_")
        return text.strip("_")[: self.max_reason]


def attach_event_trigger(recorder: FrameRecorder) -> logging.Handler:
    """把 recorder 挂到 pipeline 日志上：出现 [状态]/[注入]/[轨迹] 行即抓帧。"""

    class _EventTrigger(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            try:
                message = record.getMessage()
            except Exception:  # pragma: no cover - 防御性格式化失败
                return
            if any(tag in message for tag in FrameRecorder.TRIGGER_TAGS):
                recorder.request(message)

    handler = _EventTrigger()
    logging.getLogger("handmouse.pipeline").addHandler(handler)
    return handler
