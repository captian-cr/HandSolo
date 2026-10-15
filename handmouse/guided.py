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
    """返回 (时长秒, 提示语) 序列；总时长即基线一轮的长度。

    脚本按"进入门"设计：未做进入姿势时引擎只报 ``gate_idle``，下游手势一律
    不执行。所以每个可控动作之前都先给一段"进入"节拍；手出画会退出控制态，
    后续动作必须重新进入。

    进入门落地后，"纯日常控制"分成两个必须分开记录的语境：
      * **语境A（未进入）**：不做进入手势直接动食指 → 预期光标不动、``[注入]``
        为 0。这是"门挡住了日常手"的正面观察项。
      * **语境B（已进入后）**：进入控制态后再日常控制 → 预期正常控制且**不**出现
        非预期事件。
    另有"误进观察"节拍（手背/侧掌/半握）验证手势本身开不了闸门。
    """

    steps: list[tuple[float, str]] = [
        (8, "准备：把手完全移出画面，之后严格按提示操作"),
        (20, "无手基线：手保持在画面外（此阶段任何 [状态] 迁移/[注入] 都是污染证据）"),
        (15, "误进观察：手背朝镜 / 侧掌 / 半握 各伸进画面停 5 秒 → 应始终 gate_idle，不得进控制态（[轨迹] 全程应 idle）"),
        (8, "语境A 未进入：不做进入手势，直接把食指伸进画面并移动 → 预期光标不动、[注入]=0（此段 DTW 出非 idle 记『DTW 未纳入门』）"),
        (6, "语境B 进入：掌心朝镜 + 五指张开 保持 1 秒（此后才允许出现光标动作）"),
        (8, "第1步 1/3：整只手留在画面内，食指指向前方，慢慢移动光标"),
        (5, "第1步 1/3：整只手完全出画（应退出控制态）"),
        (6, "重新进入：掌心朝镜 + 五指张开 保持 1 秒"),
        (8, "第1步 2/3：再移动光标一次"),
        (5, "第1步 2/3：整只手完全出画（应退出控制态）"),
        (6, "重新进入：掌心朝镜 + 五指张开 保持 1 秒"),
        (8, "第1步 3/3：再移动光标一次"),
        (5, "第1步 3/3：整只手完全出画（应退出控制态）"),
        (6, "重新进入：掌心朝镜 + 五指张开 保持 1 秒"),
        (45, "语境B 日常控制：手在画面内，像平时一样慢速移动光标 → 预期正常控制且无非预期事件（不做双指悬停/捏合/翻页）"),
        (10, "第3步：食指指住一处 → 捏合 → 松开（只做一次）"),
        (12, "第4步：捏住并保持 6 秒以上再松开（只做一次）"),
    ]
    for shape in ("W", "C", "X"):
        steps.append((14, f"第5步 {shape}：双指悬停 ~300ms → 画 {shape} → 收指；不要捏合"))
        steps.append((6, f"第5步 {shape}：现在捏合确认"))
    steps += [
        (15, "第6步：用手掌或纸板完全盖住镜头 3 秒，然后拿开（手出画应退出控制态）"),
        (45, "第6步：到设备管理器禁用摄像头，再启用（禁用设备，非拔 USB——两者对引擎是不同的外设事件）；然后手回画面动一下（需重新进入）"),
        (5, "结束：保持不动，等待诊断输出"),
    ]
    return steps


def build_gate_diag_steps() -> list[tuple[float, str]]:
    """进入门定性定点：30s 内 6 段固定姿势，一次覆盖 A-F 所有断点。

    每段 5s，操作者只做一个动作，减少变量。分段严格对应诊断日志里的时序：
      0–5s   正确进入姿势 → 验证门能否打开、handedness 报什么
      5–10s  手背朝镜 → 验证误进防护
      10–15s 侧掌 → 验证 palm_frontal_ratio 是否压低
      15–20s 半握 → 验证 finger_exts 哪些 False
      20–25s 单食指伸出 → 验证是否走 partial 抢占
      25–30s 手出画 → 验证 lost 是否触发（看 [状态] 日志）
    """

    return [
        (5, "进入门诊断 1/6：掌心朝镜 + 五指张开，保持不动（验证门能否打开 + handedness 报什么）"),
        (5, "进入门诊断 2/6：手背朝镜，保持不动（应始终 gate_idle）"),
        (5, "进入门诊断 3/6：侧掌（掌面垂直镜头），保持不动（palm_frontal_ratio 应压低）"),
        (5, "进入门诊断 4/6：半握，保持不动（finger_exts 应有 False）"),
        (5, "进入门诊断 5/6：只伸食指，其余握拳（验证是否走 partial 抢占）"),
        (5, "进入门诊断 6/6：手完全移出画面（验证 lost 是否触发）"),
    ]


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
