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

import threading
import time


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


def start_prompter(stop_event: threading.Event) -> threading.Thread:
    """后台线程：按节拍打印 ``[步骤] HH:MM:SS 提示语``，播完即止。"""

    steps = build_steps()

    def _run() -> None:
        for index, (duration, prompt) in enumerate(steps, start=1):
            if stop_event.is_set():
                return
            print(f"[步骤] {time.strftime('%H:%M:%S')} ({index}/{len(steps)}) {prompt}")
            stop_event.wait(duration)

    thread = threading.Thread(target=_run, name="guided-prompter", daemon=True)
    thread.start()
    return thread
