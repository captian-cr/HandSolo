"""日常使用的行为记录器。

为什么需要它
------------
2026-10-21 用户原话：
> 「我先用这套系统在屏幕上日常工作一下，然后你来记录，看我操作过程中
>   在同一个地方反复操作、反复尝试，说明有问题，还要改善」

靠人回忆"哪里不好用"容易漏，靠我猜更不靠谱。所以把真实使用过程逐帧记下来，
再用 `scripts/analyze_usage.py` 找**同一小片区域里反复出现的事件**——
那正是手感卡住、点不中的地方。

记录内容（每行一条 JSON，追加写）
----------------------------------
``t``        相对开始的秒数
``x``/``y``  光标归一化坐标（0~1，多屏按虚拟桌面归一）
``g``        手势名（pointer / left_pinch / drag / lost …）
``p``        是否暂停
``r``        捏合比（dist(4,8)/掌宽），排查"捏了但没识别"的关键量
``c``        本帧非移动类指令（left_click / left_double_click / left_down …）
``f``        光标此刻是否被冻结（停住/锚定）

刻意**不记图像、不记窗口标题**——只记控制信号，避免隐私问题。
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path


class UsageLogger:
    """线程安全的 JSONL 记录器；失败时静默降级，绝不影响主回路。"""

    def __init__(self, directory: Path, *, flush_every: int = 20) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"usage-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self._handle = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._started = time.monotonic()
        self._pending = 0
        self._flush_every = max(1, int(flush_every))
        self.count = 0
        self._broken = False

    def log(
        self,
        *,
        x: float,
        y: float,
        gesture: str,
        paused: bool,
        pinch: float,
        commands: tuple[str, ...] = (),
        frozen: bool = False,
    ) -> None:
        if self._broken:
            return
        record = {
            "t": round(time.monotonic() - self._started, 4),
            "x": round(float(x), 4),
            "y": round(float(y), 4),
            "g": gesture,
            "p": 1 if paused else 0,
            "r": round(float(pinch), 4),
            "f": 1 if frozen else 0,
        }
        if commands:
            record["c"] = list(commands)
        try:
            with self._lock:
                self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                self.count += 1
                self._pending += 1
                if self._pending >= self._flush_every:
                    self._handle.flush()
                    self._pending = 0
        except Exception:
            # 记录器绝不能把主回路带崩：一次失败就永久关闭。
            self._broken = True

    def flush(self) -> None:
        """立刻落盘。进程被强杀时 finally 不一定跑到，主循环可周期性调用。"""

        if self._broken:
            return
        try:
            with self._lock:
                self._handle.flush()
                self._pending = 0
        except Exception:
            self._broken = True

    def close(self) -> None:
        try:
            with self._lock:
                self._handle.flush()
                self._handle.close()
        except Exception:
            pass
