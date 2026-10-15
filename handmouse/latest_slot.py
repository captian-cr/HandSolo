"""容量为1的线程安全最新值缓冲。"""

from __future__ import annotations

import threading
import time
from typing import Generic, TypeVar


T = TypeVar("T")


class LatestSlot(Generic[T]):
    """新值覆盖旧值，避免实时流水线积压产生延迟。"""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: T | None = None
        self._revision = 0
        self.overwrite_count = 0

    def put(self, value: T) -> int:
        with self._condition:
            self._value = value
            self._revision += 1
            self._condition.notify_all()
            return self._revision

    def get_latest(self) -> tuple[T | None, int]:
        with self._condition:
            return self._value, self._revision

    def wait_next(
        self,
        after_revision: int,
        timeout: float | None = None,
    ) -> tuple[T | None, int]:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while self._revision <= after_revision:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None, after_revision
                self._condition.wait(remaining)
            if self._revision > after_revision + 1:
                self.overwrite_count += self._revision - after_revision - 1
            return self._value, self._revision

    def clear(self) -> None:
        with self._condition:
            self._value = None
