"""一欧元滤波器（One Euro Filter）。

思路：低速时用很小的截止频率做重滤波（消抖），
高速时根据速度自动放大截止频率（保证跟随不滞后）。
普通摄像头的指尖信号用它做平滑，比固定窗均值延迟低得多。
"""

from __future__ import annotations

import math
import time

import numpy as np


class _LowPass:
    def __init__(self) -> None:
        self._value: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = alpha * x + (1.0 - alpha) * self._value
        return self._value

    def reset(self) -> None:
        self._value = None


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.2,
        beta: float = 0.08,
        dcutoff: float = 1.0,
        dim: int = 2,
        freq: float = 30.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.dim = int(dim)
        self.freq = float(freq)
        self._x_filters = [_LowPass() for _ in range(self.dim)]
        self._d_filters = [_LowPass() for _ in range(self.dim)]
        self._last_time: float | None = None
        self._last_x: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return dt / (dt + tau)

    def reset(self) -> None:
        for f in self._x_filters:
            f.reset()
        for f in self._d_filters:
            f.reset()
        self._last_time = None
        self._last_x = None

    def __call__(self, x, timestamp: float | None = None) -> np.ndarray:
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self.dim:
            raise ValueError(f"OneEuroFilter 期望 {self.dim} 维输入，收到 {x.size} 维")

        now = time.perf_counter() if timestamp is None else float(timestamp)
        dt = 1.0 / self.freq if self._last_time is None else now - self._last_time
        dt = max(min(dt, 0.2), 1e-3)  # 卡帧时抑制 dt 爆掉
        self._last_time = now

        if self._last_x is None:
            self._last_x = x.copy()

        derivative = (x - self._last_x) / dt
        out = np.empty(self.dim, dtype=float)
        for i in range(self.dim):
            d_hat = self._d_filters[i](derivative[i], self._alpha(self.dcutoff, dt))
            cutoff = self.min_cutoff + self.beta * abs(d_hat)
            out[i] = self._x_filters[i](x[i], self._alpha(cutoff, dt))

        self._last_x = x.copy()
        return out
