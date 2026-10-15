"""坐标映射和平滑滤波。"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, pi


def clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    """将数值裁剪到指定范围。"""

    return max(minimum, min(maximum, value))


def map_activity_region(
    x: float,
    y: float,
    margin_x: float,
    margin_y: float,
) -> tuple[float, float]:
    """将摄像头活动区域线性映射到主屏幕归一化坐标。"""

    width = max(0.01, 1.0 - 2.0 * margin_x)
    height = max(0.01, 1.0 - 2.0 * margin_y)
    return (
        clamp((x - margin_x) / width),
        clamp((y - margin_y) / height),
    )


@dataclass(slots=True)
class AdaptiveEmaFilter:
    """按移动速度自动提高响应性的二维EMA滤波器。"""

    base_alpha: float = 0.30
    fast_alpha: float = 0.72
    speed_threshold: float = 0.08
    deadzone: float = 0.0025
    _x: float | None = None
    _y: float | None = None

    def reset(self) -> None:
        self._x = None
        self._y = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        x = clamp(x)
        y = clamp(y)
        if self._x is None or self._y is None:
            self._x, self._y = x, y
            return x, y

        distance = hypot(x - self._x, y - self._y)
        if distance < self.deadzone:
            return self._x, self._y

        ratio = clamp(distance / max(self.speed_threshold, 1e-6))
        alpha = self.base_alpha + (self.fast_alpha - self.base_alpha) * ratio
        self._x = alpha * x + (1.0 - alpha) * self._x
        self._y = alpha * y + (1.0 - alpha) * self._y
        return self._x, self._y


@dataclass(slots=True)
class OneEuroPointerFilter:
    """面向手部关键点的时间感知One Euro二维滤波器。

    静止时降低截止频率以吸收关键点噪声，快速移动时根据原始速度提高
    截止频率。与固定EMA相比，它不依赖摄像头帧率，低帧率下也能保持跟手。
    """

    min_cutoff: float = 1.7
    beta: float = 0.9
    derivative_cutoff: float = 1.0
    deadzone: float = 0.0006
    _x: float | None = None
    _y: float | None = None
    _raw_x: float | None = None
    _raw_y: float | None = None
    _speed: float = 0.0
    _timestamp_ns: int | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        rate = 2.0 * pi * max(0.01, cutoff) * dt
        return rate / (rate + 1.0)

    def reset(self) -> None:
        self._x = None
        self._y = None
        self._raw_x = None
        self._raw_y = None
        self._speed = 0.0
        self._timestamp_ns = None

    def update(self, x: float, y: float, timestamp_ns: int) -> tuple[float, float]:
        x = clamp(x)
        y = clamp(y)
        if self._x is None or self._y is None or self._timestamp_ns is None:
            self._x = self._raw_x = x
            self._y = self._raw_y = y
            self._timestamp_ns = timestamp_ns
            return x, y

        dt = clamp((timestamp_ns - self._timestamp_ns) / 1_000_000_000, 0.001, 0.2)
        raw_x = self._raw_x if self._raw_x is not None else x
        raw_y = self._raw_y if self._raw_y is not None else y
        raw_speed = hypot(x - raw_x, y - raw_y) / dt
        speed_alpha = self._alpha(self.derivative_cutoff, dt)
        self._speed += speed_alpha * (raw_speed - self._speed)
        # 原始速度负责快速起步，平滑速度负责避免截止频率瞬间塌陷。
        adaptive_speed = max(raw_speed * 0.65, self._speed)
        cutoff = self.min_cutoff + self.beta * adaptive_speed
        alpha = self._alpha(cutoff, dt)

        if hypot(x - self._x, y - self._y) <= self.deadzone:
            filtered_x, filtered_y = self._x, self._y
        else:
            filtered_x = self._x + alpha * (x - self._x)
            filtered_y = self._y + alpha * (y - self._y)

        self._x, self._y = filtered_x, filtered_y
        self._raw_x, self._raw_y = x, y
        self._timestamp_ns = timestamp_ns
        return filtered_x, filtered_y


@dataclass(slots=True)
class CursorMotionInterpolator:
    """在较低视觉帧率之间生成高频、可中断的光标轨迹。

    旧实现对每个视觉帧独立做 ``smoothstep``，等价于每40毫秒左右把
    光标速度降到零再重新起步。这里改用保留速度的临界阻尼系统：
    新目标到来时只改变目标位置，不清零当前速度，因此连续移动不会出现
    周期性刹停；停止动作仍会平滑收敛且不产生弹簧回摆。
    """

    min_duration_ms: int = 16
    max_duration_ms: int = 42
    _x: float | None = None
    _y: float | None = None
    _target_x: float = 0.0
    _target_y: float = 0.0
    _velocity_x: float = 0.0
    _velocity_y: float = 0.0
    _last_sample_ns: int | None = None
    _smoothing_seconds: float = 0.020
    _active: bool = False

    def clear(self) -> None:
        self._active = False
        self._x = None
        self._y = None
        self._velocity_x = 0.0
        self._velocity_y = 0.0
        self._last_sample_ns = None

    def cancel(self) -> None:
        """停止当前过渡但保留最后位置，便于短暂手势波动后平滑续接。"""

        self._active = False
        self._velocity_x = 0.0
        self._velocity_y = 0.0

    def set_target(
        self,
        x: float,
        y: float,
        timestamp_ns: int,
    ) -> tuple[float, float] | None:
        x, y = clamp(x), clamp(y)
        if self._x is None or self._y is None:
            self._x, self._y = x, y
            self._target_x, self._target_y = x, y
            self._last_sample_ns = timestamp_ns
            return x, y

        current = self.sample(timestamp_ns)
        if current is not None:
            self._x, self._y = current
        distance = hypot(x - self._x, y - self._y)
        if distance < 1e-6:
            self._active = False
            self._target_x, self._target_y = x, y
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            self._last_sample_ns = timestamp_ns
            return None

        # 大幅甩动采用更短收敛时间，小幅移动保留更强的速度连续性。
        ratio = clamp(distance / 0.15)
        duration_ms = self.max_duration_ms - (
            self.max_duration_ms - self.min_duration_ms
        ) * ratio
        self._target_x, self._target_y = x, y
        self._smoothing_seconds = max(0.006, duration_ms * 0.0007)
        self._last_sample_ns = timestamp_ns
        self._active = True
        return None

    @staticmethod
    def _smooth_axis(
        current: float,
        velocity: float,
        target: float,
        dt: float,
        smoothing_seconds: float,
    ) -> tuple[float, float]:
        """稳定近似临界阻尼解析解，并保留跨目标更新的速度。"""

        omega = 2.0 / max(0.001, smoothing_seconds)
        scaled = omega * dt
        decay = 1.0 / (
            1.0
            + scaled
            + 0.48 * scaled * scaled
            + 0.235 * scaled * scaled * scaled
        )
        change = current - target
        temporary = (velocity + omega * change) * dt
        output = target + (change + temporary) * decay
        next_velocity = (velocity - omega * temporary) * decay

        # 目标突变时也不允许越过目标后回弹。
        if (target - current > 0.0) == (output > target):
            return target, 0.0
        return output, next_velocity

    def sample(self, timestamp_ns: int) -> tuple[float, float] | None:
        if not self._active:
            return None
        if self._x is None or self._y is None:
            return None
        if self._last_sample_ns is None:
            self._last_sample_ns = timestamp_ns
            return self._x, self._y

        elapsed_ns = timestamp_ns - self._last_sample_ns
        if elapsed_ns <= 0:
            return self._x, self._y
        dt = clamp(elapsed_ns / 1_000_000_000, 0.001, 0.05)
        self._x, self._velocity_x = self._smooth_axis(
            self._x,
            self._velocity_x,
            self._target_x,
            dt,
            self._smoothing_seconds,
        )
        self._y, self._velocity_y = self._smooth_axis(
            self._y,
            self._velocity_y,
            self._target_y,
            dt,
            self._smoothing_seconds,
        )
        self._last_sample_ns = timestamp_ns

        remaining = hypot(self._target_x - self._x, self._target_y - self._y)
        speed = hypot(self._velocity_x, self._velocity_y)
        if remaining < 0.0002 and speed < 0.01:
            self._x, self._y = self._target_x, self._target_y
            self._velocity_x = 0.0
            self._velocity_y = 0.0
            self._active = False
        return self._x, self._y
