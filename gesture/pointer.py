"""指向控制器：滤波 → 映射 → 光标移动 → 捏合左键 / 握拳暂停。

第一阶段只做「食指光标 + 捏合左键」，
HOVER / DRAG / GESTURE / LAUNCH 等状态机留到下一阶段叠加。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import winmouse
from .config import Config
from .filters import OneEuroFilter
from .hand_tracker import HandObservation
from .mapping import ScreenMapper


class PointerController:
    def __init__(self, cfg: Config, mapper: ScreenMapper) -> None:
        self.cfg = cfg
        self.mapper = mapper
        self.filter = OneEuroFilter(
            min_cutoff=cfg.filter_min_cutoff,
            beta=cfg.filter_beta,
            dim=2,
        )

        # 默认不接管鼠标：先看骨架和读数，确认无误再按 M 开启
        self.enabled = False
        self.frozen = False

        self._pinched = False
        self._on_frames = 0
        self._off_frames = 0
        self._fist_since: float | None = None
        self._last_cursor: tuple[float, float] | None = None

        # 调参用统计
        self.ratio_ema: float | None = None
        self.ratio_min = float("inf")
        self.ratio_max = 0.0
        self.clicks = 0
        self.click_ratios: list[float] = []
        self.last_click_pos: tuple[float, float] | None = None

    # ---------------- 开关 ----------------
    def set_enabled(self, enabled: bool) -> None:
        if self.enabled and not enabled:
            self._release()
        self.enabled = enabled

    def toggle(self) -> bool:
        self.set_enabled(not self.enabled)
        return self.enabled

    def reset(self) -> None:
        self.filter.reset()
        self._release()
        self._fist_since = None
        self.frozen = False
        self._last_cursor = None
        self.ratio_ema = None
        self.ratio_min = float("inf")
        self.ratio_max = 0.0

    def shutdown(self) -> None:
        """退出前务必调用：防止左键被"按住不放"卡在系统里。"""
        self._release()

    def _release(self) -> None:
        if self._pinched:
            winmouse.left_up()
            self._pinched = False
        self._on_frames = 0
        self._off_frames = 0

    # ---------------- 主循环 ----------------
    def update(self, obs: Optional[HandObservation], now: float) -> dict:
        info: dict = {
            "cam_point": None,
            "screen": None,
            "pointer_source": "-",
            "ratio": None,
            "pinched": self._pinched,
            "frozen": self.frozen,
            "active": False,
            "reason": "",
            "clicks": self.clicks,
        }

        if obs is None:
            self.filter.reset()
            self._release()
            self._fist_since = None
            self.frozen = False
            self._last_cursor = None
            info["reason"] = "no-hand"
            return info

        # 1) 捏合比值：原始值 → EMA，抑制单帧跳变
        raw = obs.pinch_ratio
        self.ratio_ema = raw if self.ratio_ema is None else 0.5 * raw + 0.5 * self.ratio_ema
        self.ratio_min = min(self.ratio_min, self.ratio_ema)
        self.ratio_max = max(self.ratio_max, self.ratio_ema)
        info["ratio"] = self.ratio_ema
        info["pointer_source"] = obs.pointer_source

        # 2) 握拳暂停：捏合期间不冻结，否则点击会卡在半按状态
        fist_like = obs.is_fist and self.ratio_ema > self.cfg.pinch_off_ratio
        if fist_like:
            if self._fist_since is None:
                self._fist_since = now
            elif now - self._fist_since >= self.cfg.fist_hold_seconds:
                self.frozen = True
        else:
            self._fist_since = None
            self.frozen = False
        info["frozen"] = self.frozen

        # 3) 滤波 + 映射。滤波始终运行，保证重新启用时光标不带跳变。
        point = self.filter(np.asarray(obs.pointer, dtype=float), now)
        info["cam_point"] = point

        if self.mapper.ready:
            sx, sy = self.mapper.to_screen(point)
            sx = float(np.clip(sx, 0, self.mapper.screen_w - 1))
            sy = float(np.clip(sy, 0, self.mapper.screen_h - 1))
            info["screen"] = (sx, sy)

        # 4) 捏合迟滞状态机
        self._step_pinch(info)

        # 5) 移动光标（死区 + 夹紧）
        if self.enabled and not self.frozen and info["screen"] is not None:
            sx, sy = info["screen"]
            if (
                self._last_cursor is None
                or abs(sx - self._last_cursor[0]) >= self.cfg.pointer_dead_zone_px
                or abs(sy - self._last_cursor[1]) >= self.cfg.pointer_dead_zone_px
            ):
                winmouse.move_to(sx, sy)
                self._last_cursor = (sx, sy)
            info["active"] = True

        return info

    def _step_pinch(self, info: dict) -> None:
        ratio = self.ratio_ema
        if ratio is None:
            return

        if self._pinched:
            if ratio > self.cfg.pinch_off_ratio:
                self._off_frames += 1
                if self._off_frames >= self.cfg.pinch_stable_frames:
                    winmouse.left_up()
                    self._pinched = False
                    self._off_frames = 0
            else:
                self._off_frames = 0
        else:
            if ratio < self.cfg.pinch_on_ratio:
                self._on_frames += 1
                if self._on_frames >= self.cfg.pinch_stable_frames:
                    self._on_frames = 0
                    if self.enabled and not self.frozen:
                        winmouse.left_down()
                        self._pinched = True
                        self.clicks += 1
                        self.click_ratios.append(ratio)
                        self.last_click_pos = info["screen"]
                        pos = info["screen"]
                        pos_text = f"({pos[0]:.0f},{pos[1]:.0f})" if pos else "n/a"
                        print(f"[click #{self.clicks}] pinch_ratio={ratio:.3f} screen={pos_text}", flush=True)
            else:
                self._on_frames = 0

        info["pinched"] = self._pinched
