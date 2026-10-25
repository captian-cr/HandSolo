"""指针模式：悬浮环驱动系统光标 + 磁吸 + 双击确认。

三个关键机制
------------
1. **双击延迟确认**：手势捏合一次要 100ms 才成立，两次捏合的实际间隔常超过
   系统的 500ms 双击窗口，位置也会偏移几像素——所以系统根本不认为这是双击，
   这是"双击有时打不开"的真正原因。这里改成：第一次捏合**先不发点击**，
   等 ``double_click_window_ms``；窗口内出现第二次就主动连发两次点击（间隔
   20ms，必被识别为双击），超时才补发单击。
2. **磁吸带滞回**：进入吸附半径就吸到目标中心，离开更大的半径才脱开。
   只有一个半径的话，光标会在边界来回跳。
3. **捏合冻结光标**：捏合时食指向拇指靠会带动锚点，不冻结就点偏。

磁吸的目标池
------------
后台线程每 1.5 秒刷新一次：前台是普通窗口就枚举它的可点控件；前台是桌面就
读桌面图标。**任务栏读不到单个图标**（Win11 任务栏是 XAML 自绘，UIA 只能
拿到"运行中的应用程序"这一整块区域），所以任务栏上没有吸附点——那里仍然
靠光标直接点。
"""

from __future__ import annotations

import argparse
import ctypes
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from handmouse.grid_judge import PinchJudger
from handmouse.grid_map import map_roi, scale_roi
from handmouse.grid_select import DEFAULT_CONFIG_PATH, HandPoller, load_config
from handmouse.grid_ui import PointerState
from handmouse.pointer_ui import PointerOverlay

_user32 = ctypes.windll.user32

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

VK_M = 0x4D

#: 前台窗口属于这些类名时，认为用户在看桌面，磁吸目标改读桌面图标。
DESKTOP_CLASSES = {"Progman", "WorkerW"}


def _send_click(double: bool = False) -> None:
    """发送左键点击。双击时连发两次，间隔 20ms 保证被系统识别为双击。"""

    repeats = 2 if double else 1
    for _ in range(repeats):
        _user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        _user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        if double:
            time.sleep(0.02)


class TargetPoller(threading.Thread):
    """后台刷新磁吸目标池。

    枚举 UIA 要几百毫秒，绝不能放在 60fps 的渲染循环里。
    """

    def __init__(self, exclude_hwnd: int, stop_event: threading.Event,
                 interval_s: float = 1.5) -> None:
        super().__init__(name="pointer-targets", daemon=True)
        self.exclude_hwnd = exclude_hwnd
        self.stop_event = stop_event
        self.interval_s = interval_s
        self._lock = threading.Lock()
        self._targets: list = []
        self._source = ""

    def snapshot(self) -> tuple[list, str]:
        with self._lock:
            return list(self._targets), self._source

    def refresh_once(self) -> None:
        from handmouse import grid_winctl as winctl
        from handmouse.grid_desktop import read_desktop_icons
        from handmouse.uia_targets import list_targets

        hwnd = winctl.foreground_window(exclude=self.exclude_hwnd)
        if not hwnd:
            with self._lock:
                self._targets, self._source = [], "无前台窗口"
            return

        buffer = ctypes.create_unicode_buffer(256)
        _user32.GetClassNameW(ctypes.c_void_p(hwnd), buffer, 256)
        class_name = buffer.value

        if class_name in DESKTOP_CLASSES:
            targets = [icon for icon in read_desktop_icons() if icon.name]
            source = f"桌面 {len(targets)} 个"
        else:
            targets = list_targets(hwnd)
            source = f"{winctl.window_title(hwnd)[:16]} {len(targets)} 个"

        with self._lock:
            self._targets = targets
            self._source = source

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.refresh_once()
            except Exception as exc:  # 目标池失败不能影响光标主流程
                with self._lock:
                    self._source = f"刷新失败: {exc}"
            # 分段睡眠，便于及时响应退出。
            waited = 0.0
            while waited < self.interval_s and not self.stop_event.is_set():
                time.sleep(0.1)
                waited += 0.1


@dataclass(slots=True)
class PointerRunner:
    config: dict
    judge: PinchJudger
    state: PointerState
    stop_event: threading.Event
    screen_w: int = 1920
    screen_h: int = 1080
    dry_run: bool = False
    _last_t: float = field(default=0.0, init=False)
    _sx: float = field(default=0.5, init=False)
    _sy: float = field(default=0.5, init=False)
    _frozen: tuple[int, int] | None = field(default=None, init=False)
    _m_down: bool = field(default=False, init=False)
    _clicks: int = field(default=0, init=False)
    _pending: tuple[int, int, float] | None = field(default=None, init=False)
    _snapped: object | None = field(default=None, init=False)
    _targets: list = field(default_factory=list, init=False)
    _target_source: str = field(default="", init=False)
    _action_message: str = field(default="", init=False)
    _action_until: float = field(default=0.0, init=False)
    _under: str = field(default="", init=False)
    _under_is_desktop: bool = field(default=False, init=False)
    _last_raw: tuple[float, float] | None = field(default=None, init=False)
    _jump_streak: int = field(default=0, init=False)
    _desktop_icons: list = field(default_factory=list, init=False)
    _desktop_icons_ms: float = field(default=0.0, init=False)
    _exclude_hwnd: int = field(default=0, init=False)
    _frame: int = field(default=0, init=False)
    poller: HandPoller = field(default=None, init=False)  # type: ignore[assignment]
    target_poller: TargetPoller | None = field(default=None, init=False)

    # ---- 主循环 ----

    def tick(self) -> None:
        self._poll_pause_key()

        sample, fps, error = self.poller.snapshot()
        self.state.fps = fps
        if error:
            self.state.message = error

        if self.target_poller is not None:
            self._targets, self._target_source = self.target_poller.snapshot()

        now = time.monotonic()
        self._expire_pending(now)

        if sample is None:
            self.state.hand_visible = False
            self.judge.reset()
            self._frozen = None
            return

        self.state.hand_visible = True
        self.state.ratio = sample.ratio

        dt = min(max(now - self._last_t, 0.0), 0.25) if self._last_t else 0.0
        self._last_t = now

        roi = scale_roi(tuple(self.config["roi"]), float(self.config["sensitivity"]))
        nx, ny = map_roi(sample.index_tip[0], sample.index_tip[1], roi)
        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        triggered = self.judge.update(sample.ratio, now * 1000.0)
        self.state.pinching = self.judge.is_closed

        if self.state.paused:
            self._frozen = None
            self.state.pending = self._pending is not None
            return

        if self.judge.is_closed:
            # 捏合期间锁定光标：食指往拇指靠会带动锚点，跟随就会点偏。
            if self._frozen is None:
                self._frozen = (
                    int(self._sx * self.screen_w),
                    int(self._sy * self.screen_h),
                )
        else:
            self._frozen = None
            # 跳变抑制：关键点偶发跳到画面另一头时，单帧位移会极大。
            # 这种帧直接丢弃，否则光标会"瞬移"一下再飘回来。
            # 连续多帧都跳说明手真的在快速移动，此时放行。
            accept = True
            if self._last_raw is not None:
                jump = math.hypot(nx - self._last_raw[0], ny - self._last_raw[1])
                if jump > 0.28:
                    self._jump_streak += 1
                    accept = self._jump_streak >= 3
                else:
                    self._jump_streak = 0
            self._last_raw = (nx, ny)
            if accept:
                tau = max(0.005, float(self.config.get("pointer_smoothing_s", 0.06)))
                alpha = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
                self._sx += (nx - self._sx) * alpha
                self._sy += (ny - self._sy) * alpha

        if self._frozen is not None:
            px, py = self._frozen
        else:
            px = int(self._sx * self.screen_w)
            py = int(self._sy * self.screen_h)
            px, py = self._apply_snap(px, py)

        # 球必须画在**光标的实际位置**上。此前用的是平滑后的手指坐标，
        # 于是捏合冻结光标时球还在跟着手跑——用户看到球在图标上，而真正
        # 的光标停在别处，点击自然打空。
        self.state.x = px / self.screen_w
        self.state.y = py / self.screen_h

        if not self.dry_run:
            _user32.SetCursorPos(px, py)

        # 每 6 帧更新一次"光标下面是哪个窗口"，用于诊断双击打空。
        self._frame += 1
        if self._frame % 6 == 0:
            from handmouse import grid_winctl as _winctl

            under = _winctl.window_at(px, py, self._exclude_hwnd)
            self._under = _winctl.window_title(under) if under else ""
            # 桌面窗口（Progman/WorkerW）标题为空。标题非空说明有别的窗口
            # 压在上面，此时双击会打到那个窗口上。
            self._under_is_desktop = bool(under) and not self._under

        if triggered:
            self._on_confirm(px, py, now)

        self.state.pending = self._pending is not None
        self.state.snapped = self._snapped is not None
        if now < self._action_until:
            self.state.message = self._action_message
        elif self._pending is not None:
            self.state.message = "已选中：再捏一次双击（超时则单击）"
        elif self._snapped is not None:
            name = getattr(self._snapped, "name", "目标")
            self.state.message = f"吸附：{name}　{self._target_source}"
        else:
            self.state.message = (
                f"已点击 {self._clicks} 次　{self._target_source}"
            )

        # 诊断：显示光标下面究竟是哪个窗口。桌面图标被遮挡时，这里会直接
        # 写出遮挡者的标题（如"微信音视频通话"），不必再猜为什么点不开。
        if self._under:
            self.state.message += f"\n光标下：{self._under[:40]}"
        elif self._under_is_desktop:
            self.state.message += "\n光标下：桌面"

    # ---- 磁吸 ----

    def _apply_snap(self, px: int, py: int) -> tuple[int, int]:
        """带滞回的磁吸。

        进入用 ``snap_radius``，脱离用 ``snap_release_radius``。单一半径会让
        光标在边界反复跳进跳出。
        """

        if not self.config.get("snap_enabled", True) or not self._targets:
            self._snapped = None
            return px, py

        if self._snapped is not None:
            cx, cy = self._snapped.center  # type: ignore[attr-defined]
            release = float(self.config.get("pointer_snap_release_px", 50))
            if math.hypot(px - cx, py - cy) > release:
                self._snapped = None
            else:
                return int(cx), int(cy)

        radius = float(self.config.get("pointer_snap_px", 25))
        best = None
        best_distance = radius
        for target in self._targets:
            cx, cy = target.center
            distance = math.hypot(px - cx, py - cy)
            if distance < best_distance:
                best_distance = distance
                best = target
        if best is not None:
            self._snapped = best
            cx, cy = best.center
            return int(cx), int(cy)
        return px, py

    # ---- 点击 ----

    def _on_confirm(self, x: int, y: int, now: float) -> None:
        window = float(self.config.get("double_click_window_ms", 500)) / 1000.0

        if self._pending is not None:
            pending_x, pending_y, started = self._pending
            if now - started <= window:
                # 第二次捏合：主动连发两次点击，系统必然识别为双击。
                self._pending = None
                self._clicks += 1
                if not self.dry_run:
                    # 桌面图标被别的窗口盖住时（实测很常见：微信通话悬浮窗、
                    # Chrome 渲染窗口都会压在最上层），双击会打在遮挡窗口上。
                    # 这里直接按位置找到对应图标并启动，绕开遮挡。
                    opened = (
                        None if self._under_is_desktop
                        else self._open_desktop_icon_at(pending_x, pending_y)
                    )
                    if opened:
                        self._action_message = opened
                        self._action_until = now + 2.0
                        return
                    _user32.SetCursorPos(pending_x, pending_y)
                    time.sleep(0.04)
                    _send_click(double=True)
                self._action_message = "双击"
                self._action_until = now + 1.0
                return
            # 已超时，先把上一次的单击补发掉，本次重新计时。
            self._flush_single(now)

        self._pending = (int(x), int(y), now)

    def _open_desktop_icon_at(self, x: int, y: int) -> str | None:
        """光标位置若落在某个桌面图标内，直接启动它并返回提示。

        缓存 5 秒：读桌面图标要走 SysListView32 的跨进程内存，双击时才可能
        用到，没必要每帧都读。
        """

        now = time.monotonic()
        if now - self._desktop_icons_ms > 5.0:
            try:
                from handmouse.grid_desktop import read_desktop_icons
                self._desktop_icons = [
                    icon for icon in read_desktop_icons() if icon.name
                ]
            except Exception:
                self._desktop_icons = []
            self._desktop_icons_ms = now

        for icon in self._desktop_icons:
            if icon.left <= x <= icon.right and icon.top <= y <= icon.bottom:
                try:
                    from handmouse.grid_desktop import launch_icon
                    ok, message = launch_icon(icon)
                except Exception as exc:
                    return f"启动失败：{exc}"
                return message if ok else None
        return None

    def _expire_pending(self, now: float) -> None:
        if self._pending is not None:
            pending_x, pending_y, started = self._pending
            window = float(self.config.get("double_click_window_ms", 500)) / 1000.0
            if now - started > window:
                self._flush_single(now)

    def _flush_single(self, now: float) -> None:
        if self._pending is None:
            return
        pending_x, pending_y, _ = self._pending
        self._pending = None
        self._clicks += 1
        if not self.dry_run:
            _user32.SetCursorPos(pending_x, pending_y)
            time.sleep(0.04)
            _send_click(double=False)
        self._action_message = "单击"
        self._action_until = now + 1.0

    # ---- 按键 ----

    def _poll_pause_key(self) -> None:
        try:
            down = bool(_user32.GetAsyncKeyState(VK_M) & 0x8000)
        except Exception:
            return
        if down and not self._m_down:
            self.state.paused = not self.state.paused
            if self.state.paused:
                self.state.message = "已暂停：光标不再跟随"
            else:
                self._sx = self.state.x
                self._sy = self.state.y
                self.state.message = "已恢复"
        self._m_down = down


def make_judge(config: dict) -> PinchJudger:
    return PinchJudger(
        enter_ratio=float(config["pinch_enter"]),
        release_ratio=float(config["pinch_release"]),
        hold_ms=float(config["pinch_hold_ms"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m handmouse.pointer_select",
        description="用手驱动系统光标：悬浮环 + 磁吸 + 单击/双击。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--no-snap", action="store_true", help="关闭磁吸")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不开界面、不移动光标，只打印坐标",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    if args.camera is not None:
        config["camera_index"] = args.camera
    if args.no_mirror:
        config["mirror"] = False
    if args.no_snap:
        config["snap_enabled"] = False

    from handmouse.screen_map import enable_dpi_awareness

    enable_dpi_awareness()
    screen_w = int(_user32.GetSystemMetrics(0))
    screen_h = int(_user32.GetSystemMetrics(1))

    stop_event = threading.Event()
    state = PointerState()
    runner = PointerRunner(
        config=config,
        judge=make_judge(config),
        state=state,
        stop_event=stop_event,
        screen_w=screen_w,
        screen_h=screen_h,
        dry_run=args.dry_run,
    )
    runner.poller = HandPoller(config, stop_event)

    print(
        f"[指针] 屏幕 {screen_w}x{screen_h}　"
        f"灵敏度 {float(config['sensitivity']):.2f}　"
        f"平滑 {float(config.get('pointer_smoothing_s', 0.06)):.3f}s　"
        f"磁吸 {'开' if config.get('snap_enabled', True) else '关'}　"
        f"双击窗口 {int(config.get('double_click_window_ms', 500))}ms",
        file=sys.stderr,
    )

    if args.dry_run:
        runner.target_poller = TargetPoller(0, stop_event)
        runner.poller.start()
        runner.target_poller.start()
        try:
            while not stop_event.is_set():
                runner.tick()
                if state.hand_visible:
                    print(
                        f"x={state.x:.3f} y={state.y:.3f} "
                        f"ratio={state.ratio:.2f} 捏合={state.pinching} "
                        f"吸={state.snapped} 待定={state.pending} | {state.message}",
                        flush=True,
                    )
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            runner.poller.join(timeout=2.0)
        return 0

    overlay = PointerOverlay(state)
    state.message = "正在加载手部模型…"
    if not overlay.show():
        stop_event.set()
        return 1
    runner.screen_w = overlay.screen_w
    runner.screen_h = overlay.screen_h

    # 诊断"光标下面是哪个窗口"时要跳过自己的叠加层。
    runner._exclude_hwnd = overlay.hwnd
    runner.target_poller = TargetPoller(overlay.hwnd, stop_event)
    runner.target_poller.start()
    runner.poller.start()
    try:
        overlay.run(runner.tick, interval_ms=16)
    finally:
        stop_event.set()
        runner.poller.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
