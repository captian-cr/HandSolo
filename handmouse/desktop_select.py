"""直接操作桌面真实图标：手指指到哪个图标就选哪个。

启动：``python -m handmouse.desktop_select``

与 ``grid_select`` 的分工
-------------------------
* ``grid_select``：自己铺格子，条目来自配置文件，适合"启动器"。
* ``desktop_select``（本文件）：读桌面图标的真实坐标，在你自己的桌面上工作，
  不新建任何图标。桌面怎么摆，就是怎么用。

不注入连续光标
--------------
和网格模式一样，只在命中时高亮 + 捏合确认，不移动系统光标。因此帧率低
（当前摄像头 15fps 甚至更低）也不影响"选得中"，只影响球跟手的顺滑度。
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path

from handmouse import grid_winctl as winctl
from handmouse.desktop_ui import DesktopIconMap, DesktopOverlay
from handmouse.grid_desktop import DesktopIcon, launch_icon, read_desktop_icons
from handmouse.grid_judge import ClickGate, PinchJudger, TargetStabilizer
from handmouse.grid_map import map_roi, scale_roi
from handmouse.grid_select import DEFAULT_CONFIG_PATH, HandPoller, load_config
from handmouse.grid_ui import PointerState
from handmouse.uia_targets import click_target, list_targets

#: 默认命中距离（屏幕像素）。图标约 94px 宽，取 1.3 倍左右。
DEFAULT_PICK_PX = 120.0


@dataclass(slots=True)
class DesktopRunner:
    config: dict
    icons: list[DesktopIcon]
    icon_map: DesktopIconMap
    judge: PinchJudger
    stabilizer: TargetStabilizer
    state: PointerState
    stop_event: threading.Event
    dry_run: bool = False
    screen_w: int = 1920
    screen_h: int = 1080
    _message_until_ms: float = field(default=0.0, init=False)
    _locked_target: int | None = field(default=None, init=False)
    _click_gate: ClickGate = field(default_factory=ClickGate, init=False)
    _mode: str = field(default="desktop", init=False)
    _exclude_hwnd: int = field(default=0, init=False)
    _tab_down: bool = field(default=False, init=False)
    poller: HandPoller = field(default=None, init=False)  # type: ignore[assignment]

    # ---- 主循环 ----

    def tick(self) -> None:
        self._poll_mode_key()
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
            self._locked_target = None
            return

        self.state.hand_visible = True
        self.state.ratio = sample.ratio

        roi = scale_roi(tuple(self.config["roi"]), float(self.config["sensitivity"]))
        nx, ny = map_roi(sample.index_tip[0], sample.index_tip[1], roi)
        self.state.x = min(max(nx, 0.0), 1.0)
        self.state.y = min(max(ny, 0.0), 1.0)

        px = self.state.x * self.screen_w
        py = self.state.y * self.screen_h
        raw_target = self.icon_map.hit(px, py)
        stable_target = self.stabilizer.update(raw_target)

        triggered = self.judge.update(sample.ratio, now_ms)
        self.state.pinching = self.judge.is_closed

        # 捏合会让食指向拇指靠拢，锚点因此滑动。在闭合瞬间钉住目标。
        if self.judge.just_closed:
            self._locked_target = (
                stable_target if stable_target is not None else raw_target
            )
        elif self.judge.is_closed:
            if self._locked_target is not None:
                icon = self.icons[self._locked_target]
                drift = hypot(px - icon.center[0], py - icon.center[1])
                if drift > self.icon_map.max_pick_px * 1.8:
                    self._locked_target = None
        else:
            self._locked_target = None

        active = self._locked_target if self.judge.is_closed else stable_target
        self.state.target = active

        if triggered and active is not None:
            self._trigger(active, now_ms)

        # 双击窗口内没有等到第二次捏合：回落成单次，只提示选中，不执行。
        expired = self._click_gate.poll(now_ms)
        if expired is not None:
            self.state.message = (
                f"已选中：{self.icons[expired].name}（再捏一次执行）"
            )
            self._message_until_ms = now_ms + 1600

        if now_ms < self._message_until_ms:
            return
        if active is None:
            self.state.message = (
                "捏合时目标丢失" if self.judge.is_closed else "没指到图标"
            )
        elif self._click_gate.armed and self._click_gate.pending_target is not None:
            pending = self.icons[self._click_gate.pending_target].name
            self.state.message = f"{pending}　再捏一次执行"
        else:
            self.state.message = f"指向：{self.icons[active].name}　捏合两次执行"

    def _trigger(self, index: int, now_ms: float) -> None:
        icon = self.icons[index]
        if self._click_gate.register(index, now_ms) != "double":
            self.state.message = f"{icon.name}　再捏一次执行"
            self._message_until_ms = now_ms + 800
            return

        if self.dry_run:
            self.state.message = f"[dry-run] 已确认：{icon.name}"
            self._message_until_ms = now_ms + 1500
            return

        if self._mode == "window":
            click_target(icon)
            self.state.message = f"已点击：{icon.name}"
            self._message_until_ms = now_ms + 1200
            return

        ok, message = launch_icon(icon)
        self.state.message = message
        self._message_until_ms = now_ms + 1800
        if not ok:
            # 虚拟项（此电脑 / 回收站）没有对应文件，保持提示即可。
            self._locked_target = None

    # ---- 模式切换 ----

    def _poll_mode_key(self) -> None:
        """F2 在"桌面图标"与"当前窗口控件"之间切换。

        不用 Tab：Tab 在正常打字/切焦点时太常用，全局轮询会被误触发。
        """

        try:
            down = bool(ctypes.windll.user32.GetAsyncKeyState(0x71) & 0x8000)  # VK_F2
        except Exception:
            return
        if down and not self._tab_down:
            self._switch_mode()
        self._tab_down = down

    def _switch_mode(self) -> None:
        now_ms = time.monotonic() * 1000.0
        if self._mode == "desktop":
            targets = self._load_window_targets()
            if not targets:
                self.state.message = "当前窗口没有读到可点元素（可能是游戏/自绘界面）"
                self._message_until_ms = now_ms + 2000
                return
            self._mode = "window"
            self.state.message = f"窗口模式：{len(targets)} 个可点元素　F2 返回"
        else:
            targets = [icon for icon in read_desktop_icons() if icon.name]
            self._mode = "desktop"
            self.state.message = f"桌面模式：{len(targets)} 个图标　F2 切换"

        self.icons = targets
        self.icon_map.icons = targets
        self.state.target = None
        self._locked_target = None
        self.stabilizer.reset()
        self._click_gate.reset()
        self._message_until_ms = now_ms + 2500

    def _load_window_targets(self) -> list:
        hwnd = winctl.foreground_window(exclude=self._exclude_hwnd)
        if not hwnd:
            return []
        return list_targets(hwnd)


def make_judge(config: dict) -> PinchJudger:
    return PinchJudger(
        enter_ratio=float(config["pinch_enter"]),
        release_ratio=float(config["pinch_release"]),
        hold_ms=float(config["pinch_hold_ms"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m handmouse.desktop_select",
        description="用手指直接选中桌面上的真实图标，不新建面板。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--pick-px", type=float, default=DEFAULT_PICK_PX,
        help=f"命中距离上限（屏幕像素），默认 {DEFAULT_PICK_PX:.0f}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不开界面，只在控制台打印命中的图标",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    if args.camera is not None:
        config["camera_index"] = args.camera
    if args.no_mirror:
        config["mirror"] = False

    # 先声明 DPI 感知再取屏幕尺寸，否则拿到的是缩放后的逻辑像素，
    # 与桌面图标的物理像素坐标不一致，命中会整体偏移。
    from handmouse.screen_map import enable_dpi_awareness

    enable_dpi_awareness()
    screen_w = int(ctypes.windll.user32.GetSystemMetrics(0))
    screen_h = int(ctypes.windll.user32.GetSystemMetrics(1))

    icons = read_desktop_icons()
    if not icons:
        print(
            "读不到桌面图标。可能原因：\n"
            "  1) 桌面图标被隐藏（右键桌面 → 查看 → 显示桌面图标）\n"
            "  2) 与资源管理器不同权限等级，无法读取其内存\n"
            "  3) 桌面由第三方工具（如 Fences）接管",
            file=sys.stderr,
        )
        return 1

    named = [icon for icon in icons if icon.name]
    icon_map = DesktopIconMap(named, max_pick_px=args.pick_px)
    print(
        f"[桌面] 读取到 {len(named)} 个图标，命中上限 {args.pick_px:.0f}px",
        file=sys.stderr,
    )
    for icon in named:
        print(
            f"        [{icon.index:2d}] {icon.name}  "
            f"({icon.left},{icon.top})-({icon.right},{icon.bottom})",
            file=sys.stderr,
        )

    stop_event = threading.Event()
    runner = DesktopRunner(
        config=config,
        icons=named,
        icon_map=icon_map,
        judge=make_judge(config),
        stabilizer=TargetStabilizer(confirm_frames=int(config["confirm_frames"])),
        state=PointerState(),
        stop_event=stop_event,
        dry_run=args.dry_run,
        screen_w=screen_w,
        screen_h=screen_h,
    )
    runner.poller = HandPoller(config, stop_event)

    if args.dry_run:
        runner.poller.start()
        try:
            while not stop_event.is_set():
                runner.tick()
                if runner.state.hand_visible:
                    name = (
                        named[runner.state.target].name
                        if runner.state.target is not None
                        else "--"
                    )
                    print(
                        f"x={runner.state.x:.3f} y={runner.state.y:.3f} "
                        f"ratio={runner.state.ratio:.2f} 图标={name} "
                        f"捏合={runner.state.pinching}",
                        flush=True,
                    )
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            runner.poller.join(timeout=2.0)
        return 0

    overlay = DesktopOverlay(icon_map, runner.state)
    runner.screen_w = overlay.screen_w
    runner.screen_h = overlay.screen_h
    runner.state.message = "正在加载手部模型…"
    if not overlay.show():
        stop_event.set()
        return 1
    # 枚举窗口目标时要排除自己的叠加层，否则会把自己当成前台窗口。
    runner._exclude_hwnd = overlay.hwnd
    runner.poller.start()
    try:
        overlay.run(runner.tick)
    finally:
        stop_event.set()
        runner.poller.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
