"""主题桌面入口：4×3 分页网格 + 底部三键胶囊 + 三姿态手势。

启动：``python -m handmouse.theme_select [--new-desktop]``

三种手型，互斥
--------------
* **只伸食指** → 操控：指哪选哪，捏合两次打开
* **五指张开** → 翻页：左右挥动（面板打开时翻面板的分页）
* **握拳 / 其他** → 空闲：光标隐藏，不响应任何操作

底部胶囊三个键（单击即生效，都是非破坏性导航）
----------------------------------------------
* **文件** → 弹出「桌面文件」面板（全部 / 文件夹 / 文档 / 图片）
* **HOME** → 弹出「全部应用」抽屉；面板已打开时收起
* **设置** → 循环切换壁纸

为什么必须加姿态门槛
--------------------
之前只要画面里出现手就接管光标，抬手、放下、路过的动作全成了误触。
用姿态把"操作"和"日常动作"隔开，是这版最关键的改动。

放在哪个桌面
------------
Windows 没有公开 API 能创建虚拟桌面（``IVirtualDesktopManager`` 只有查询接口），
所以 ``--new-desktop`` 用系统快捷键 ``Win+Ctrl+D`` 先建并切过去，再在那边启动：
窗口天然归属新桌面，原桌面不受影响。``Win+Ctrl+←/→`` 随时切回来。
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from handmouse.grid_judge import PinchJudger
from handmouse.grid_map import map_roi, scale_roi
from handmouse.grid_select import DEFAULT_CONFIG_PATH, HandPoller, load_config
from handmouse.grid_ui import PointerState
from handmouse.hand_pose import EXTEND_RATIO, classify_pose, finger_ratio
from handmouse.theme_config import load_theme_config
from handmouse.theme_ui import PER_PAGE, ThemeApp, ThemePanel
from handmouse.virtual_desktop import open_new_desktop

VK_M = 0x4D

#: 挥手判定：这个时间窗内掌心水平位移超过阈值算一次挥动。
SWING_WINDOW_S = 0.40
SWING_DX = 0.17

#: 两次挥动之间的冷却，避免一次挥动被拆成两次。
SWING_COOLDOWN_S = 0.55


#: 系统虚拟项没有真实文件路径，用 shell 别名启动。
#: 这些项只出现在桌面控件里，桌面文件夹中并不存在。
SHELL_ALIASES = {
    "回收站": "shell:RecycleBinFolder",
    "此电脑": "shell:MyComputerFolder",
    "电脑": "shell:MyComputerFolder",
    "网络": "shell:NetworkPlacesFolder",
    "控制面板": "shell:ControlPanelFolder",
    "用户文件夹": "shell:UsersFilesFolder",
}


def load_theme_apps(limit: int = 30) -> list[ThemeApp]:
    """读取桌面上**全部**图标，包含回收站、此电脑这类虚拟项。

    不用扫桌面文件夹，而是读 ``SysListView32`` 的条目——那样才能拿到虚拟项，
    并且顺序与用户桌面上看到的一致。虚拟项解析不到文件路径时按 ``shell:``
    别名打开。
    """

    from handmouse.grid_desktop import read_desktop_icons, resolve_icon_path

    found: list[ThemeApp] = []
    for icon in read_desktop_icons():
        name = icon.name.strip()
        if not name:
            continue
        path = resolve_icon_path(name)
        if path is not None:
            found.append(ThemeApp(name=name, path=path))
        elif name in SHELL_ALIASES:
            found.append(ThemeApp(name=name, path=SHELL_ALIASES[name]))
        if len(found) >= limit:
            break
    return found


@dataclass(slots=True)
class ThemeRunner:
    config: dict
    panel: ThemePanel
    judge: PinchJudger
    state: PointerState
    stop_event: threading.Event
    dry_run: bool = False
    _last_t: float = field(default=0.0, init=False)
    _sx: float = field(default=0.5, init=False)
    _sy: float = field(default=0.5, init=False)
    _pending: tuple[str, int, float] | None = field(default=None, init=False)
    _m_down: bool = field(default=False, init=False)
    _frozen: tuple[float, float] | None = field(default=None, init=False)
    _action: str = field(default="", init=False)
    _action_until: float = field(default=0.0, init=False)
    _last_raw: tuple[float, float] | None = field(default=None, init=False)
    _jump_streak: int = field(default=0, init=False)
    _swing_track: list[tuple[float, float]] = field(default_factory=list, init=False)
    _swing_ready: float = field(default=0.0, init=False)
    #: 最近一次指中的分类标签（``tab`` 区域没有下标，只能靠它传递）。
    _hover_tab: str = field(default="", init=False)
    poller: HandPoller = field(default=None, init=False)  # type: ignore[assignment]

    # ---- 主循环 ----

    def tick(self) -> None:
        self._poll_pause_key()

        sample, fps, error = self.poller.snapshot()
        self.state.fps = fps
        if error:
            self.state.message = error

        now = time.monotonic()
        self._expire_pending(now)

        if sample is None:
            self.state.hand_visible = False
            self.state.target = None
            self.state.pose = ""
            self.judge.reset()
            self._frozen = None
            self._swing_track.clear()
            return

        pose = classify_pose(sample.points) if sample.points else "other"
        self.state.pose = pose
        if sample.points:
            self.state.finger_hint = (
                f"阈值{EXTEND_RATIO:.2f}　"
                + " ".join(
                    f"{label}{finger_ratio(sample.points, name):.2f}"
                    for label, name in (
                        ("食", "index"), ("中", "middle"),
                        ("无", "ring"), ("小", "pinky"),
                    )
                )
            )
        else:
            self.state.finger_hint = ""
        self.state.ratio = sample.ratio
        self.state.hand_visible = pose in {"point", "open"}

        if pose == "open":
            self._handle_swing(sample, now)
            self.state.target = None
            self.judge.reset()
            self._frozen = None
            return

        self._swing_track.clear()

        if pose != "point":
            # 握拳或未识别：一律不动作，避免抬手/放下时误触。
            self.state.target = None
            self.state.pinching = False
            self.judge.reset()
            self._frozen = None
            self.state.message = "握拳中（空闲）" if pose == "fist" else ""
            return

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
            return

        if self.judge.is_closed:
            if self._frozen is None:
                self._frozen = (self._sx, self._sy)
        else:
            self._frozen = None
            # 跳变抑制：关键点偶发跳到画面另一头时，单帧位移会极大。不过滤
            # 的话光标会"瞬移"一下再飘回，看起来就像识别不准。
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
                tau = max(0.005, float(self.config.get("pointer_smoothing_s", 0.075)))
                alpha = 1.0 - math.exp(-dt / tau) if dt > 0 else 0.0
                self._sx += (nx - self._sx) * alpha
                self._sy += (ny - self._sy) * alpha

        if self._frozen is not None:
            self.state.x, self.state.y = self._frozen
        else:
            self.state.x, self.state.y = self._sx, self._sy

        px = int(self.state.x * self.panel.screen_w)
        py = int(self.state.y * self.panel.screen_h)
        # 系统光标跟着环走，两者重合。不移的话用户会看到鼠标停在原处不动，
        # 以为程序没在响应。
        if not self.dry_run:
            ctypes.windll.user32.SetCursorPos(px, py)

        where, index = self._resolve(px, py)
        self.state.target = index if where == "icon" else None
        self.state.snapped = self.state.target is not None

        if triggered:
            self._on_confirm(where, index, now)

        if now < self._action_until:
            self.state.message = self._action
        elif self._pending is not None and self._pending[0] == "icon":
            name = self.panel.current_apps()[self._pending[1]].name
            self.state.message = f"已选中「{name}」　再捏一次打开"
        elif where == "icon":
            name = self.panel.current_apps()[index].name
            self.state.message = f"指向「{name}」　捏合两次打开"
        elif where == "tab":
            self.state.message = f"分类「{self._hover_tab}」　捏一下切换"
        elif where == "folder":
            self.state.message = "文件　捏一下打开文件面板"
        elif where == "home":
            self.state.message = (
                "HOME　捏一下收起面板" if self.panel.panel_open
                else "HOME　捏一下弹出全部应用"
            )
        elif where == "settings":
            self.state.message = "设置　捏一下换壁纸"
        else:
            self.state.message = "没指到图标"

    # ---- 挥手翻页 ----

    def _handle_swing(self, sample, now: float) -> None:
        x = sample.palm[0]
        self._swing_track.append((now, x))
        while self._swing_track and now - self._swing_track[0][0] > SWING_WINDOW_S:
            self._swing_track.pop(0)

        if len(self._swing_track) < 3 or now < self._swing_ready:
            self.state.message = "五指张开　左右挥动翻页"
            return

        dx = self._swing_track[-1][1] - self._swing_track[0][1]
        if abs(dx) < SWING_DX:
            self.state.message = "五指张开　左右挥动翻页"
            return

        # 手向左挥 → 翻到下一页（内容往左走，和手机一致）。
        delta = 1 if dx < 0 else -1
        self.panel.turn_page(delta)
        self._swing_ready = now + SWING_COOLDOWN_S
        self._swing_track.clear()
        self._action = f"已切到第 {self.panel.page + 1} 页"
        self._action_until = now + 1.2

    # ---- 命中与确认 ----

    def _resolve(self, px: int, py: int) -> tuple[str, int]:
        """返回 (区域, 下标)。区域：icon / tab / folder / home / settings / none。"""

        index = self.panel.hit(px, py)
        if index is not None:
            return "icon", index

        if self.panel.panel_open:
            label = self.panel.tab_hit(px, py)
            if label:
                self._hover_tab = label
                return "tab", -1
            return "none", -1

        button = self.panel.pill_button(px, py)
        if button:
            return button, -1
        return "none", -1

    def _on_confirm(self, where: str, index: int, now: float) -> None:
        window = float(self.config.get("double_click_window_ms", 600)) / 1000.0

        # 导航类动作（HOME / 文件 / 设置 / 分类）都是非破坏性的：单手单击即生效。
        if where == "tab":
            self.panel.set_panel_tab(self._hover_tab or "全部")
            self._action = f"分类：{self._hover_tab}"
            self._action_until = now + 1.0
            self._pending = None
            return

        if where == "folder":
            self.panel.open_panel("files")
            self._action = "已打开文件面板"
            self._action_until = now + 1.2
            self._pending = None
            return

        if where == "home":
            if self.panel.panel_open:
                self.panel.close_panel()
                self._action = "已收起面板"
            else:
                self.panel.open_panel("apps")
                self._action = "已弹出全部应用"
            self._action_until = now + 1.2
            self._pending = None
            return

        if where == "settings":
            spec = self.panel.cycle_wallpaper()
            self._action = f"壁纸：{spec}"
            self._action_until = now + 1.4
            self._pending = None
            return

        if where != "icon":
            self._pending = None
            return

        if self._pending is not None:
            pending_where, pending_index, started = self._pending
            if (
                pending_where == where
                and pending_index == index
                and now - started <= window
            ):
                self._pending = None
                self._launch(index, now)
                return
            self._pending = None

        self._pending = (where, index, now)

    def _expire_pending(self, now: float) -> None:
        if self._pending is None:
            return
        window = float(self.config.get("double_click_window_ms", 600)) / 1000.0
        if now - self._pending[2] > window:
            self._pending = None

    def _launch(self, index: int, now: float) -> None:
        try:
            app = self.panel.current_apps()[index]
        except IndexError:
            return
        if self.dry_run:
            self._action = f"[dry-run] 打开「{app.name}」"
            self._action_until = now + 2.0
            return
        try:
            # shell: 别名要原样传给 Shell，不能转成 Path。
            os.startfile(app.path if isinstance(app.path, str) else str(app.path))  # noqa: S606
        except OSError as exc:
            self._action = f"启动失败：{exc}"
        else:
            self._action = f"已打开「{app.name}」"
        self._action_until = now + 2.0

    # ---- 按键 ----

    def _poll_pause_key(self) -> None:
        try:
            down = bool(ctypes.windll.user32.GetAsyncKeyState(VK_M) & 0x8000)
        except Exception:
            return
        if down and not self._m_down:
            self.state.paused = not self.state.paused
            self.state.message = "已暂停" if self.state.paused else "已恢复"
        self._m_down = down


def make_judge(config: dict) -> PinchJudger:
    return PinchJudger(
        enter_ratio=float(config["pinch_enter"]),
        release_ratio=float(config["pinch_release"]),
        hold_ms=float(config["pinch_hold_ms"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m handmouse.theme_select",
        description="VR 风格手势桌面：4×3 分页、HOME 抽屉、文件面板、挥手翻页。",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--new-desktop", action="store_true",
        help="先用 Win+Ctrl+D 新建一个虚拟桌面并在那边启动（原桌面不受影响）",
    )
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    if args.camera is not None:
        config["camera_index"] = args.camera
    if args.no_mirror:
        config["mirror"] = False

    from handmouse.screen_map import enable_dpi_awareness

    enable_dpi_awareness()

    apps = load_theme_apps()
    if not apps:
        print("桌面上没有可展示的应用（.lnk / .url / 文件夹）。", file=sys.stderr)
        return 1

    pages = max(1, math.ceil(len(apps) / PER_PAGE))
    print(f"[主题] {len(apps)} 个应用，分 {pages} 页：", file=sys.stderr)
    for app in apps:
        print(f"        {app.name}", file=sys.stderr)

    state = PointerState()
    stop_event = threading.Event()
    cfg = load_theme_config(config)
    panel = ThemePanel(apps, state, files=apps, cfg=cfg)
    runner = ThemeRunner(
        config=config,
        panel=panel,
        judge=make_judge(config),
        state=state,
        stop_event=stop_event,
        dry_run=args.dry_run,
    )
    runner.poller = HandPoller(config, stop_event)

    if args.dry_run:
        runner.poller.start()
        try:
            while not stop_event.is_set():
                runner.tick()
                if state.hand_visible or state.pose:
                    print(
                        f"姿态={state.pose:6s} x={state.x:.3f} y={state.y:.3f} "
                        f"页={panel.page + 1} 目标={state.target} | {state.message}",
                        flush=True,
                    )
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            stop_event.set()
            runner.poller.join(timeout=2.0)
        return 0

    state.message = "正在加载手部模型…"
    if args.new_desktop:
        # 先切到新桌面再建窗口：窗口归属创建时所在的桌面，原来的桌面不受影响。
        if open_new_desktop():
            print("[桌面] 已新建虚拟桌面并切过去（Win+Ctrl+←/→ 可切回）。", file=sys.stderr)
        else:
            print("[桌面] 新建虚拟桌面失败，改在当前桌面启动。", file=sys.stderr)
    if not panel.show():
        stop_event.set()
        return 1
    runner.poller.start()
    try:
        panel.run(runner.tick)
    finally:
        stop_event.set()
        runner.poller.join(timeout=2.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
