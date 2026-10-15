"""命令行入口。

为什么没有 GUI：MVP 阶段一切围绕"先把引擎跑通、先 safe-mode 验证"。配置走
``config.json`` + 少量 CLI flags，交互走 OpenCV 调试窗口和全局热键，不引入
PySide6，也就不会有打包体积和线程-事件循环耦合的问题。

用法：
    python -m handmouse                    # 启动，默认暂停，按 M 接管鼠标
    python -m handmouse --safe-mode        # 不注入任何输入，只跑状态机
    python -m handmouse --diagnostic-seconds 20
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from math import hypot
from types import SimpleNamespace

from handmouse import __version__, debug_view
from handmouse.config import AppConfig, ConfigStore
from handmouse.hotkeys import HotkeyMonitor
from handmouse.launch import LaunchDispatcher
from handmouse.mouse import MouseController, NullMouseBackend, Win32MouseBackend
from handmouse.pipeline import GesturePipeline
from handmouse.screen_map import ScreenMap, dpi_state, system_dpi_scale

LOGGER = logging.getLogger(__name__)


def _arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="handmouse",
        description="HandMouse 视觉手势鼠标（引擎层移植自 Vimoose，MIT）",
    )
    parser.add_argument("--config", metavar="PATH", help="配置文件路径，默认仓库根目录 config.json")
    parser.add_argument("--camera", type=int, help="摄像头序号，覆盖 config")
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        help="不注入任何系统输入，只跑状态机与调试窗口",
    )
    parser.add_argument(
        "--auto-control",
        action="store_true",
        help="启动即接管鼠标（默认先暂停，按 M 或 Ctrl+Alt+G 才接管）",
    )
    parser.add_argument("--no-motion", action="store_true", help="关闭轨迹手势（DTW）")
    parser.add_argument("--no-snap", action="store_true", help="关闭控件磁吸（需要 handmouse[uia]）")
    parser.add_argument(
        "--multi-monitor",
        action="store_true",
        help="按虚拟桌面映射坐标（覆盖 config.multi_monitor）",
    )
    parser.add_argument(
        "--virtual-desk",
        action="store_true",
        help="实验：单屏也走 MOUSEEVENTF_VIRTUALDESK，用于多屏机器上验证",
    )
    parser.add_argument(
        "--margin",
        type=float,
        nargs=2,
        metavar=("X", "Y"),
        help="活动区左右/上下内缩比例，覆盖 config.pointer_margin_*",
    )
    parser.add_argument(
        "--diagnostic-seconds",
        type=float,
        default=0.0,
        help="运行指定秒数后打印指标并退出",
    )
    parser.add_argument(
        "--list-bindings",
        action="store_true",
        help="打印当前轨迹动作绑定与 LAUNCH 白名单后退出",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.camera is not None:
        config.camera_index = args.camera
    if args.margin is not None:
        config.pointer_margin_x, config.pointer_margin_y = args.margin
    if args.multi_monitor:
        config.multi_monitor = True
    if args.virtual_desk:
        config.virtual_desk_experimental = True
    if args.no_snap:
        config.cursor_snap_enabled = False
    if args.no_motion:
        config.motion_gestures_enabled = False
    config.validate()
    return config


def _pinch_ratio(detection) -> float | None:
    """从最新检测里现算捏合比，只用于调试窗口显示。"""

    if detection is None or not detection.observations:
        return None
    points = detection.observations[0].landmarks
    if len(points) != 21:
        return None
    palm_width = max(hypot(points[5].x - points[17].x, points[5].y - points[17].y), 1e-4)
    return hypot(points[4].x - points[8].x, points[4].y - points[8].y) / palm_width


def _print_bindings(config: AppConfig) -> None:
    print(f"轨迹动作绑定（画完形状后捏合确认才执行）：")
    for name, spec in config.motion_bindings.items():
        print(f"  {name:<8} {spec['kind']:<7} → {spec['target']}")
    print("LAUNCH 白名单（别名 → 命令）：")
    for alias, command in config.launch_whitelist.items():
        print(f"  {alias:<10} → {command}")


def main(argv: list[str] | None = None) -> int:
    args = _arguments(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    store = ConfigStore(args.config)
    config = _apply_cli_overrides(store.load(), args)

    if args.list_bindings:
        _print_bindings(config)
        return 0

    screen_map = ScreenMap(
        multi_monitor=config.multi_monitor,
        virtual_desk_experimental=config.virtual_desk_experimental,
    )
    safe_mode = bool(args.safe_mode) or sys.platform != "win32"

    try:
        backend = NullMouseBackend(screen_map) if safe_mode else Win32MouseBackend(config, screen_map)
    except RuntimeError as error:
        print(f"平台不受支持：{error}")
        return 2

    mouse = MouseController(backend)
    dispatcher = LaunchDispatcher(config, dry_run=safe_mode)
    pipeline = GesturePipeline(config, mouse, dispatcher=dispatcher)

    state = SimpleNamespace(
        quit=False,
        toggle=False,
        emergency=False,
        started=False,
    )
    hotkeys = HotkeyMonitor(
        lambda: setattr(state, "toggle", True),
        lambda: setattr(state, "emergency", True),
        lambda: setattr(state, "quit", True),
    )
    hotkeys.start()

    print(f"HandMouse {__version__}  后端={type(backend).__name__}  {screen_map.describe()}")
    print(f"配置={store.path}  安全模式={'开' if safe_mode else '关'}")
    if not safe_mode:
        store.save(config)
    _print_bindings(config)
    print("按键：M 暂停/恢复(接管) | ESC/Q 退出 | Ctrl+Alt+G 接管 | Ctrl+Alt+Esc 急停 | Ctrl+Alt+Q 退出")

    if not pipeline.start():
        print("流水线启动失败，详见上方日志。")
        hotkeys.stop()
        mouse.shutdown()
        return 3

    state.started = True
    engine_started_paused = not args.auto_control
    if engine_started_paused:
        pipeline.set_paused(True)
        print("默认未接管鼠标：按 M 或 Ctrl+Alt+G 开始接管。")

    scale = system_dpi_scale()
    deadline = (
        time.monotonic() + args.diagnostic_seconds if args.diagnostic_seconds > 0 else None
    )
    exit_code = 0
    try:
        while not state.quit:
            _apply_pending_hotkeys(state, pipeline)

            frame_packet, detection = pipeline.get_preview()
            if frame_packet is None:
                time.sleep(0.01)
            else:
                snapshot = pipeline.snapshot()
                view = debug_view.build_view(
                    frame_packet.frame,
                    snapshot,
                    detection,
                    pipeline.motion_outcome,
                    screen_map,
                    config,
                    _pinch_ratio(detection),
                )
                key = debug_view.present(debug_view.WINDOW_NAME, view, scale)
                if not _handle_key(key, state, pipeline, config):
                    break

            if deadline is not None and time.monotonic() >= deadline:
                _print_diagnostics(pipeline.snapshot())
                break
    except KeyboardInterrupt:
        print("\n收到中断，正在安全退出。")
    finally:
        pipeline.stop()
        hotkeys.stop()
        mouse.shutdown()
        debug_view.close()

    print(f"退出。DPI={dpi_state()}")
    return exit_code


def _apply_pending_hotkeys(state: SimpleNamespace, pipeline: GesturePipeline) -> None:
    if state.emergency:
        state.emergency = False
        print("[急停] 已释放按键并暂停控制。")
        pipeline.emergency_stop()
    if state.toggle:
        state.toggle = False
        pipeline.toggle_paused()
        print(f"[控制] {'已接管' if not pipeline.engine.paused else '已暂停'}")


def _handle_key(
    key: int,
    state: SimpleNamespace,
    pipeline: GesturePipeline,
    config: AppConfig,
) -> bool:
    """返回 False 表示请求退出。"""

    if key in (-1, 255):
        return True
    if key in (27, ord("q"), ord("Q")):
        return False
    if key in (ord("m"), ord("M")):
        pipeline.toggle_paused()
        print(f"[控制] {'已接管' if not pipeline.engine.paused else '已暂停'}")
    elif key in (ord("r"), ord("R")):
        pipeline.motion_engine.reset()
        pipeline.engine.filter.reset()
        print("[状态] 已重置滤波器与轨迹手势状态")
    return True


def _print_diagnostics(snapshot) -> None:
    metrics = snapshot.metrics
    print(
        "诊断完成 "
        f"capture_fps={metrics.capture_fps:.1f} "
        f"inference_fps={metrics.inference_fps:.1f} "
        f"latency_ms={metrics.latency_ms:.1f} "
        f"latency_p50_ms={metrics.latency_p50_ms:.1f} "
        f"latency_p95_ms={metrics.latency_p95_ms:.1f} "
        f"hand_detection_rate={metrics.hand_detection_rate:.1f} "
        f"dropped_frames={metrics.dropped_frames} "
        f"cpu_percent={metrics.cpu_percent:.1f} "
        f"memory_mb={metrics.memory_mb:.1f} "
        f"error={snapshot.camera_error or 'none'}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
